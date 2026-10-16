import json
import numpy as np
from generate_caption import load_model, greedy_decode
from dataset import tokenize

model, config, vocab, id_to_word, ckpt = load_model(checkpoint_path="checkpoints/temporal_s42_best.pt")
val = json.load(open("data/annotations/msvd_val.json", encoding="utf-8"))
train = json.load(open("data/annotations/msvd_train.json", encoding="utf-8"))

toks = [w for r in train for c in r["caption"] for w in tokenize(c)]
print("train <unk> token rate: %.2f%%" % (100 * sum(w not in vocab for w in toks) / len(toks)))
after_article = 0
for r in train:
    for c in r["caption"]:
        t = tokenize(c)
        after_article += sum(1 for p, w in zip(t, t[1:]) if p in ("a", "an", "the") and w not in vocab)
print("train: rare word (<unk>) directly after a/an/the:", after_article)

ends = {"a", "an", "the", "of", "with", "on", "in", "to", "and"}
for suppress in (True, False):
    caps = []
    for r in val:
        f = np.load("data/features/vgg16_spatial/%s.npy" % r["video_id"])
        c, _ = greedy_decode(model, features=f, id_to_word=id_to_word, max_length=30,
                             repetition_penalty=1.0, suppress_unk=suppress)
        caps.append(c)
    bad = [c for c in caps if c.split() and c.split()[-1] in ends]
    print("\nsuppress_unk=%s | captions ending in a function word: %d/%d" % (suppress, len(bad), len(caps)))
    print("  unique words:", len({w for c in caps for w in c.split()}))
    for c in caps[:8]:
        print("   ", c)