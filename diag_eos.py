import json
import numpy as np
import torch
import torch.nn.functional as F
from generate_caption import load_model
from dataset import tokenize

ends = {"a", "an", "the", "of", "with", "on", "in", "to", "and"}

# Part A: how often do the REFERENCE captions end in a function word?
for split in ("train", "val"):
    recs = json.load(open("data/annotations/msvd_%s.json" % split, encoding="utf-8"))
    toks = [tokenize(c) for r in recs for c in r["caption"]]
    toks = [t for t in toks if t]
    n = len(toks)
    print("%s: %d captions | end in function word: %.2f%% | <=3 tokens: %.2f%% | mean length: %.2f" % (
        split, n,
        100 * sum(t[-1] in ends for t in toks) / n,
        100 * sum(len(t) <= 3 for t in toks) / n,
        sum(len(t) for t in toks) / n))
train = json.load(open("data/annotations/msvd_train.json", encoding="utf-8"))
ex = [c for r in train for c in r["caption"] if tokenize(c) and tokenize(c)[-1] in ends]
print("train examples ending in a function word:", ex[:10])

# Part B: what does the model think at the point where it stops after "a"?
model, config, vocab, id_to_word, ckpt = load_model(checkpoint_path="checkpoints/temporal_s42_best.pt")

@torch.inference_mode()
def trace(feats, max_len=30):
    enc = model.encoder(torch.from_numpy(feats).float().unsqueeze(0))
    h = enc.mean(1)
    c = torch.zeros_like(h)
    W1 = model.decoder.attention.W1(enc)
    tok, words, steps = 1, [], []
    for _ in range(max_len - 1):
        e = model.decoder.embedding(torch.tensor([tok]))
        ctx, _ = model.decoder.attention(enc, h, precomputed_W1=W1)
        h, c = model.decoder.lstm_cell(torch.cat([e, ctx], 1), (h, c))
        p = F.softmax(model.decoder.output_projection(h).squeeze(0), -1)
        top = torch.topk(p, 5)
        steps.append([(id_to_word[int(i)], round(float(v), 2)) for v, i in zip(top.values, top.indices)])
        tok = int(p.argmax())
        if tok == 2:
            break
        words.append(id_to_word[tok])
    return words, steps

val = json.load(open("data/annotations/msvd_val.json", encoding="utf-8"))
shown = 0
for r in val:
    f = np.load("data/features/vgg16_spatial/%s.npy" % r["video_id"])
    words, steps = trace(f)
    if words and words[-1] in ends and shown < 6:
        print("\n", " ".join(words), "-> next-token top5:", steps[-1])
        shown += 1