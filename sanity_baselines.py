import collections
import json
import random

import numpy as np

from metrics import evaluate_predictions

ANN = "data/annotations/"
FEAT = "data/features/vgg16_spatial/"
train = json.load(open(ANN + "msvd_train.json", encoding="utf-8"))
test = json.load(open(ANN + "msvd_test.json", encoding="utf-8"))
refs = {r["video_id"]: r["caption"] for r in test}


def pooled(vid):
    return np.load(FEAT + vid + ".npy", allow_pickle=False).mean(axis=(0, 2, 3))


results = {}

const = collections.Counter(c.lower().strip() for r in train for c in r["caption"]).most_common(1)[0][0]
results["constant caption"] = {r["video_id"]: const for r in test}

rng = random.Random(0)
all_caps = [c for r in train for c in r["caption"]]
results["random train caption"] = {r["video_id"]: rng.choice(all_caps) for r in test}

tf = np.stack([pooled(r["video_id"]) for r in train])
tf /= np.linalg.norm(tf, axis=1, keepdims=True) + 1e-8
nn = {}
for r in test:
    q = pooled(r["video_id"])
    q /= np.linalg.norm(q) + 1e-8
    nn[r["video_id"]] = train[int(np.argmax(tf @ q))]["caption"][0]
results["nearest-neighbour train caption"] = nn

keys = ("BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr")
print("%-34s" % "baseline" + "".join("%9s" % k for k in keys))
for name, hyps in results.items():
    s = evaluate_predictions(refs, hyps)
    print("%-34s" % name + "".join("%9.2f" % s[k] for k in keys))