import json
import sys

import numpy as np

from dataset import feature_dir_for, is_valid_feature_shape
from generate_caption import greedy_decode, load_model

ckpt = sys.argv[1]
model, config, vocab, id_to_word, checkpoint = load_model(checkpoint_path=ckpt)
feat_type = config.get("feature_type", checkpoint.get("feature_type", "vgg16_spatial"))
feat_dir = feature_dir_for(feat_type)
recs = json.load(open("data/annotations/msvd_val.json", encoding="utf-8"))
maxw, ent, n = [], [], 0
for r in recs:
    f = np.load(feat_dir / f"{r['video_id']}.npy")
    if not is_valid_feature_shape(f.shape):
        raise ValueError(f"Unexpected feature shape for {r['video_id']}: {f.shape}")
    _, hist = greedy_decode(model, features=f, id_to_word=id_to_word, max_length=30)
    for w in hist:
        p = w.numpy().astype(np.float64)
        n = len(p)
        maxw.append(p.max())
        ent.append(-(p * np.log(p + 1e-12)).sum() / np.log(n))
print(ckpt)
print("positions N = %d | uniform weight = %.3f | steps analysed = %d" % (n, 1.0 / n, len(maxw)))
print("mean max attention weight = %.3f | mean normalised entropy = %.3f (1.0 = perfectly uniform)" % (np.mean(maxw), np.mean(ent)))