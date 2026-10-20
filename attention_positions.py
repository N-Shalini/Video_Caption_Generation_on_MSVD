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
per_video = []
for r in recs:
    f = np.load(feat_dir / f"{r['video_id']}.npy")
    if not is_valid_feature_shape(f.shape):
        raise ValueError(f"Unexpected feature shape for {r['video_id']}: {f.shape}")
    _, hist = greedy_decode(model, features=f, id_to_word=id_to_word, max_length=30)
    if hist:
        per_video.append(np.mean([w.numpy() for w in hist], axis=0))
M = np.stack(per_video)
N = M.shape[1]
frames = M.reshape(len(M), 8, 49).sum(axis=2) if N == 392 else M
print(ckpt, "| videos:", len(M), "| positions:", N)
print("mean attention mass per frame :", np.round(frames.mean(axis=0), 3))
print("std of that mass across videos:", np.round(frames.std(axis=0), 3))
if N == 392:
    print("mean spatial map (7x7, summed over frames):")
    print(np.round(M.mean(axis=0).reshape(8, 49).sum(axis=0).reshape(7, 7), 3))
Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
S = Mn @ Mn.T
print("mean pairwise cosine similarity between videos' attention maps: %.3f"
      % ((S.sum() - len(S)) / (len(S) * (len(S) - 1))))