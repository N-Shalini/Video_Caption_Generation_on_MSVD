import json
import sys

import numpy as np

from dataset import feature_dir_for, is_valid_feature_shape
from generate_caption import greedy_decode, load_model

model, config, vocab, id_to_word, checkpoint = load_model(checkpoint_path=sys.argv[1])
feat_type = config.get("feature_type", checkpoint.get("feature_type", "vgg16_spatial"))
feat_dir = feature_dir_for(feat_type)
recs = json.load(open("data/annotations/msvd_val.json", encoding="utf-8"))
within = []
for r in recs:
    f = np.load(feat_dir / f"{r['video_id']}.npy")
    if not is_valid_feature_shape(f.shape):
        raise ValueError(f"Unexpected feature shape for {r['video_id']}: {f.shape}")
    _, hist = greedy_decode(model, features=f, id_to_word=id_to_word, max_length=30)
    if len(hist) < 4:
        continue
    A = np.stack([h.numpy() for h in hist[:-1]])
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    S = A @ A.T
    n = len(S)
    within.append((S.sum() - n) / (n * (n - 1)))
print(sys.argv[1], "| videos:", len(within))
print("mean cosine similarity between one video's attention maps at different words: %.3f" % np.mean(within))