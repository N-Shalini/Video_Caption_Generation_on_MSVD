# Lightweight video captioning on MSVD

A small, reproducible video-captioning code base (frozen frame features, a single-layer
LSTM decoder with additive attention) used for a controlled study on the Microsoft Video
Description corpus (MSVD). Everything runs on a laptop CPU.

The study compares four frame encoders and two frozen feature extractors (VGG16 and CLIP
ViT-B/32) under one fixed evaluation protocol, with three seeds per configuration.

**Main finding.** Swapping frozen VGG16 features for frozen CLIP ViT-B/32 features raises
CIDEr by about 35 points for every encoder. Changing the encoder (learned frame attention,
mean pooling, BiGRU, spatial attention) moves the score by at most a few points, and
differences between encoders are mostly within seed noise.

## Results

Test set (670 videos), beam search with width 5, standard COCO caption metrics (x100),
mean +/- standard deviation over seeds 42, 43 and 44.

| Features | Encoder | CIDEr | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | `temporal` (learned frame attention) | 101.81 +/- 1.21 | 58.94 +/- 1.50 | 38.41 +/- 0.47 | 76.17 +/- 0.31 |
| CLIP ViT-B/32 | `temporal_bigru` | 99.57 +/- 2.67 | 59.53 +/- 0.33 | 37.91 +/- 0.47 | 75.74 +/- 0.43 |
| CLIP ViT-B/32 | `temporal_mean` (no attention) | 99.24 +/- 2.00 | 58.45 +/- 0.83 | 37.55 +/- 0.40 | 75.42 +/- 0.24 |
| VGG16 | `temporal` | 67.22 +/- 2.90 | 46.66 +/- 0.62 | 31.70 +/- 0.33 | 68.72 +/- 0.22 |
| VGG16 | `temporal_mean` | 63.30 +/- 3.37 | 46.21 +/- 1.56 | 31.31 +/- 0.40 | 68.22 +/- 0.50 |
| VGG16 | `spatiotemporal_attention` | 62.10 +/- 1.56 | 43.11 +/- 0.59 | 30.01 +/- 0.29 | 66.12 +/- 0.57 |
| VGG16 | `temporal_bigru` | 61.77 +/- 0.45 | 45.17 +/- 1.26 | 30.81 +/- 0.24 | 67.20 +/- 0.36 |

Greedy and beam-3 results and all per-seed values are in [`docs/`](docs/)
(`summary_final.txt`, `per_seed_final.txt`).

Reference points on the same test set (standard COCO toolkit, one number per system):

| Baseline | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| Random training caption | 4.8 | 13.3 | 37.9 | 2.2 |
| Constant caption ("a man is playing a guitar") | 21.5 | 23.3 | 59.6 | 15.3 |
| Nearest-neighbour training caption, VGG16 features | 13.5 | 20.2 | 45.8 | 23.0 |
| Nearest-neighbour training caption, CLIP features | 17.4 | 24.0 | 50.8 | 34.8 |

Cost (laptop CPU, 4 threads; single measurements with roughly 10-15% run-to-run variation):

| Component | Parameters | Time per video |
|---|---|---|
| VGG16 convolutional features (frozen) | 14.7M | 1.67 s |
| CLIP ViT-B/32 image encoder (frozen) | 87.8M | 0.86 s |
| Captioner (`temporal`, either feature type) | 7.49M | greedy about 7-8 ms, beam 5 about 150-165 ms |

## Evaluation protocol

The protocol is fixed and shared by every run.

* **Data:** MSVD, standard split of 1,200 training, 100 validation and 670 test videos,
  about 40 English reference captions per video. Vocabulary is built from training captions
  only (minimum frequency 2). Test and validation references are never filtered.
* **Frames and features:** 8 uniformly sampled frames per video. The same frame indices are
  used for both feature extractors. Features are frozen and cached.
* **Training:** settings in `config.json` are shared by all encoders and were not tuned per
  encoder. Up to 40 epochs, 3 randomly resampled captions per video per epoch, learning
  rate reduced on plateau, early stopping with patience 6.
  **The checkpoint is selected by validation loss.** Seeds 42, 43 and 44.
* **Decoding (fixed in advance, no per-model tuning):** greedy, beam 3 and beam 5, each with
  length penalty 0.7, no repetition penalty, no n-gram blocking. `<pad>`, `<bos>` and
  `<unk>` are suppressed. Beam 5 is the headline setting.
* **Metrics:** `pycocoevalcap` (PTB tokenizer, BLEU-1 to 4, METEOR 1.5, ROUGE-L, CIDEr),
  corpus level, against all references, reported x100. Java is required.
* **Test set use:** each trained model is evaluated on the test set once, with the fixed
  decoding settings above.

Why the protocol is spelled out: on the same predictions, BLEU-4 ranges from 8 (first
reference only) to 47 (all references), and METEOR from 32 (standard toolkit) to 62
(NLTK multi-reference maximum). A constant caption already reaches BLEU-1 66 and METEOR 23.
See `protocol_sensitivity.py` and `sanity_baselines.py`.

## Models

| Encoder | Description |
|---|---|
| `temporal` | Spatial grid mean-pooled per frame, then Linear + LayerNorm + ReLU + Dropout; the decoder attends over the 8 frame vectors (additive attention) |
| `temporal_mean` | Same encoder, but the context vector is the plain mean of the frame vectors at every step (no attention parameters). Select with `--attention-type mean` |
| `temporal_bigru` | Mean and max pooling, projection, bidirectional GRU with a residual connection; additive attention over frames |
| `spatiotemporal_attention` | All 8 x 49 region vectors with learned frame and position embeddings; additive attention over 392 positions (VGG16 features only) |

The decoder is a single-layer LSTM cell (embedding 256, hidden 512, dropout 0.3) with
additive attention and scheduled sampling during training. Feature types: `vgg16_spatial`
(512 x 7 x 7 per frame) and `clip_vitb32` (one L2-normalised 512-d embedding per frame,
stored as 512 x 1 x 1).

## Repository layout

| Path | Purpose |
|---|---|
| `extract_features.py` | VGG16 feature extraction (resumable) |
| `extract_features_clip.py` | CLIP ViT-B/32 feature extraction (resumable) |
| `build_vocab.py` | Vocabulary from the training captions |
| `dataset.py`, `model.py` | Data loading, encoders, attention and decoder |
| `train_spatiotemporal.py` | Training for all encoders (resume, seeds, run names) |
| `generate_caption.py` | Greedy and beam decoding, caption generation for one video |
| `evaluate_test_set.py` | Test-set evaluation with fixed decoding settings |
| `tune_decoding.py` | Decoding grid search on the validation set (kept for reference; see disclosures) |
| `metrics.py`, `metrics_legacy.py` | Standard COCO metrics; legacy custom scorers (not used) |
| `run_queue.ps1` | Runs train + evaluate for a list of jobs (PowerShell) |
| `summarize_runs.py`, `per_seed.py` | Tables over seeds |
| `sanity_baselines.py`, `protocol_sensitivity.py` | Trivial baselines; scoring-protocol comparison |
| `attention_stats.py`, `attention_positions.py`, `attention_dynamics.py`, `visualize_attention.py` | Attention analyses and figures |
| `efficiency.py` | Parameters, latency and memory |
| `ablate_eos_rule.py`, `check_blocklist.py` | Optional end-of-sentence blocklist rule (off by default) |
| `configs/` | Fixed decoding configurations |
| `tests/` | Unit tests |
| `CHANGES.md` | Development log |

Checkpoints, results, logs and data are not tracked by git.

## Setup

Python 3.11 was used, on Windows, with CPU-only PyTorch.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install opencv-python pillow torchvision numpy
```

Java (version 8 or later) must be on the PATH: the COCO tokenizer and METEOR need it.
Check with `java -version`. The first CLIP extraction downloads the OpenAI ViT-B/32
weights (about 350-600 MB) through `open_clip`.

## Data

The MSVD videos and captions are not redistributed here. The corpus is described in
Chen and Dolan (2011). Place the files as follows:

```
data/raw/YouTubeClips/                 the 1,970 video clips
data/annotations/msvd_train.json       1,200 videos
data/annotations/msvd_val.json         100 videos
data/annotations/msvd_test.json        670 videos
```

Each annotation file is a JSON list of records of the form
`{"video_id": "...", "video": "<file name>.avi", "caption": ["...", "..."]}` with all
English reference captions of that video. A video must appear in exactly one split.

## Reproducing the results

All commands are run from the repository root. PowerShell is assumed.

**1. Features**

```powershell
python extract_features.py           # VGG16 -> data/features/vgg16_spatial   (about 1 s per video)
python extract_features_clip.py      # CLIP  -> data/features/clip_vitb32     (about 0.9 s per video)
python build_vocab.py                # -> data/annotations/vocab.json
```

Extract the VGG16 features first: the CLIP extractor reads the sampled frame indices from
`data/features/vgg16_spatial/manifest.jsonl` so both use the same frames.

**2. Train one model**

```powershell
python train_spatiotemporal.py --encoder-type temporal --seed 42 --run-name temporal_v2_s42
```

Useful options: `--encoder-type {temporal, temporal_bigru, spatiotemporal_attention}`,
`--attention-type {additive, mean}`, `--feature-type {vgg16_spatial, clip_vitb32}`,
`--seed`, `--run-name`, `--epochs`, `--resume`. Hyperparameters live in `config.json`
and should not be edited between runs. Checkpoints go to `checkpoints/{run_name}_best.pt`
and `_last.pt`; the history and the resolved configuration go to `results/`.

**3. Evaluate it (test set, once, fixed decoding)**

```powershell
python evaluate_test_set.py --checkpoint checkpoints\temporal_v2_s42_best.pt --decoding-config configs\fixed_beam5.json
```

This prints the headline beam-5 result and a controlled greedy / beam-3 / beam-5
comparison, and writes `results/{checkpoint}_test_evaluation.json` and the predictions.
The checkpoint's feature type is read from the checkpoint, so CLIP models need no extra flag.

**4. The full grid**

`run_queue.ps1` trains and evaluates a list of `encoder:seed[:attention[:features]]` jobs
one after another and resumes interrupted runs. It expects the decoding configuration at
`results\fixed_beam5_config.json`:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
Copy-Item configs\fixed_beam5.json results\fixed_beam5_config.json

# VGG16 features
.\run_queue.ps1 -Jobs temporal:42,temporal:43,temporal:44,temporal_bigru:42,temporal_bigru:43,temporal_bigru:44,spatiotemporal_attention:42,spatiotemporal_attention:43,spatiotemporal_attention:44,temporal:42:mean,temporal:43:mean,temporal:44:mean

# CLIP features
.\run_queue.ps1 -Jobs temporal:42:additive:clip_vitb32,temporal:43:additive:clip_vitb32,temporal:44:additive:clip_vitb32,temporal:42:mean:clip_vitb32,temporal:43:mean:clip_vitb32,temporal:44:mean:clip_vitb32,temporal_bigru:42:additive:clip_vitb32,temporal_bigru:43:additive:clip_vitb32,temporal_bigru:44:additive:clip_vitb32

python summarize_runs.py
python per_seed.py
```

On the laptop CPU used here (Intel Core i7-1165G7, 4 cores, no GPU) one training run takes
roughly 1.3-2.5 hours (VGG16 `spatiotemporal_attention` is the slowest), and the 24 runs
take about 35 hours in total.

**5. Analyses**

```powershell
python sanity_baselines.py
python protocol_sensitivity.py results\temporal_v2_s43_best_test_predictions.json
python attention_stats.py checkpoints\temporal_v2_s42_best.pt
python attention_positions.py checkpoints\spatiotemporal_attention_v2_s42_best.pt
python attention_dynamics.py checkpoints\spatiotemporal_attention_v2_s42_best.pt
python visualize_attention.py --checkpoint checkpoints\temporal_v2_s42_best.pt --video-ids <video_id> --decoding greedy
$env:OMP_NUM_THREADS = "4"; python efficiency.py --checkpoint checkpoints\temporal_v2_s42_best.pt
```

Set `OMP_NUM_THREADS` before timing anything; without it, latency measurements were
off by a factor of 2.5 in our runs. Close other programs during timing runs.

## Tests

```powershell
python -m pytest tests -q
```

The tests cover the models and decoding (including beam search against greedy decoding),
caption resampling, the metrics wrapper, feature-type handling and the end-of-sentence
rule.

## Disclosures and limitations

* **Scope:** one dataset (MSVD), 8 frames per video, frozen features, one CLIP model,
  three seeds per configuration. With three seeds, differences of a few CIDEr points are
  not conclusive; per-seed values are provided.
* **Shared hyperparameters:** all encoders use the same settings, so none was tuned for
  its own architecture.
* **Epoch cap:** a few CLIP runs reached their best validation loss at or near the final
  (40th) epoch, so the cap may have limited them slightly.
* **Selection rule and decoding:** an earlier choice of checkpoint by validation CIDEr
  (100 videos) proved unstable, and per-model decoding tuning on the validation set chose a
  setting that scored lower on test than a fixed beam-5 setting. The protocol above
  (selection by validation loss, fixed decoding) was adopted after those observations; it
  is documented in `CHANGES.md`. `tune_decoding.py` is kept for transparency.
* **End-of-sentence rule:** greedy decoding often ends captions after words such as "a"
  because of a miscalibrated end-of-sequence probability. An optional rule that forbids
  ending after such words is implemented (`ablate_eos_rule.py`); it is off in all reported
  headline results.
* **CLIP pretraining data** is not public, so overlap with MSVD videos cannot be ruled out.
* **Superseded results:** an earlier version of this project (repository `video-captioning`)
  reported much higher scores with a non-standard evaluation. Those numbers are superseded
  and are not reproducible under the protocol above.
* **Comparisons with published systems** use different features, decoders and training and
  are only indicative.

## Provenance

The code behind the reported numbers is tagged `results-v1`. The VGG16 grid was produced
before the mean-context, CLIP and tooling changes; regression checks confirmed that the
later code reproduces the earlier results exactly (same test CIDEr for a re-evaluated
checkpoint and the same first-epoch training loss). See `git log` and `CHANGES.md`.

## Citing and licence

If you use this code, please cite the repository. The datasets and pretrained models keep
their own licences and terms (MSVD, VGG16 weights from torchvision, OpenAI CLIP weights via
`open_clip`). No licence file has been added yet, so all rights remain with the authors
until one is.
