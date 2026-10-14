# Changes & Assumptions: Rigorous & Reproducible Evaluation Pipeline

## 1. Overview of Changed Files

### [`model.py`](file:///c:/dev/video/model.py)
- **Task 1: Spatial-Temporal Attention & Generalization**:
  - Renamed the mean/max-pooled BiGRU encoder to `TemporalBiGRUFeatureEncoder`. Maintained `SpatioTemporalFeatureEncoder = TemporalBiGRUFeatureEncoder` as an alias and preserved all internal sub-module names (`spatial_projection`, `temporal_gru`, `spatial_norm`, etc.) ensuring full backward-compatibility with existing checkpoints.
  - Implemented `SpatioTemporalAttentionFeatureEncoder`: projects all $8 \times 49 = 392$ region vectors ($512 \rightarrow \text{hidden\_dim}$), adds learned frame-index embeddings ($8 \times \text{hidden\_dim}$) and spatial-position embeddings ($49 \times \text{hidden\_dim}$), followed by `LayerNorm`, `ReLU`, and `Dropout`. Output shape: `[B, 392, hidden_dim]`.
  - Generalized attention from temporal frames to any number of positions $N$ in `AdditiveAttention` (with `AdditiveTemporalAttention` alias). Precomputes $W_1(\text{encoder\_out})$ once per sequence and accepts optional `precomputed_W1`.
  - Added `reshape_attention_weights(weights, num_frames=8, height=7, width=7)` helper reshaping weights of shape `[B, L, 392]` to `[B, L, 8, 7, 7]`.
  - Updated `VideoCaptionModel` to support `encoder_type in ("spatiotemporal", "temporal_bigru")`, `"spatiotemporal_attention"`, and `"temporal"`.

### [`dataset.py`](file:///c:/dev/video/dataset.py)
- **Task 3: Full Caption Utilization**:
  - Implemented `MSVDDataset.resample(epoch)`: for training, samples $\min(k, \text{len(captions)})$ captions per video without replacement using dedicated `random.Random(seed * 1000003 + epoch)`.
  - Added support for `captions_per_video = 0` meaning all reference captions are utilized.
  - Handled $k=1$ via the same deterministic sampling mechanism.
  - Removed `random.choice` from `__getitem__`.
  - For validation with `captions_per_video = 0`, expands all reference captions into `self.samples` for complete, deterministic validation loss computation.

### [`config.json`](file:///c:/dev/video/config.json)
- **Task 6: Authoritative Configuration**:
  - Included all hyperparameters: `encoder_type`, `run_name`, `device`, `num_workers`, `seed`, `feature_type`, `num_frames`, `feature_channels`, `feature_height`, `feature_width`, `min_word_frequency`, `max_caption_length`, `embedding_dim`, `hidden_dim`, `attention_dim`, `dropout`, `batch_size`, `epochs`, `learning_rate`, `weight_decay`, `early_stopping_patience`, `gradient_clip_norm`, `captions_per_video`, `selection_metric`, `scheduled_sampling` (`max_prob`, `increment_per_epoch`), `scheduler` (`factor`, `patience`), and `suppress_unk`.

### [`train_spatiotemporal.py`](file:///c:/dev/video/train_spatiotemporal.py)
- **Tasks 4 & 6: Checkpoint Selection, Validation Loss & Config Resolution**:
  - Command-line arguments default to `None` and override `config.json` only when explicitly passed. Added `--encoder-type` and `--run-name`.
  - Saves resolved config to `results/{run_name}_config_resolved.json` and embeds it into checkpoints.
  - Saves both best (`checkpoints/{run_name}_best.pt`) and last (`checkpoints/{run_name}_last.pt`) checkpoints, and saves history to `results/{run_name}_history.json`.
  - Supports `--max-train-videos` and `--max-val-videos` dev smoke flags, recorded in the resolved config.
  - Calls `train_dataset.resample(epoch)` at the start of every epoch.
  - Evaluates validation loss over all reference captions (`captions_per_video=0`).
  - Evaluates greedy decoding validation using standard COCO metrics (`CIDEr`, `BLEU-4`, `METEOR`, `ROUGE-L`) on all validation videos.
  - Restores `model.train()` after validation decoding.
  - Added `selection_metric` (`"cider"` default, or `"loss"`): controls checkpoint selection, early stopping patience, and `ReduceLROnPlateau` mode (`"max"` for CIDEr, `"min"` for loss).

### [`generate_caption.py`](file:///c:/dev/video/generate_caption.py)
- **Task 7: Standard Beam Search & Token Suppression**:
  - Standard beam search pruning: keeps $k$ live hypotheses pruned by raw cumulative log-probability.
  - On `<eos>`, moves hypothesis to finished list scored by $\frac{\text{cum\_log\_prob}}{\text{length}^{\text{length\_penalty}}}$, where length counts all generated tokens including `<eos>`.
  - Early stopping bound: stops when length_penalty is 0.0 and no live hypothesis can beat the best finished score; otherwise searches to `max_length`.
  - Falls back to best live hypothesis with same length normalization if nothing finished.
  - Suppresses `<pad>` (0) and `<bos>` (1) in both greedy and beam decoding; suppresses `<unk>` (3) when `suppress_unk=True`.
  - Updated defaults: `repetition_penalty = 1.0`, `ngram_block_size = 0`.
  - Added optional `encoder_out` parameter to reuse precomputed encoder representations for fast evaluation.

### [`metrics.py`](file:///c:/dev/video/metrics.py)
- **Task 8: Standard COCO Caption Metrics**:
  - Validates `shutil.which("java")`, raising a descriptive error if Java is missing.
  - Implemented `evaluate_predictions(refs, hyps)` using `PTBTokenizer`, `Bleu(4)`, `Meteor()`, `Rouge()`, and `Cider()` from `pycocoevalcap`.
  - Returns scores scaled by 100 and rounded to 2 decimals, includes `"scale": "x100"`, and provides per-video scores under `"per_video"`.
  - Retained `compute_all_metrics(records, predictions)` delegating to `evaluate_predictions` while preserving diversity and repetition statistics (`distinct_1`, `distinct_2`, `repeated_bigram_caption_rate`, etc.).

### [`metrics_legacy.py`](file:///c:/dev/video/metrics_legacy.py)
- **Task 8: Legacy Scorer Archive**:
  - Moved old custom implementations (`lcs`, `sentence_rouge_l`, `CIDErScorer`) to this file for backward-compatibility.

### [`tune_decoding.py`](file:///c:/dev/video/tune_decoding.py)
- **Task 2: Validation-Only Decoding Tuning**:
  - Implemented grid search strictly on the validation set:
    - `beam_width` $\in \{1, 3, 5\}$
    - `length_penalty` $\in \{0.0, 0.5, 0.7, 1.0\}$
    - `repetition_penalty` $\in \{1.0, 1.1, 1.2\}$
    - `ngram_block_size` $\in \{0, 3\}$
  - Precomputes video features through the encoder once for high speed.
  - Scores all combinations using standard metrics (`metrics.evaluate_predictions`).
  - Selects best configuration by validation CIDEr.
  - Saves full validation table and best configuration to `results/decoding_selection.json`.

### [`evaluate_test_set.py`](file:///c:/dev/video/evaluate_test_set.py)
- **Task 2: Strict Test Set Evaluation**:
  - Requires `--decoding-config` (default `results/decoding_selection.json`); explicitly refuses to run without it, enforcing zero test-set tuning.
  - Evaluates test set once with the chosen configuration.
  - Includes controlled ablation (`greedy`, `beam_3`, `beam_5` with identical base parameters).
  - Saves decoding settings and per-video metrics alongside predictions in `results/`.

### [`requirements.txt`](file:///c:/dev/video/requirements.txt)
- Added `pycocoevalcap` and `pytest`.

### Unit Tests ([`tests/`](file:///c:/dev/video/tests/))
- [`tests/conftest.py`](file:///c:/dev/video/tests/conftest.py): ensures project root is on `sys.path`.
- [`tests/test_model.py`](file:///c:/dev/video/tests/test_model.py): verifies encoder output shapes (`temporal`, `temporal_bigru`, `spatiotemporal`, `spatiotemporal_attention`), attention weights sum to 1, precomputed $W_1$ equivalence, `reshape_attention_weights`, and backward-compatible checkpoint loading.
- [`tests/test_beam_search.py`](file:///c:/dev/video/tests/test_beam_search.py): verifies beam_width=1 equals greedy on 20 videos, no `<pad>`/`<bos>` in outputs, max length respected, and determinism.
- [`tests/test_dataset.py`](file:///c:/dev/video/tests/test_dataset.py): verifies captions differ across epochs, union over 10 epochs exceeds $k$, reproducibility for same `(seed, epoch)`, and `captions_per_video=0`.
- [`tests/test_metrics.py`](file:///c:/dev/video/tests/test_metrics.py): verifies identical hyp/ref yields BLEU-4 and CIDEr scores, empty hypothesis handling, key consistency, and backward compatibility.
- [`tests/test_decoding_tuning.py`](file:///c:/dev/video/tests/test_decoding_tuning.py): verifies `evaluate_test_set.py` refuses to execute without `--decoding-config`.

---

## 2. Key Assumptions

1. **Metric Scaling**: In accordance with Task 8, all standard metrics (`BLEU-1` through `BLEU-4`, `METEOR`, `ROUGE-L`, `CIDEr`) are scaled by $\times 100$ and rounded to 2 decimal places with `"scale": "x100"`.
2. **Strict Test Set Isolation**: The test set is never accessed for hyperparameter selection, tuning, or early stopping. Validation-only tuning (`tune_decoding.py`) must be executed first to generate `results/decoding_selection.json`, which is required by `evaluate_test_set.py`.
3. **Checkpoint Compatibility**: Old checkpoints produced with `encoder_type="temporal"` and `encoder_type="spatiotemporal"` continue to load without errors. `spatiotemporal` maps to `TemporalBiGRUFeatureEncoder` with identical parameter names.
4. **Attention Key Projection**: Key projection $W_1(\text{encoder\_out})$ is precomputed once per sequence in `CaptionDecoder.forward`, `greedy_decode`, and `beam_search_decode` to optimize decoding efficiency across all $N$ positions.
5. **Caption Resampling Seed Strategy**: Deterministic pseudo-random number generator `random.Random(seed * 1000003 + epoch)` guarantees reproducibility while providing distinct subsets across epochs.
6. **Validation Loss vs. Validation Scoring**: Validation loss is computed across all reference captions ($k=0$) using cross-entropy, whereas greedy decoding for validation metric selection produces one hypothesis per video evaluated against all references.
7. **Deleted Legacy Scripts**: `demo.py`, `train.py`, and `train_backup.py` were requested by the user to be left as deleted.
8. **Clean Workspace & Unique Run Identifiers**: All temporary smoke-test artifacts (checkpoints, histories, and 20-video `decoding_selection.json`) were purged. `train_spatiotemporal.py` dynamically assigns distinct run names (`{encoder_type}_vgg` or explicit `--run-name`) for full training runs to prevent stale configuration use or cross-run collisions.
9. **Device, Seed, and CLI Parameter Handling**: `tune_decoding.py`, `evaluate_test_set.py`, `generate_caption.py`, and `train_spatiotemporal.py` dynamically resolve `device` from config or checkpoint with explicit `--device` CLI overrides, gracefully falling back to CPU if CUDA is requested on non-GPU environments. All CLI interfaces accept `--checkpoint` and `--seed` flags for multi-seed experiments. Model-specific decoding configs (`results/{checkpoint_stem}_decoding_selection.json`) are automatically written and discovered. The authoritative config and resolved configs default dev smoke flags (`max_train_videos`, `max_val_videos`) to `null`.
