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
- **Tasks 4, 6 & Resume/Logging**:
  - Command-line arguments default to `None` and override `config.json` only when explicitly passed. Added `--encoder-type` and `--run-name`.
  - Added `--resume` flag: resumes training from `checkpoints/{run_name}_last.pt`.
  - Checkpoint overwrite protection: without `--resume`, refuses to overwrite existing `checkpoints/{run_name}_best.pt` or `checkpoints/{run_name}_last.pt`, exiting with a clear error message and code 1 before modifying any filesystem state.
  - Complete state serialization: `checkpoints/{run_name}_last.pt` persists all training state every epoch (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_metric_val`, `epochs_without_improvement`, `history`, `val_scores`, `config`, `vocab`, and `architecture`).
  - Resumption restores model, optimizer (ensuring tensors map to active device), scheduler, sets `start_epoch = ckpt["epoch"] + 1`, restores `best_metric_val`, `epochs_without_improvement`, and `history`.
  - Epoch logging: prints a single line per epoch with train loss, val loss, val CIDEr, BLEU-4, METEOR, ROUGE-L, learning rate, and elapsed seconds.
  - History tracking: appends each epoch record to `results/{run_name}_history.json`.
  - Saves resolved config to `results/{run_name}_config_resolved.json` and embeds it into checkpoints.
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

---

## 3. End-Of-Sequence (<eos>) Blocklist Rule & Decoding Tuning Optimization

### 3.1 Motivation & Method
Auto-regressive caption models frequently suffer from premature `<eos>` emission following non-terminal tokens (e.g. articles such as "a", "the", prepositions such as "in", "into", "of", and auxiliary/transitive verbs), resulting in incomplete phrases like *"a woman is playing with a <eos>"* or *"a man in the <eos>"*. The EOS blocklist rule eliminates this failure mode without fine-tuning weights by masking `logits[2]` (`<eos>`) to $-\infty$ immediately following any token known to rarely or never terminate complete sentences.

### 3.2 Key Implementations
- **Strict TRAIN Provenance & Caching ([`generate_caption.py`](file:///c:/dev/video/generate_caption.py))**:
  - Implemented `compute_eos_blocklist(train_annotations_path, vocab, min_count=50, max_eos_rate=0.002, cache_path=None)`: tokenizes TRAIN split captions only (`dataset.tokenize`), tracking total occurrences and final-token frequencies for each token $w$. Tokens with $\text{count} \ge 50$ and $\frac{\text{final\_count}}{\text{count}} \le 0.002$ are added to the blocklist.
  - Caches results to `data/annotations/eos_blocklist.json` along with configuration parameters and an explicit provenance note affirming extraction from the TRAIN split only (preventing validation/test leakage). Reuses the cache if parameters match.
- **Decoding Rule Enforcement ([`generate_caption.py`](file:///c:/dev/video/generate_caption.py))**:
  - Added optional `block_eos_after` (set of token IDs, default `None`) to both `greedy_decode` and `beam_search_decode`.
  - In `greedy_decode`: sets `logits[2] = -float("inf")` before `torch.argmax` whenever the previous token ID is in `block_eos_after`.
  - In `beam_search_decode`: sets `logits[2] = -float("inf")` per hypothesis before log-softmax and top-$k$ pruning whenever `hyp.last_token_id` is in `block_eos_after`.
  - Retains identical behavior when `block_eos_after=None`, ensuring zero disruption to training validation routines in `train_spatiotemporal.py`.
- **Deduplicated Grid Search ([`tune_decoding.py`](file:///c:/dev/video/tune_decoding.py))**:
  - Deduplicated greedy search: for `beam_width == 1`, loops only over `repetition_penalty` with `length_penalty=0.0` and `ngram_block_size=0`, eliminating 21 redundant evaluations. For `beam_width > 1`, retains the full grid ($2 \times 4 \times 3 \times 2 = 48$), reducing total combinations from 72 to 51.
  - Added `--eos-blocklist` / `--no-eos-blocklist` CLI flags (default: enabled).
  - Serializes `"eos_blocklist"` (bool) and `"blocklist_size"` into `best_config` and `result_payload`.
- **Decoding Config Propagation ([`evaluate_test_set.py`](file:///c:/dev/video/evaluate_test_set.py))**:
  - Added `--eos-blocklist` / `--no-eos-blocklist` CLI flags (defaulting to config setting).
  - Reads `"eos_blocklist"` directly from the tuned decoding configuration, loading the train blocklist and passing it through `decode_one_setting` to both the primary evaluation and controlled ablation.
- **Validation Ablation Script ([`ablate_eos_rule.py`](file:///c:/dev/video/ablate_eos_rule.py))**:
  - Standalone script evaluating greedy and beam_5 ($lp=0.7$) with and without the blocklist strictly on the validation set.
  - Measures standard COCO metrics (`CIDEr`, `BLEU-1`..`4`, `METEOR`, `ROUGE-L`), share of captions ending in a blocklist token, mean caption length, and unique word count.
  - Outputs formatted results table and serializes to `results/{checkpoint_stem}_eos_rule_val.json`.
- **Unit Tests ([`tests/test_eos_blocklist.py`](file:///c:/dev/video/tests/test_eos_blocklist.py) & [`tests/test_decoding_tuning.py`](file:///c:/dev/video/tests/test_decoding_tuning.py))**:
  - Validates blocklist membership: includes 'a' and 'the', and excludes common sentence-ending nouns ("bowl", "egg", "water", "cat", "dog", etc.).
  - Validates tiny seeded model behavior: `<eos>` is never emitted right after a blocked token in greedy and beam decoding.
  - Validates forced logits test: blocking intervenes when `<eos>` would otherwise be emitted.
  - Validates backward compatibility: output is completely unchanged when rule is disabled (`block_eos_after=None`).
  - Validates grid deduplication: exactly 51 unique configurations with fixed parameters for $k=1$.
  - Validates test set config reading: `"eos_blocklist"` is properly read from JSON configurations.

---

## 4. Mean-Context Attention Ablation, Efficiency Profiling & Attention Visualization

### 4.1 Mean-Context Ablation (No Learned Attention)
- **Model Architecture ([`model.py`](file:///c:/dev/video/model.py))**:
  - Parameterized `CaptionDecoder` and `VideoCaptionModel` with `attention_type` (`"additive"` default, or `"mean"`).
  - For `attention_type == "mean"`, attention weights and projection parameters are eliminated entirely (`self.attention = None`).
  - At every decoding step in `CaptionDecoder.forward`, the context vector is directly the spatial/temporal mean of encoder positions (`encoder_out.mean(dim=1)`), with uniform attention history ($1/N$) recorded for diagnostic compatibility.
- **Decoding & Checkpoint Loading ([`generate_caption.py`](file:///c:/dev/video/generate_caption.py))**:
  - Updated `load_model` to extract `attention_type` from `config` or top-level checkpoint dictionary, defaulting cleanly to `"additive"` for backward compatibility with all historical checkpoints.
  - Updated `greedy_decode` and `beam_search_decode` to bypass key projection ($W_1$) and use mean context vectors when `attention_type == "mean"`.
- **Training CLI & Serialization ([`train_spatiotemporal.py`](file:///c:/dev/video/train_spatiotemporal.py))**:
  - Added `--attention-type {additive,mean}` CLI flag (default: `None`, falling back to `config.json` value or `"additive"` without mutating `config.json`).
  - Propagated resolved `attention_type` into `VideoCaptionModel`, resolved config files (`results/{run_name}_config_resolved.json`), and checkpoint dictionaries (`best` and `last`).
- **Queue Script Extension ([`run_queue.ps1`](file:///c:/dev/video/run_queue.ps1))**:
  - Extended job specification to allow optional third field: `"enc:seed:attn"`.
  - Sets run name to `${enc}_${attn}_v2_s${seed}` when `attn` is supplied and not `"additive"`, and passes `--attention-type $attn` to `train_spatiotemporal.py`.

### 4.2 Efficiency Profiler ([`efficiency.py`](file:///c:/dev/video/efficiency.py))
- Implemented standalone CLI profiling script:
  - Trains/counts trainable parameters on `VideoCaptionModel`.
  - Counts frozen VGG16 feature extractor parameters directly from torchvision (`vgg16(weights=None).features`: 14,714,688 parameters).
  - Evaluates mean greedy and beam-5 latency per video across 100 test videos with 10 warm-up runs, strictly on CPU with thread configuration derived from `OMP_NUM_THREADS`.
  - Asynchronously samples peak process memory (RSS) using `psutil`.
  - Queries system CPU model string across Windows registry, Linux `/proc/cpuinfo`, and platform identifiers.
  - Serializes report to `results/{checkpoint_stem}_efficiency.json`.

### 4.3 Attention Visualization ([`visualize_attention.py`](file:///c:/dev/video/visualize_attention.py))
- Implemented visualization tool accepting `--checkpoint PATH --video-ids ID [ID ...] --decoding greedy`:
  - For temporal encoders (`temporal`, `temporal_bigru`, `spatiotemporal`): plots words $\times$ 8 frames attention heatmap with numerical cell annotations, saving PNGs to `results/attention/{run_name}/`.
  - For `spatiotemporal_attention`:
    - Marginalizes the 392 region weights over the $7 \times 7$ spatial grid to generate frame attention heatmaps.
    - Reads frame indices from `data/features/vgg16_spatial/manifest.jsonl` and loads sampled video frames from `data/raw/YouTubeClips`.
    - Overlays bicubic-upsampled $7 \times 7$ spatial attention heatmaps onto the 8 sampled video frames for each generated word.

### 4.4 Dependencies & Testing
- Updated [`requirements.txt`](file:///c:/dev/video/requirements.txt) to include `psutil` and `matplotlib`.
- Added unit test suite [`tests/test_mean_attention_and_efficiency.py`](file:///c:/dev/video/tests/test_mean_attention_and_efficiency.py) covering mean-context decoder parameter isolation, uniform attention weights, greedy vs. beam-1 equivalence, checkpoint backward compatibility, VGG16 feature parameter counting, memory sampling, and visualization routines.

---

## 5. CLIP ViT-B/32 Feature Extraction & Representation Support

### 5.1 CLIP Feature Extraction Script ([`extract_features_clip.py`](file:///c:/dev/video/extract_features_clip.py))
- Implemented standalone feature extraction using CLIP ViT-B/32 image encoder:
  - Prefers `open_clip_torch` with model `"ViT-B-32"` and `"openai"` pretrained weights, falling back gracefully to Hugging Face `transformers` (`"openai/clip-vit-base-patch32"`).
  - Reuses exact `sampled_frame_indices` from [`data/features/vgg16_spatial/manifest.jsonl`](file:///c:/dev/video/data/features/vgg16_spatial/manifest.jsonl) ensuring identical temporal sampling across feature backbones.
  - Decodes frames with OpenCV, converts BGR to RGB, applies CLIP preprocessing transforms, and runs inference in batches of up to 16 frames under `torch.inference_mode` on CPU (respecting `OMP_NUM_THREADS`).
  - L2-normalises projected 512-d embeddings along the embedding dimension and saves float32 numpy arrays of shape `(8, 512, 1, 1)` into `data/features/clip_vitb32/{video_id}.npy`.
  - Fully resumable extraction via `data/features/clip_vitb32/manifest.jsonl` (validating shape and finiteness on resume), logging errors to `errors.jsonl`, and supporting `--limit N`.

### 5.2 Feature Type Plumbing & Shape Check Relaxation
- **Path Resolution & Shape Rule ([`dataset.py`](file:///c:/dev/video/dataset.py))**:
  - Implemented `feature_dir_for(feature_type)` mapping `"vgg16_spatial"` $\rightarrow$ `data/features/vgg16_spatial` and `"clip_vitb32"` $\rightarrow$ `data/features/clip_vitb32` (defaulting to `"vgg16_spatial"` when `None`).
  - Relaxed shape validation via `is_valid_feature_shape(shape)` from strictly `(8, 512, 7, 7)` to `(8, 512, H, W)` where $H \times W \in \{1, 49\}$.
  - Updated [`MSVDDataset`](file:///c:/dev/video/dataset.py#L52) to accept `feature_type` and/or `feature_dir`, ensuring backward compatibility with legacy calls.
- **Training CLI & Checkpoint Serialization ([`train_spatiotemporal.py`](file:///c:/dev/video/train_spatiotemporal.py))**:
  - Added `--feature-type {vgg16_spatial,clip_vitb32}` CLI flag (default: `None`, falling back to `config.json` value or `"vgg16_spatial"` without modifying `config.json`).
  - Propagated resolved `feature_dir` to training/validation datasets and validation evaluation routines (`evaluate_val_greedy`).
  - Recorded `feature_type` in `resolved_config` and checkpoint dictionaries (`best` and `last`).
- **Inference & Evaluation Scripts**:
  - Updated [`evaluate_test_set.py`](file:///c:/dev/video/evaluate_test_set.py), [`tune_decoding.py`](file:///c:/dev/video/tune_decoding.py), [`generate_caption.py`](file:///c:/dev/video/generate_caption.py), [`efficiency.py`](file:///c:/dev/video/efficiency.py), [`visualize_attention.py`](file:///c:/dev/video/visualize_attention.py), [`attention_stats.py`](file:///c:/dev/video/attention_stats.py), [`attention_positions.py`](file:///c:/dev/video/attention_positions.py), [`attention_dynamics.py`](file:///c:/dev/video/attention_dynamics.py), and [`ablate_eos_rule.py`](file:///c:/dev/video/ablate_eos_rule.py) to extract `feature_type` from checkpoint configurations (defaulting to `"vgg16_spatial"` when missing) and load features from the corresponding directory using the relaxed shape rule.

### 5.3 Queue Runner Extension ([`run_queue.ps1`](file:///c:/dev/video/run_queue.ps1))
- Extended job parsing to accept an optional fourth field: `"enc:seed:attn:feat"`.
- Sets run name to `${enc}_${feat}_v2_s${seed}` when `feat` is provided and not `"vgg16_spatial"` (or `${enc}_${attn}_${feat}_v2_s${seed}` when a non-additive attention mechanism is also provided), passing `--feature-type $feat` to `train_spatiotemporal.py`.

### 5.4 Dependencies & Testing
- Updated [`requirements.txt`](file:///c:/dev/video/requirements.txt) to include `open_clip_torch`.
- Added unit test suite [`tests/test_clip_features.py`](file:///c:/dev/video/tests/test_clip_features.py) verifying directory resolution, shape validation, dataset loading with `(8, 512, 1, 1)` features, encoder and decoder forward passes, extraction L2 normalization, and checkpoint configuration compatibility without triggering model downloads.



