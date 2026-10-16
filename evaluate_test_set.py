import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from generate_caption import load_model, greedy_decode, beam_search_decode, compute_eos_blocklist
from metrics import evaluate_predictions, caption_statistics

ROOT = Path(__file__).resolve().parent
TRAIN_ANNOTATIONS_PATH = ROOT / "data" / "annotations" / "msvd_train.json"
ANNOTATIONS_PATH = ROOT / "data" / "annotations" / "msvd_test.json"
FEATURE_DIR = ROOT / "data" / "features" / "vgg16_spatial"
RESULTS_DIR = ROOT / "results"
DEFAULT_DECODING_CONFIG = RESULTS_DIR / "decoding_selection.json"


def find_default_checkpoint():
    priority = [
        ROOT / "checkpoints" / "spatiotemporal_attention_vgg_best.pt",
        ROOT / "checkpoints" / "temporal_attention_continued_best.pt",
        ROOT / "checkpoints" / "spatiotemporal_vgg_best.pt",
        ROOT / "checkpoints" / "best_model.pt",
    ]
    for p in priority:
        if p.is_file():
            return p
    return ROOT / "checkpoints" / "temporal_attention_continued_best.pt"


def decode_one_setting(
    model,
    encoded_cache,
    records,
    id_to_word,
    max_length,
    device,
    setting,
    suppress_unk=True,
    block_eos_after=None,
):
    bw = setting.get("beam_width", 1)
    lp = setting.get("length_penalty", 0.7)
    rp = setting.get("repetition_penalty", 1.0)
    nb = setting.get("ngram_block_size", 0)

    setting_blocklist = setting.get("eos_blocklist", True)
    effective_block_eos = block_eos_after if setting_blocklist else None

    hyps = {}
    with torch.inference_mode():
        for r in records:
            vid = r["video_id"]
            enc_out = encoded_cache[vid]
            if bw == 1:
                cap, _ = greedy_decode(
                    model=model,
                    encoder_out=enc_out,
                    id_to_word=id_to_word,
                    max_length=max_length,
                    device=device,
                    repetition_penalty=rp,
                    suppress_unk=suppress_unk,
                    block_eos_after=effective_block_eos,
                )
            else:
                cap, _ = beam_search_decode(
                    model=model,
                    encoder_out=enc_out,
                    id_to_word=id_to_word,
                    max_length=max_length,
                    device=device,
                    beam_width=bw,
                    length_penalty=lp,
                    repetition_penalty=rp,
                    ngram_block_size=nb,
                    suppress_unk=suppress_unk,
                    block_eos_after=effective_block_eos,
                )
            hyps[vid] = cap
    return hyps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video captioning checkpoint on MSVD test set strictly with a tuned decoding config."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(find_default_checkpoint()),
        help="Path to checkpoint to evaluate.",
    )
    parser.add_argument(
        "--decoding-config",
        type=str,
        default=None,
        help="Path to decoding configuration JSON produced on val set (defaults to results/{checkpoint_stem}_decoding_selection.json or results/decoding_selection.json).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(ANNOTATIONS_PATH),
        help="Path to test set annotations JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', etc.; overrides checkpoint/config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-test-videos",
        type=int,
        default=None,
        help="Dev/smoke flag: limit number of test videos.",
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        default=str(TRAIN_ANNOTATIONS_PATH),
        help="Path to train annotations JSON (used for EOS blocklist).",
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--eos-blocklist",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable EOS blocklist (defaults to setting in decoding config, or True).",
        )
    else:
        parser.add_argument(
            "--eos-blocklist",
            dest="eos_blocklist",
            action="store_true",
            default=None,
            help="Enable EOS blocklist (defaults to setting in decoding config, or True).",
        )
        parser.add_argument(
            "--no-eos-blocklist",
            dest="eos_blocklist",
            action="store_false",
            help="Disable EOS blocklist.",
        )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip controlled ablation (greedy vs beam k=3 vs beam k=5).",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Resolve decoding config: prioritize model-specific selection, then default
    if args.decoding_config is not None:
        decoding_config_path = Path(args.decoding_config)
    else:
        model_specific_config = RESULTS_DIR / f"{checkpoint_path.stem}_decoding_selection.json"
        if model_specific_config.is_file():
            decoding_config_path = model_specific_config
        else:
            decoding_config_path = DEFAULT_DECODING_CONFIG

    # Rule: Refuse to run without valid decoding config
    if not decoding_config_path.is_file():
        sys.stderr.write(
            f"ERROR: Decoding configuration file not found at '{decoding_config_path}'.\n"
            "evaluate_test_set.py requires --decoding-config (default results/{checkpoint_stem}_decoding_selection.json or results/decoding_selection.json) "
            "and refuses to run without it to prevent tuning decoding hyperparameters on the TEST set.\n"
            "Please run tune_decoding.py on the validation set first.\n"
        )
        sys.exit(1)

    with decoding_config_path.open("r", encoding="utf-8") as f:
        decoding_data = json.load(f)

    chosen_config = decoding_data.get("best_config", decoding_data)

    if args.eos_blocklist is not None:
        use_eos_blocklist = bool(args.eos_blocklist)
    else:
        use_eos_blocklist = bool(
            chosen_config.get("eos_blocklist", decoding_data.get("eos_blocklist", True))
        )

    # Resolve device
    device_arg = None
    if args.device is not None:
        dev_str = args.device
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"Warning: {dev_str} requested but CUDA is not available. Falling back to cpu.")
            dev_str = "cpu"
        device_arg = torch.device(dev_str)

    model, config, vocab, id_to_word, checkpoint = load_model(
        device=device_arg, checkpoint_path=checkpoint_path
    )
    device = next(model.parameters()).device
    model.eval()

    if use_eos_blocklist:
        train_annotations = Path(args.train_annotations)
        block_eos_after = compute_eos_blocklist(train_annotations, vocab)
    else:
        block_eos_after = None

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print("EVALUATING MSVD TEST SET")
    print(f"Checkpoint: {checkpoint_path.name} | Device: {device}")
    print(f"Decoding config source: {decoding_config_path}")
    print(f"Loaded config: beam_width={chosen_config.get('beam_width', 1)}, "
          f"length_penalty={chosen_config.get('length_penalty', 0.7)}, "
          f"repetition_penalty={chosen_config.get('repetition_penalty', 1.0)}, "
          f"ngram_block_size={chosen_config.get('ngram_block_size', 0)}, "
          f"eos_blocklist={use_eos_blocklist}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.max_test_videos is not None:
        records = records[: args.max_test_videos]

    total_test = len(records)
    refs = {r["video_id"]: r["caption"] for r in records}

    # Precompute test video encoder representations
    encoded_cache = {}
    print(f"Precomputing encoder features for {total_test} test videos...")
    start_time = time.time()
    with torch.inference_mode():
        for r in records:
            vid = r["video_id"]
            feat_file = FEATURE_DIR / f"{vid}.npy"
            if not feat_file.is_file():
                raise FileNotFoundError(f"Feature file not found: {feat_file}")
            feat_np = np.load(feat_file, allow_pickle=False)
            if feat_np.shape != (8, 512, 7, 7):
                raise ValueError(f"Unexpected feature shape for {vid}: {feat_np.shape}")
            if feat_np.dtype != np.float32:
                raise TypeError(f"Expected float32 features for {vid}, got {feat_np.dtype}")
            if not np.isfinite(feat_np).all():
                raise ValueError(f"Non-finite feature values for {vid}")
            feat_t = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)
            encoded_cache[vid] = model.encoder(feat_t)

    print(f"Encoder precomputation completed in {time.time() - start_time:.1f}s")
    max_length = config.get("max_caption_length", 30)
    suppress_unk = config.get("suppress_unk", True)

    # 1. Evaluate test set ONCE with the chosen configuration
    print("\nRunning test evaluation with tuned config...")
    chosen_hyps = decode_one_setting(
        model=model,
        encoded_cache=encoded_cache,
        records=records,
        id_to_word=id_to_word,
        max_length=max_length,
        device=device,
        setting=chosen_config,
        suppress_unk=suppress_unk,
        block_eos_after=block_eos_after,
    )

    test_metrics = evaluate_predictions(refs, chosen_hyps)
    diversity_stats = caption_statistics(list(chosen_hyps.values()))
    test_metrics.update(diversity_stats)

    # 2. Controlled ablation: greedy vs beam k=3 vs beam k=5 with identical base parameters
    ablation_results = {}
    ab_hyps_dict = {}
    if not args.no_ablation:
        print("\nRunning controlled ablation (greedy vs beam k=3 vs beam k=5)...")
        ablation_configs = {
            "greedy": {"beam_width": 1, "length_penalty": 0.7, "repetition_penalty": 1.0, "ngram_block_size": 0},
            "beam_3": {"beam_width": 3, "length_penalty": 0.7, "repetition_penalty": 1.0, "ngram_block_size": 0},
            "beam_5": {"beam_width": 5, "length_penalty": 0.7, "repetition_penalty": 1.0, "ngram_block_size": 0},
        }
        for name, ab_cfg in ablation_configs.items():
            ab_hyps = decode_one_setting(
                model=model,
                encoded_cache=encoded_cache,
                records=records,
                id_to_word=id_to_word,
                max_length=max_length,
                device=device,
                setting=ab_cfg,
                suppress_unk=suppress_unk,
                block_eos_after=block_eos_after,
            )
            ab_hyps_dict[name] = ab_hyps
            ab_scores = evaluate_predictions(refs, ab_hyps)
            ab_div = caption_statistics(list(ab_hyps.values()))
            ab_scores.update(ab_div)
            ablation_results[name] = ab_scores

    # Save output artifacts
    ckpt_stem = checkpoint_path.stem
    metrics_path = RESULTS_DIR / f"{ckpt_stem}_test_evaluation.json"
    predictions_path = RESULTS_DIR / f"{ckpt_stem}_test_predictions.json"

    # Also save standard names for default pipeline
    standard_metrics_path = RESULTS_DIR / "test_set_evaluation.json"
    standard_predictions_path = RESULTS_DIR / "test_set_predictions.json"

    per_video_scores = test_metrics.get("per_video", {})

    output_records = [
        {
            "video_id": r["video_id"],
            "prediction": chosen_hyps[r["video_id"]],
            "predictions": {
                "chosen": chosen_hyps[r["video_id"]],
                **(
                    {name: ab_hyps_dict[name][r["video_id"]] for name in ab_hyps_dict}
                    if not args.no_ablation
                    else {}
                ),
            },
            "references": r["caption"],
            "scores": per_video_scores.get(r["video_id"], {}),
        }
        for r in records
    ]

    chosen_config["eos_blocklist"] = use_eos_blocklist
    chosen_config["blocklist_size"] = len(block_eos_after) if block_eos_after is not None else 0

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "best_val_cider": checkpoint.get("best_val_cider"),
        "test_video_count": total_test,
        "decoding_settings": chosen_config,
        "decoding_config_source": str(decoding_config_path),
        "evaluation_results": {k: v for k, v in test_metrics.items() if k != "per_video"},
        "ablation_results": ablation_results,
    }

    for p in [metrics_path, standard_metrics_path]:
        with p.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    for p in [predictions_path, standard_predictions_path]:
        with p.open("w", encoding="utf-8") as f:
            json.dump(output_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("TEST EVALUATION RESULTS (TUNED DECODING)")
    print(f"CIDEr:       {test_metrics.get('CIDEr', 0.0):.2f}")
    print(f"BLEU-4:      {test_metrics.get('BLEU-4', 0.0):.2f}")
    print(f"BLEU-3:      {test_metrics.get('BLEU-3', 0.0):.2f}")
    print(f"BLEU-2:      {test_metrics.get('BLEU-2', 0.0):.2f}")
    print(f"BLEU-1:      {test_metrics.get('BLEU-1', 0.0):.2f}")
    print(f"METEOR:      {test_metrics.get('METEOR', 0.0):.2f}")
    print(f"ROUGE-L:     {test_metrics.get('ROUGE-L', 0.0):.2f}")
    print(f"Distinct-1:  {test_metrics.get('distinct_1', 0.0):.4f}")
    print(f"Distinct-2:  {test_metrics.get('distinct_2', 0.0):.4f}")
    print(f"Rep Bigram%: {test_metrics.get('repeated_bigram_caption_rate', 0.0):.4f}")
    print("=" * 70)

    if ablation_results:
        print("\n" + "=" * 75)
        print("CONTROLLED ABLATION RESULTS (identical rep_penalty=1.0, len_penalty=0.7, ngram_block=0)")
        print(f"{'Strategy':<16} | {'CIDEr':<8} | {'BLEU-4':<8} | {'METEOR':<8} | {'ROUGE-L':<8}")
        print("-" * 75)
        for name, ab in ablation_results.items():
            print(f"{name:<16} | {ab.get('CIDEr', 0.0):<8.2f} | {ab.get('BLEU-4', 0.0):<8.2f} | "
                  f"{ab.get('METEOR', 0.0):<8.2f} | {ab.get('ROUGE-L', 0.0):<8.2f}")
        print("=" * 75)

    print(f"\nSaved test report to:      {metrics_path}")
    print(f"Saved test predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
