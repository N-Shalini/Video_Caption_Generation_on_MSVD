import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from dataset import tokenize, feature_dir_for, is_valid_feature_shape
from generate_caption import load_model, greedy_decode, beam_search_decode, compute_eos_blocklist
from metrics import evaluate_predictions

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "temporal_attention_continued_best.pt"
TRAIN_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "msvd_train.json"
VAL_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "msvd_val.json"
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
RESULTS_DIR = PROJECT_ROOT / "results"


def find_default_checkpoint():
    priority = [
        PROJECT_ROOT / "checkpoints" / "spatiotemporal_attention_vgg_best.pt",
        PROJECT_ROOT / "checkpoints" / "temporal_attention_continued_best.pt",
        PROJECT_ROOT / "checkpoints" / "spatiotemporal_vgg_best.pt",
        PROJECT_ROOT / "checkpoints" / "best_model.pt",
    ]
    for p in priority:
        if p.is_file():
            return p
    return DEFAULT_CHECKPOINT


def run_ablation(
    checkpoint_path,
    val_annotations_path=VAL_ANNOTATIONS,
    train_annotations_path=TRAIN_ANNOTATIONS,
    feature_dir=None,
    output_path=None,
    device_arg=None,
    seed=None,
    max_val_videos=None,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model, config, vocab, id_to_word, checkpoint = load_model(
        device=device_arg, checkpoint_path=checkpoint_path
    )
    device = next(model.parameters()).device
    model.eval()

    run_seed = seed if seed is not None else config.get("seed", 42)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    with open(val_annotations_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if max_val_videos is not None:
        records = records[:max_val_videos]

    val_count = len(records)
    refs = {r["video_id"]: r["caption"] for r in records}

    # Load / compute EOS blocklist strictly from train annotations
    block_eos_after = compute_eos_blocklist(train_annotations_path, vocab)
    blocklist_words = {id_to_word[tid] for tid in block_eos_after if tid in id_to_word}
    blocklist_size = len(block_eos_after)

    print("=" * 80)
    print("EOS BLOCKLIST RULE ABLATION ON VALIDATION SET ONLY")
    print(f"Checkpoint:       {checkpoint_path.name} | Device: {device}")
    print(f"Validation videos: {val_count}")
    print(f"Blocklist size:    {blocklist_size} tokens")
    print("=" * 80)

    # Precompute validation features
    if feature_dir is None:
        feature_type = config.get("feature_type", checkpoint.get("feature_type", "vgg16_spatial"))
        feature_dir = feature_dir_for(feature_type)
    else:
        feature_dir = Path(feature_dir)
    encoded_cache = {}
    print(f"Precomputing encoder features for {val_count} validation videos...")
    with torch.inference_mode():
        for r in records:
            vid = r["video_id"]
            feat_file = feature_dir / f"{vid}.npy"
            if not feat_file.is_file():
                raise FileNotFoundError(f"Feature file not found: {feat_file}")
            feat_np = np.load(feat_file, allow_pickle=False)
            if not is_valid_feature_shape(feat_np.shape):
                raise ValueError(
                    f"Unexpected feature shape for {vid}: {feat_np.shape}; expected (8, 512, H, W) with H*W in {{1, 49}}"
                )
            feat_t = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)
            encoded_cache[vid] = model.encoder(feat_t)

    max_length = config.get("max_caption_length", 30)
    suppress_unk = config.get("suppress_unk", True)

    configs_to_run = [
        {
            "key": "greedy_without_blocklist",
            "name": "Greedy (Baseline)",
            "beam_width": 1,
            "length_penalty": 0.0,
            "repetition_penalty": 1.0,
            "ngram_block_size": 0,
            "block_eos_after": None,
        },
        {
            "key": "greedy_with_blocklist",
            "name": "Greedy + Blocklist",
            "beam_width": 1,
            "length_penalty": 0.0,
            "repetition_penalty": 1.0,
            "ngram_block_size": 0,
            "block_eos_after": block_eos_after,
        },
        {
            "key": "beam_5_without_blocklist",
            "name": "Beam k=5 (lp=0.7 Baseline)",
            "beam_width": 5,
            "length_penalty": 0.7,
            "repetition_penalty": 1.0,
            "ngram_block_size": 0,
            "block_eos_after": None,
        },
        {
            "key": "beam_5_with_blocklist",
            "name": "Beam k=5 + Blocklist",
            "beam_width": 5,
            "length_penalty": 0.7,
            "repetition_penalty": 1.0,
            "ngram_block_size": 0,
            "block_eos_after": block_eos_after,
        },
    ]

    results_table = {}
    all_predictions = {r["video_id"]: {} for r in records}

    for cfg in configs_to_run:
        key = cfg["key"]
        print(f"Decoding {cfg['name']}...")
        hyps = {}
        with torch.inference_mode():
            for r in records:
                vid = r["video_id"]
                enc_out = encoded_cache[vid]
                if cfg["beam_width"] == 1:
                    cap, _ = greedy_decode(
                        model=model,
                        encoder_out=enc_out,
                        id_to_word=id_to_word,
                        max_length=max_length,
                        device=device,
                        repetition_penalty=cfg["repetition_penalty"],
                        suppress_unk=suppress_unk,
                        block_eos_after=cfg["block_eos_after"],
                    )
                else:
                    cap, _ = beam_search_decode(
                        model=model,
                        encoder_out=enc_out,
                        id_to_word=id_to_word,
                        max_length=max_length,
                        device=device,
                        beam_width=cfg["beam_width"],
                        length_penalty=cfg["length_penalty"],
                        repetition_penalty=cfg["repetition_penalty"],
                        ngram_block_size=cfg["ngram_block_size"],
                        suppress_unk=suppress_unk,
                        block_eos_after=cfg["block_eos_after"],
                    )
                hyps[vid] = cap
                all_predictions[vid][key] = cap

        eval_scores = evaluate_predictions(refs, hyps)

        # 1. Share of captions ending in a blocklist token
        ending_in_blocklist_count = 0
        total_tokens = 0
        unique_tokens = set()
        lengths = []

        for cap in hyps.values():
            toks = tokenize(cap)
            lengths.append(len(toks))
            total_tokens += len(toks)
            unique_tokens.update(toks)
            if toks and toks[-1] in blocklist_words:
                ending_in_blocklist_count += 1

        share_ending_in_blocklist = (
            round(ending_in_blocklist_count / val_count, 4) if val_count else 0.0
        )
        mean_caption_length = round(float(np.mean(lengths)), 2) if lengths else 0.0
        unique_words_count = len(unique_tokens)

        results_table[key] = {
            "name": cfg["name"],
            "beam_width": cfg["beam_width"],
            "length_penalty": cfg["length_penalty"],
            "repetition_penalty": cfg["repetition_penalty"],
            "block_eos_enabled": cfg["block_eos_after"] is not None,
            "CIDEr": eval_scores.get("CIDEr", 0.0),
            "BLEU-4": eval_scores.get("BLEU-4", 0.0),
            "BLEU-3": eval_scores.get("BLEU-3", 0.0),
            "BLEU-2": eval_scores.get("BLEU-2", 0.0),
            "BLEU-1": eval_scores.get("BLEU-1", 0.0),
            "METEOR": eval_scores.get("METEOR", 0.0),
            "ROUGE-L": eval_scores.get("ROUGE-L", 0.0),
            "share_ending_in_blocklist": share_ending_in_blocklist,
            "blocklist_ending_captions_count": ending_in_blocklist_count,
            "mean_caption_length": mean_caption_length,
            "unique_words": unique_words_count,
        }

    # Save output JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        ckpt_stem = checkpoint_path.stem
        output_path = RESULTS_DIR / f"{ckpt_stem}_eos_rule_val.json"
    else:
        output_path = Path(output_path)

    output_payload = {
        "checkpoint": str(checkpoint_path),
        "val_video_count": val_count,
        "blocklist_size": blocklist_size,
        "results": results_table,
        "predictions": all_predictions,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print("EOS BLOCKLIST ABLATION RESULTS (VAL SET ONLY)")
    print(f"{'Strategy':<26} | {'CIDEr':<7} | {'BLEU-4':<7} | {'METEOR':<7} | {'ROUGE-L':<7} | {'BlkEnd%':<8} | {'MeanLen':<7} | {'UniqueWords':<11}")
    print("-" * 100)
    for key, res in results_table.items():
        blk_end_pct = f"{res['share_ending_in_blocklist'] * 100:.1f}%"
        print(
            f"{res['name']:<26} | "
            f"{res['CIDEr']:<7.2f} | "
            f"{res['BLEU-4']:<7.2f} | "
            f"{res['METEOR']:<7.2f} | "
            f"{res['ROUGE-L']:<7.2f} | "
            f"{blk_end_pct:<8} | "
            f"{res['mean_caption_length']:<7.2f} | "
            f"{res['unique_words']:<11}"
        )
    print("=" * 100)
    print(f"Results saved to: {output_path}")

    return output_payload


def main():
    parser = argparse.ArgumentParser(
        description="Ablate EOS blocklist rule on MSVD validation set ONLY (no test set)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(find_default_checkpoint()),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--val-annotations",
        type=str,
        default=str(VAL_ANNOTATIONS),
        help="Path to validation annotations JSON.",
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        default=str(TRAIN_ANNOTATIONS),
        help="Path to train annotations JSON (used for EOS blocklist).",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=str(FEATURE_DIR),
        help="Directory with precomputed video features.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ablation JSON (defaults to results/{checkpoint_stem}_eos_rule_val.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', etc.).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-val-videos",
        type=int,
        default=None,
        help="Dev/smoke flag: limit number of validation videos.",
    )

    args = parser.parse_args()

    device_arg = None
    if args.device is not None:
        dev_str = args.device
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"Warning: {dev_str} requested but CUDA is not available. Falling back to cpu.")
            dev_str = "cpu"
        device_arg = torch.device(dev_str)

    run_ablation(
        checkpoint_path=args.checkpoint,
        val_annotations_path=args.val_annotations,
        train_annotations_path=args.train_annotations,
        feature_dir=args.feature_dir,
        output_path=args.output,
        device_arg=device_arg,
        seed=args.seed,
        max_val_videos=args.max_val_videos,
    )


if __name__ == "__main__":
    main()
