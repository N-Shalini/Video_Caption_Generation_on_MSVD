import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from generate_caption import load_model, greedy_decode, beam_search_decode
from metrics import evaluate_predictions

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "temporal_attention_continued_best.pt"
VAL_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "msvd_val.json"
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "decoding_selection.json"


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


def main():
    parser = argparse.ArgumentParser(
        description="Tune decoding hyperparameters on MSVD validation set ONLY (no test data)."
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
        "--feature-dir",
        type=str,
        default=str(FEATURE_DIR),
        help="Directory with precomputed video features.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for decoding selection JSON (defaults to results/{checkpoint_stem}_decoding_selection.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', etc.; defaults to checkpoint/config device).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (defaults to config seed).",
    )
    parser.add_argument(
        "--max-val-videos",
        type=int,
        default=None,
        help="Dev/smoke flag: limit number of validation videos.",
    )

    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Inspect device
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

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(args.val_annotations, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.max_val_videos is not None:
        records = records[: args.max_val_videos]

    val_count = len(records)
    refs = {r["video_id"]: r["caption"] for r in records}

    print("=" * 70)
    print("DECODING HYPERPARAMETER TUNING (VAL SET ONLY)")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Validation videos: {val_count}")
    print("=" * 70)

    # Precompute encoder features for all validation videos
    feature_dir = Path(args.feature_dir)
    encoded_cache = {}
    print(f"Precomputing encoder representations for {val_count} videos...")
    with torch.inference_mode():
        for r in records:
            vid = r["video_id"]
            feat_file = feature_dir / f"{vid}.npy"
            if not feat_file.is_file():
                raise FileNotFoundError(f"Feature file not found: {feat_file}")
            feat_np = np.load(feat_file, allow_pickle=False)
            feat_t = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)
            enc_out = model.encoder(feat_t)
            encoded_cache[vid] = enc_out

    # Define hyperparameter grid
    beam_widths = [1, 3, 5]
    length_penalties = [0.0, 0.5, 0.7, 1.0]
    repetition_penalties = [1.0, 1.1, 1.2]
    ngram_block_sizes = [0, 3]

    max_length = config.get("max_caption_length", 30)
    suppress_unk = config.get("suppress_unk", True)

    total_combinations = len(beam_widths) * len(length_penalties) * len(repetition_penalties) * len(ngram_block_sizes)
    print(f"Evaluating {total_combinations} grid combinations on validation set...\n")

    val_table = []
    combo_idx = 0
    start_time = time.time()

    # Cache greedy outputs to avoid redundant calls when beam_width=1 and length_penalty varies
    greedy_cache = {}

    for bw in beam_widths:
        for lp in length_penalties:
            for rp in repetition_penalties:
                for nb in ngram_block_sizes:
                    combo_idx += 1
                    hyps = {}

                    if bw == 1:
                        # Greedy decoding (independent of length_penalty)
                        cache_key = (rp, nb, suppress_unk)
                        if cache_key in greedy_cache:
                            hyps = greedy_cache[cache_key]
                        else:
                            with torch.inference_mode():
                                for r in records:
                                    vid = r["video_id"]
                                    enc_out = encoded_cache[vid]
                                    cap, _ = greedy_decode(
                                        model=model,
                                        encoder_out=enc_out,
                                        id_to_word=id_to_word,
                                        max_length=max_length,
                                        device=device,
                                        repetition_penalty=rp,
                                        suppress_unk=suppress_unk,
                                    )
                                    hyps[vid] = cap
                            greedy_cache[cache_key] = hyps
                    else:
                        with torch.inference_mode():
                            for r in records:
                                vid = r["video_id"]
                                enc_out = encoded_cache[vid]
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
                                )
                                hyps[vid] = cap

                    scores = evaluate_predictions(refs, hyps)

                    entry = {
                        "beam_width": bw,
                        "length_penalty": lp,
                        "repetition_penalty": rp,
                        "ngram_block_size": nb,
                        "CIDEr": scores["CIDEr"],
                        "BLEU-4": scores["BLEU-4"],
                        "BLEU-3": scores["BLEU-3"],
                        "BLEU-2": scores["BLEU-2"],
                        "BLEU-1": scores["BLEU-1"],
                        "METEOR": scores["METEOR"],
                        "ROUGE-L": scores["ROUGE-L"],
                    }
                    val_table.append(entry)

                    if combo_idx % 12 == 0 or combo_idx == total_combinations:
                        elapsed = time.time() - start_time
                        print(
                            f"[{combo_idx}/{total_combinations}] bw={bw} lp={lp} rp={rp} nb={nb} | "
                            f"CIDEr={scores['CIDEr']:.2f} BLEU-4={scores['BLEU-4']:.2f} ({elapsed:.1f}s)"
                        )

    # Sort descending by val CIDEr
    val_table.sort(key=lambda x: x["CIDEr"], reverse=True)
    best_config = val_table[0]

    if args.output is not None:
        output_path = Path(args.output)
    else:
        ckpt_stem = checkpoint_path.stem
        output_path = PROJECT_ROOT / "results" / f"{ckpt_stem}_decoding_selection.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "best_config": best_config,
        "checkpoint": str(checkpoint_path),
        "val_video_count": val_count,
        "val_table": val_table,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    # Also write results/decoding_selection.json
    with DEFAULT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("TUNING COMPLETE - BEST VALIDATION CONFIGURATION")
    print(f"Beam width:         {best_config['beam_width']}")
    print(f"Length penalty:     {best_config['length_penalty']}")
    print(f"Repetition penalty: {best_config['repetition_penalty']}")
    print(f"N-gram block size:  {best_config['ngram_block_size']}")
    print(f"Val CIDEr:          {best_config['CIDEr']:.2f}")
    print(f"Val BLEU-4:         {best_config['BLEU-4']:.2f}")
    print(f"Val METEOR:         {best_config['METEOR']:.2f}")
    print(f"Val ROUGE-L:        {best_config['ROUGE-L']:.2f}")
    print(f"Saved results to:   {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
