import json
import time
from pathlib import Path

import numpy as np
import torch

from model import VideoCaptionModel
from generate_caption import greedy_decode, beam_search_decode
from metrics import compute_all_metrics


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "temporal_attention_continued_best.pt"
ANNOTATIONS_PATH = ROOT / "data" / "annotations" / "msvd_test.json"
FEATURE_DIR = ROOT / "data" / "features" / "vgg16_spatial"
RESULTS_DIR = ROOT / "results"


import argparse

def load_model(device, checkpoint_path=CHECKPOINT_PATH):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    encoder_type = config.get("encoder_type", "temporal")

    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=config["feature_channels"],
        embed_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        encoder_type=encoder_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_to_word = {idx: word for word, idx in vocab.items()}
    return model, config, id_to_word, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate video captioning checkpoint on MSVD test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help=f"Path to checkpoint to evaluate (default: {CHECKPOINT_PATH.name}).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    eval_checkpoint = Path(args.checkpoint)
    model, config, id_to_word, checkpoint = load_model(device, checkpoint_path=eval_checkpoint)

    print("=" * 60)
    print("EVALUATING TEST SET ON MSVD DATASET")
    print("Checkpoint:", eval_checkpoint.name)
    print("Epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("Test videos:", len(records))
    print("=" * 60)

    # Strategy configurations to evaluate:
    # '1.0' and '1.2' preserve exact backwards compatibility for existing predictions
    strategies = {
        "1.0": {"mode": "greedy", "repetition_penalty": 1.0},
        "1.2": {"mode": "greedy", "repetition_penalty": 1.2},
        "beam_3": {"mode": "beam", "beam_width": 3, "length_penalty": 0.7, "repetition_penalty": 1.1, "ngram_block_size": 3},
        "beam_5": {"mode": "beam", "beam_width": 5, "length_penalty": 0.7, "repetition_penalty": 1.2, "ngram_block_size": 3},
    }

    predictions_by_strategy = {s: [] for s in strategies}
    output_records = []

    start_time = time.time()

    for index, record in enumerate(records):
        video_id = record["video_id"]
        feature_path = FEATURE_DIR / f"{video_id}.npy"

        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing features for {video_id}: {feature_path}")

        features = np.load(feature_path, allow_pickle=False)

        if features.shape != (8, 512, 7, 7):
            raise ValueError(f"Unexpected feature shape for {video_id}: {features.shape}")

        if features.dtype != np.float32:
            raise TypeError(f"Expected float32 features for {video_id}, got {features.dtype}")

        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite feature values for {video_id}")

        record_predictions = {}

        for strat_name, strat_cfg in strategies.items():
            if strat_cfg["mode"] == "beam":
                caption, _ = beam_search_decode(
                    model=model,
                    features=features,
                    id_to_word=id_to_word,
                    max_length=config["max_caption_length"],
                    device=device,
                    beam_width=strat_cfg["beam_width"],
                    length_penalty=strat_cfg["length_penalty"],
                    repetition_penalty=strat_cfg["repetition_penalty"],
                    ngram_block_size=strat_cfg["ngram_block_size"],
                )
            else:
                caption, _ = greedy_decode(
                    model=model,
                    features=features,
                    id_to_word=id_to_word,
                    max_length=config["max_caption_length"],
                    device=device,
                    repetition_penalty=strat_cfg["repetition_penalty"],
                )

            predictions_by_strategy[strat_name].append(caption)
            record_predictions[strat_name] = caption

        output_records.append({
            "video_id": video_id,
            "references": record["caption"],
            "predictions": record_predictions,
        })

        if (index + 1) % 50 == 0 or index + 1 == len(records):
            elapsed = time.time() - start_time
            print(f"Processed {index + 1}/{len(records)} videos ({elapsed:.1f}s)")

    print("\nComputing comprehensive evaluation metrics (BLEU, METEOR, ROUGE-L, CIDEr, Diversity)...")
    comparison = {}

    for strat_name in strategies:
        comparison[strat_name] = compute_all_metrics(
            records, predictions_by_strategy[strat_name]
        )

    if eval_checkpoint.stem == CHECKPOINT_PATH.stem:
        metrics_path = RESULTS_DIR / "test_set_evaluation.json"
        captions_path = RESULTS_DIR / "test_set_predictions.json"
    else:
        metrics_path = RESULTS_DIR / f"{eval_checkpoint.stem}_evaluation.json"
        captions_path = RESULTS_DIR / f"{eval_checkpoint.stem}_predictions.json"

    report = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_file": eval_checkpoint.name,
        "best_validation_loss": checkpoint["best_val_loss"],
        "test_video_count": len(records),
        "strategies": strategies,
        "evaluation_results": comparison,
        "notes": [
            "Evaluated on the latest best model: checkpoints/temporal_attention_continued_best.pt.",
            "Includes greedy decoding (penalties 1.0, 1.2) and beam search decoding (k=3, k=5).",
            "CIDEr uses consensus TF-IDF matching across reference n-grams.",
            "ROUGE-L uses multi-reference LCS F-measure.",
            "BLEU values use smoothed corpus-level NLTK BLEU.",
            "Distinct-n measures unique n-grams divided by all n-gram occurrences.",
            "Unique token count measures the total vocabulary footprint across all 670 generated captions.",
        ],
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    with captions_path.open("w", encoding="utf-8") as file:
        json.dump(output_records, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"{'Metric':<32} | {'Greedy 1.0':<10} | {'Greedy 1.2':<10} | {'Beam (k=3)':<10} | {'Beam (k=5)':<10}")
    print("-" * 80)

    metric_keys = [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr",
        "distinct_1", "distinct_2", "unique_token_count", "mean_caption_tokens",
        "repeated_bigram_caption_rate"
    ]

    for k in metric_keys:
        v_g10 = str(comparison["1.0"].get(k, "-"))
        v_g12 = str(comparison["1.2"].get(k, "-"))
        v_b3 = str(comparison["beam_3"].get(k, "-"))
        v_b5 = str(comparison["beam_5"].get(k, "-"))
        print(f"{k:<32} | {v_g10:<10} | {v_g12:<10} | {v_b3:<10} | {v_b5:<10}")

    print("=" * 80)
    print("\nSaved metrics report to:", metrics_path)
    print("Saved all predictions to:", captions_path)


if __name__ == "__main__":
    main()
