"""Interactive Academic Demonstration Script for Video Captioning on MSVD Dataset.

Compares:
1. Baseline Model (Temporal Additive Attention - Epoch 42)
2. Enhanced Model (Spatio-Temporal Dual Pooling + BiGRU - Epoch 17)
across Greedy and Beam Search decoding modes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_caption import load_model, greedy_decode, beam_search_decode

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "annotations" / "msvd_test.json"

BASELINE_CKPT = PROJECT_ROOT / "checkpoints" / "temporal_attention_continued_best.pt"
ENHANCED_CKPT = PROJECT_ROOT / "checkpoints" / "spatiotemporal_vgg_best.pt"

# Curated demonstration test videos highlighting specific actions and objects
CURATED_DEMO_VIDEOS = [
    {
        "id": "fr9H1WLcF1A_256_261",
        "category": "Sports / Action",
        "description": "Table tennis / ping pong match",
    },
    {
        "id": "qzQzooI4BqU_5_15",
        "category": "Music / Instrument",
        "description": "Man playing a flute on stage",
    },
    {
        "id": "puineN1UMto_25_29",
        "category": "Animal / Movement",
        "description": "Squirrel hopping/dancing in grass",
    },
    {
        "id": "tJHUH9tpqPg_113_118",
        "category": "Kitchen / Manipulation",
        "description": "Woman squeezing / juicing a lemon",
    },
    {
        "id": "glii-kazad8_21_29",
        "category": "Outdoor / Water",
        "description": "Dog barking at ball floating near pool/dock",
    },
    {
        "id": "pGsU4FekJQM_10_18",
        "category": "Animal / Posture",
        "description": "Dog lying on its back / rolled over",
    },
    {
        "id": "uppFvcVwqqY_5_15",
        "category": "Instrument / Unique Action",
        "description": "Man playing piano / keyboard with nose",
    },
    {
        "id": "klteYv1Uv9A_27_33",
        "category": "Vehicle / Stunt",
        "description": "Man riding motorcycle while lying down",
    },
]


def run_demo_for_video(video_id, base_model, enh_model, id_to_word, config, device, record=None):
    feat_path = FEATURE_DIR / f"{video_id}.npy"
    if not feat_path.is_file():
        print(f"Error: Feature file not found for {video_id}")
        return

    features = np.load(feat_path, allow_pickle=False)

    # 1. Baseline Model Inferences
    base_greedy, _ = greedy_decode(
        base_model, features, id_to_word, config["max_caption_length"], device, repetition_penalty=1.2
    )
    base_beam, _ = beam_search_decode(
        base_model, features, id_to_word, config["max_caption_length"], device, beam_width=5, repetition_penalty=1.2
    )

    # 2. Enhanced Model Inferences
    enh_greedy, _ = greedy_decode(
        enh_model, features, id_to_word, config["max_caption_length"], device, repetition_penalty=1.2
    )
    enh_beam, _ = beam_search_decode(
        enh_model, features, id_to_word, config["max_caption_length"], device, beam_width=5, repetition_penalty=1.2
    )

    print("\n" + "=" * 76)
    print(f"  VIDEO ID: {video_id}")
    print("=" * 76)

    if record:
        print("\n  GROUND TRUTH REFERENCES (Sample):")
        for ref in record["caption"][:4]:
            print(f"    * \"{ref}\"")

    print("\n  [1] BASELINE MODEL (Temporal Additive Attention - Epoch 42)")
    print(f"      - Greedy (1.2):  \"{base_greedy}\"")
    print(f"      - Beam (k=5):    \"{base_beam}\"")

    print("\n  [2] ENHANCED MODEL (Spatio-Temporal Dual Pooling + BiGRU - Epoch 17)")
    print(f"      - Greedy (1.2):  \"{enh_greedy}\"")
    print(f"      - Beam (k=5):    \"{enh_beam}\"")
    print("-" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="Run Academic Demonstration for Video Captioning on MSVD."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Optional specific video ID to caption.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Zero-based test index in msvd_test.json.",
    )
    parser.add_argument(
        "--curated",
        action="store_true",
        help="Run through curated academic demonstration videos sequentially.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    print("=" * 76)
    print("  LOADING DEMONSTRATION CHECKPOINTS...")
    print("=" * 76)
    print(f"  Baseline Checkpoint: {BASELINE_CKPT.name}")
    print(f"  Enhanced Checkpoint: {ENHANCED_CKPT.name}")

    base_model, config, vocab, id_to_word, base_info = load_model(device, BASELINE_CKPT)
    enh_model, _, _, _, enh_info = load_model(device, ENHANCED_CKPT)

    print(f"  Baseline Best Val Loss: {base_info['best_val_loss']:.4f} (Epoch {base_info['epoch']})")
    print(f"  Enhanced Best Val Loss: {enh_info['best_val_loss']:.4f} (Epoch {enh_info['epoch']})")
    print("  All models loaded successfully on CPU.\n")

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    records_by_id = {r["video_id"]: r for r in records}

    if args.video_id:
        record = records_by_id.get(args.video_id)
        run_demo_for_video(args.video_id, base_model, enh_model, id_to_word, config, device, record)

    elif args.index is not None:
        if 0 <= args.index < len(records):
            record = records[args.index]
            run_demo_for_video(record["video_id"], base_model, enh_model, id_to_word, config, device, record)
        else:
            print(f"Index must be between 0 and {len(records) - 1}")

    elif args.curated:
        print(f"Running Curated Academic Demonstration ({len(CURATED_DEMO_VIDEOS)} Showcase Videos)...")
        for item in CURATED_DEMO_VIDEOS:
            vid = item["id"]
            print(f"\n[Category: {item['category']}] -- Context: {item['description']}")
            record = records_by_id.get(vid)
            run_demo_for_video(vid, base_model, enh_model, id_to_word, config, device, record)

    else:
        # Default interactive menu
        print("ACADEMIC DEMONSTRATION MENU:")
        print("----------------------------")
        print("Choose an option:")
        print("  1. Run Curated Showcase Videos (Side-by-side comparison of error cases)")
        print("  2. Test First 5 Videos in Test Set")
        print("  3. Specify Video ID")
        print("\nTip: You can also pass flags directly:")
        print("  python demo.py --curated")
        print("  python demo.py --video-id fr9H1WLcF1A_256_261")
        print("  python demo.py --index 0")


if __name__ == "__main__":
    main()
