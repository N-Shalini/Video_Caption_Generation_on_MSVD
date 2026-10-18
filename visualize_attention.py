import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_caption import load_model, greedy_decode
from model import reshape_attention_weights

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
DEFAULT_MANIFEST_PATH = DEFAULT_FEATURE_DIR / "manifest.jsonl"
DEFAULT_RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw" / "YouTubeClips"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "attention"


def load_manifest(manifest_path=DEFAULT_MANIFEST_PATH):
    """Load video manifest mapping video_id to manifest metadata."""
    manifest = {}
    if not manifest_path.is_file():
        return manifest

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                manifest[record["video_id"]] = record
            except Exception:
                continue
    return manifest


def extract_sampled_frames(video_id, manifest_record, raw_videos_dir=DEFAULT_RAW_VIDEOS_DIR, num_frames=8):
    """Extract sampled video frames as RGB arrays."""
    video_filename = (
        manifest_record.get("video", f"{video_id}.avi")
        if manifest_record
        else f"{video_id}.avi"
    )
    video_path = raw_videos_dir / video_filename

    if not video_path.is_file():
        # Check alternative extensions
        for alt_ext in [".mp4", ".mkv", ".webm"]:
            alt_path = raw_videos_dir / f"{video_id}{alt_ext}"
            if alt_path.is_file():
                video_path = alt_path
                break

    if manifest_record and "sampled_frame_indices" in manifest_record:
        sampled_indices = manifest_record["sampled_frame_indices"]
    else:
        sampled_indices = None

    frames_rgb = []
    actual_indices = []

    if video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if sampled_indices is None:
                if total_frames > 0:
                    sampled_indices = (
                        np.linspace(0, total_frames - 1, num=min(num_frames, total_frames))
                        .round()
                        .astype(int)
                        .tolist()
                    )
                else:
                    sampled_indices = list(range(num_frames))

            for idx in sampled_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    actual_indices.append(int(idx))
                else:
                    frames_rgb.append(np.zeros((240, 320, 3), dtype=np.uint8))
                    actual_indices.append(int(idx))
            cap.release()

    # Fallback placeholder frames if video could not be read
    while len(frames_rgb) < num_frames:
        frames_rgb.append(np.full((240, 320, 3), 128, dtype=np.uint8))
        actual_indices.append(len(frames_rgb) - 1)

    return frames_rgb[:num_frames], actual_indices[:num_frames]


def plot_frame_attention_heatmap(attn_matrix, words, video_id, caption, out_path):
    """Plot frame attention per generated word as a words x 8 heatmap."""
    num_words = len(words)
    num_frames = attn_matrix.shape[1]

    fig_height = max(3.5, 0.45 * num_words + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    vmax = float(attn_matrix.max()) if float(attn_matrix.max()) > 0 else 1.0
    im = ax.imshow(
        attn_matrix,
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(num_frames))
    ax.set_xticklabels([f"Frame {i}" for i in range(num_frames)], fontsize=10)
    ax.set_yticks(range(num_words))
    ax.set_yticklabels(words, fontsize=10)

    ax.set_xlabel("Video Frames", fontsize=11, labelpad=8)
    ax.set_ylabel("Generated Words", fontsize=11, labelpad=8)
    ax.set_title(
        f"Frame Attention Heatmap: {video_id}\nCaption: \"{caption}\"",
        fontsize=11,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=10)

    # Numerical cell values for readability
    if num_words <= 30:
        for r in range(num_words):
            for c in range(num_frames):
                val = float(attn_matrix[r, c])
                text_color = "white" if val > 0.5 * vmax else "black"
                ax.text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spatiotemporal_spatial_overlay(
    reshaped_weights,
    frames_rgb,
    frame_indices,
    words,
    video_id,
    caption,
    out_path,
):
    """Overlay 7x7 spatial attention weights (upsampled) on 8 sampled frames per word."""
    # reshaped_weights: numpy array of shape [num_words, 8, 7, 7]
    num_words = len(words)
    num_frames = min(8, len(frames_rgb))

    fig, axes = plt.subplots(
        num_words,
        num_frames,
        figsize=(2.2 * num_frames, 2.0 * num_words),
        squeeze=False,
    )

    for w_idx in range(num_words):
        for f_idx in range(num_frames):
            ax = axes[w_idx, f_idx]
            frame = frames_rgb[f_idx]
            h, w, _ = frame.shape
            spatial_weights = reshaped_weights[w_idx, f_idx]  # 7x7

            ax.imshow(frame)
            ax.imshow(
                spatial_weights,
                cmap="jet",
                alpha=0.5,
                interpolation="bicubic",
                extent=[0, w, h, 0],
                vmin=0.0,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if w_idx == 0:
                frame_label = (
                    f"Frame {f_idx}\n(idx {frame_indices[f_idx]})"
                    if f_idx < len(frame_indices)
                    else f"Frame {f_idx}"
                )
                ax.set_title(frame_label, fontsize=9)

            if f_idx == 0:
                ax.set_ylabel(
                    f'"{words[w_idx]}"',
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=32,
                    va="center",
                )

    fig.suptitle(
        f"Spatiotemporal Attention Overlay: {video_id}\nCaption: \"{caption}\"",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention maps (temporal heatmap and spatiotemporal overlay) for video captioning."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        required=True,
        help="One or more video IDs to visualize.",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        choices=["greedy"],
        default="greedy",
        help="Decoding strategy: greedy (default: greedy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/attention/{run_name}).",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = None
    if args.device:
        device = torch.device(args.device)

    model, config, vocab, id_to_word, checkpoint = load_model(
        device=device, checkpoint_path=checkpoint_path
    )
    active_device = next(model.parameters()).device
    model.eval()

    run_name = config.get("run_name", checkpoint_path.stem)
    encoder_type = config.get("encoder_type", "temporal")
    max_caption_length = config.get("max_caption_length", 30)

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = DEFAULT_RESULTS_DIR / run_name

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    print(f"Visualizing attention for run: {run_name} (encoder: {encoder_type})")
    print(f"Saving visualizations to: {out_dir}")

    for video_id in args.video_ids:
        feature_path = DEFAULT_FEATURE_DIR / f"{video_id}.npy"
        if not feature_path.is_file():
            print(f"Warning: Feature file not found for {video_id} at {feature_path}. Skipping.")
            continue

        features = np.load(feature_path, allow_pickle=False)
        caption, attention_history = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_caption_length,
            device=active_device,
        )

        words = caption.split()
        if not words:
            words = ["<eos>"]
            step_attns = attention_history[:1] if attention_history else [torch.zeros(features.shape[0])]
        else:
            step_attns = attention_history[: len(words)]

        stacked_attn = torch.stack(step_attns)  # [num_words, num_positions]
        num_words, num_positions = stacked_attn.shape

        print(f"  [{video_id}] Generated caption: \"{caption}\" ({num_words} words)")

        if encoder_type == "spatiotemporal_attention":
            # Reshape [num_words, 392] -> [num_words, 8, 7, 7]
            reshaped = reshape_attention_weights(stacked_attn, num_frames=8, height=7, width=7)
            # Marginalized temporal frame attention: sum over 7x7 spatial grid
            frame_attn = reshaped.sum(dim=(-2, -1)).numpy()  # [num_words, 8]

            # 1. Temporal heatmap
            heatmap_path = out_dir / f"{video_id}_temporal_attention.png"
            plot_frame_attention_heatmap(frame_attn, words, video_id, caption, heatmap_path)

            # 2. Spatial overlay on sampled video frames
            manifest_rec = manifest.get(video_id, {})
            frames_rgb, actual_indices = extract_sampled_frames(video_id, manifest_rec)

            overlay_path = out_dir / f"{video_id}_spatial_overlay.png"
            plot_spatiotemporal_spatial_overlay(
                reshaped.numpy(),
                frames_rgb,
                actual_indices,
                words,
                video_id,
                caption,
                overlay_path,
            )

            # Primary visualization output file
            primary_path = out_dir / f"{video_id}.png"
            plot_spatiotemporal_spatial_overlay(
                reshaped.numpy(),
                frames_rgb,
                actual_indices,
                words,
                video_id,
                caption,
                primary_path,
            )
            print(f"    Saved: {heatmap_path.name}, {overlay_path.name}, and {primary_path.name}")

        else:
            # Temporal encoder (or temporal_bigru): positions correspond directly to 8 frames
            attn_matrix = stacked_attn.numpy()  # [num_words, 8]
            primary_path = out_dir / f"{video_id}.png"
            temporal_path = out_dir / f"{video_id}_temporal_attention.png"

            plot_frame_attention_heatmap(attn_matrix, words, video_id, caption, primary_path)
            plot_frame_attention_heatmap(attn_matrix, words, video_id, caption, temporal_path)
            print(f"    Saved: {primary_path.name}")


if __name__ == "__main__":
    main()
