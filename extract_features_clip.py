import argparse
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_ROOT / "raw" / "YouTubeClips"
ANNOTATION_DIR = DATA_ROOT / "annotations"
VGG_FEATURE_DIR = DATA_ROOT / "features" / "vgg16_spatial"
VGG_MANIFEST_PATH = VGG_FEATURE_DIR / "manifest.jsonl"
FEATURE_DIR = DATA_ROOT / "features" / "clip_vitb32"
MANIFEST_PATH = FEATURE_DIR / "manifest.jsonl"
ERROR_LOG_PATH = FEATURE_DIR / "errors.jsonl"

NUM_FRAMES = 8
FRAME_BATCH_SIZE = 16
EXPECTED_FEATURE_SHAPE = (8, 512, 1, 1)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_vgg_manifest(path=VGG_MANIFEST_PATH):
    """Load sampled_frame_indices per video_id from the VGG16 manifest."""
    mapping = {}
    if not path.is_file():
        raise FileNotFoundError(f"VGG manifest not found at: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                mapping[item["video_id"]] = item
            except Exception:
                continue
    return mapping


def load_completed(manifest_path=MANIFEST_PATH, feature_dir=FEATURE_DIR):
    """Load validated completed video records from the CLIP manifest."""
    completed = {}
    if not manifest_path.exists():
        return completed

    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                vid = item["video_id"]
                output_path = feature_dir / item.get("feature_file", f"{vid}.npy")

                if output_path.is_file():
                    features = np.load(output_path, allow_pickle=False)
                    if (
                        features.shape == EXPECTED_FEATURE_SHAPE
                        and features.dtype == np.float32
                        and np.isfinite(features).all()
                    ):
                        completed[vid] = item
            except Exception:
                continue

    return completed


def decode_frames(video_path, indices, preprocess):
    """Decode specified frame indices from video using cv2 and apply preprocessing."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode frame {index}: {video_path.name}"
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    finally:
        cap.release()

    return torch.stack(frames), frame_count


def load_clip_encoder():
    """Load CLIP ViT-B-32-quickgelu encoder and preprocessing (preferring open_clip_torch, fallback to transformers)."""
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
        model = model.eval().cpu()
        model_name = "ViT-B-32-quickgelu"
        weights_name = "openai"

        def encode_fn(batch_tensors):
            # batch_tensors: [N, 3, 224, 224]
            # model.encode_image returns projected 512-d embeddings
            return model.encode_image(batch_tensors)

        return encode_fn, preprocess, model_name, weights_name

    except Exception as e_clip:
        try:
            from transformers import CLIPModel, CLIPProcessor

            hf_id = "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(hf_id).eval().cpu()
            processor = CLIPProcessor.from_pretrained(hf_id)
            model_name = "clip-vit-base-patch32"
            weights_name = "openai/clip-vit-base-patch32"

            def preprocess_fn(pil_img):
                inputs = processor(images=pil_img, return_tensors="pt")
                return inputs["pixel_values"].squeeze(0)

            def encode_fn(batch_tensors):
                return model.get_image_features(pixel_values=batch_tensors)

            return encode_fn, preprocess_fn, model_name, weights_name

        except Exception as e_hf:
            raise RuntimeError(
                f"Could not initialize CLIP ViT-B/32 encoder. "
                f"open_clip error: {e_clip}; transformers error: {e_hf}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-B/32 image features for MSVD videos."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many pending videos; 0 means all.",
    )
    args = parser.parse_args()

    # OMP_NUM_THREADS plumbing for CPU execution
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        try:
            torch.set_num_threads(int(omp_threads))
        except ValueError:
            pass

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    annotation_files = [
        ANNOTATION_DIR / "msvd_train.json",
        ANNOTATION_DIR / "msvd_val.json",
        ANNOTATION_DIR / "msvd_test.json",
    ]

    records = []
    for path in annotation_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")

        split = path.stem.replace("msvd_", "")
        data = json.loads(path.read_text(encoding="utf-8"))

        for record in data:
            records.append({
                "split": split,
                "video_id": record["video_id"],
                "video": record["video"],
            })

    ids = [record["video_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs found across annotation splits.")

    vgg_manifest = load_vgg_manifest(VGG_MANIFEST_PATH)
    completed = load_completed(MANIFEST_PATH, FEATURE_DIR)

    pending = [
        record for record in records
        if record["video_id"] not in completed
    ]

    if args.limit > 0:
        pending = pending[: args.limit]

    print("Device: CPU")
    print("Backbone: CLIP ViT-B/32")
    print(f"Threads: {torch.get_num_threads()} (OMP_NUM_THREADS={omp_threads})")
    print("Frame count target:", NUM_FRAMES)
    print("Batch size (frames):", FRAME_BATCH_SIZE)
    print("Annotated videos:", len(records))
    print("Previously validated features:", len(completed))
    print("Videos pending in this run:", len(pending))

    if not pending:
        print("No pending videos to extract. Done.")
        return

    encode_fn, preprocess, model_name, weights_name = load_clip_encoder()
    print(f"Loaded CLIP model: {model_name} (weights: {weights_name})")

    run_start = time.perf_counter()
    succeeded = 0
    failed = 0

    # Process pending videos in chunks of 2 videos = 16 frames
    # (or 1 video = 8 frames for the remainder)
    chunk_size = FRAME_BATCH_SIZE // NUM_FRAMES  # 16 // 8 = 2 videos per batch
    if chunk_size < 1:
        chunk_size = 1

    for chunk_start in range(0, len(pending), chunk_size):
        chunk_records = pending[chunk_start : chunk_start + chunk_size]

        for record in chunk_records:
            vid = record["video_id"]
            video_path = VIDEO_DIR / record["video"]
            output_path = FEATURE_DIR / f"{vid}.npy"

            try:
                video_start = time.perf_counter()

                if vid not in vgg_manifest:
                    raise KeyError(f"Video {vid} not found in VGG manifest {VGG_MANIFEST_PATH}")

                sampled_indices = vgg_manifest[vid]["sampled_frame_indices"]
                if len(sampled_indices) != NUM_FRAMES:
                    raise ValueError(
                        f"Expected {NUM_FRAMES} sampled indices for {vid}, got {len(sampled_indices)}"
                    )

                if not video_path.is_file():
                    raise FileNotFoundError(f"Missing video file: {video_path}")

                frame_batch, source_frame_count = decode_frames(
                    video_path, sampled_indices, preprocess
                )  # [8, 3, H, W]

                # Encode in batches of up to 16 frames under inference mode
                with torch.inference_mode():
                    embs = []
                    for f_idx in range(0, len(frame_batch), FRAME_BATCH_SIZE):
                        sub_batch = frame_batch[f_idx : f_idx + FRAME_BATCH_SIZE]
                        emb = encode_fn(sub_batch)
                        embs.append(emb)

                    projected = torch.cat(embs, dim=0)  # [8, 512]

                    # L2-normalise each of the 8 projected 512-d embeddings
                    normed = projected / (projected.norm(p=2, dim=-1, keepdim=True) + 1e-12)

                    # Save float32 array of shape (8, 512, 1, 1)
                    features = (
                        normed.view(NUM_FRAMES, 512, 1, 1)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                if features.shape != EXPECTED_FEATURE_SHAPE:
                    raise ValueError(
                        f"Unexpected shape {features.shape}; expected {EXPECTED_FEATURE_SHAPE}"
                    )

                if not np.isfinite(features).all():
                    raise ValueError("Features contain NaN or infinite values.")

                # Atomic write
                temp_path = output_path.with_suffix(".tmp.npy")
                np.save(temp_path, features)
                temp_path.replace(output_path)

                manifest_record = {
                    "video_id": vid,
                    "video": record["video"],
                    "split": record["split"],
                    "feature_file": output_path.name,
                    "num_frames": len(sampled_indices),
                    "sampled_frame_indices": sampled_indices,
                    "source_frame_count": source_frame_count,
                    "feature_shape": list(features.shape),
                    "dtype": str(features.dtype),
                    "backbone": "CLIP-ViT-B/32",
                    "model": model_name,
                    "weights": weights_name,
                    "feature_layer": "visual_projection",
                    "preprocessing": f"{model_name}_{weights_name}_transforms",
                    "elapsed_seconds": round(time.perf_counter() - video_start, 4),
                }

                append_jsonl(MANIFEST_PATH, manifest_record)
                succeeded += 1

            except Exception as exc:
                failed += 1
                append_jsonl(
                    ERROR_LOG_PATH,
                    {
                        "video_id": vid,
                        "video": record["video"],
                        "split": record["split"],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

        done_so_far = min(chunk_start + len(chunk_records), len(pending))
        if done_so_far % 25 == 0 or done_so_far == len(pending):
            elapsed = time.perf_counter() - run_start
            print(
                f"Progress {done_so_far}/{len(pending)} | "
                f"success={succeeded} | failed={failed} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print("\nRun finished.")
    print("Succeeded:", succeeded)
    print("Failed:", failed)
    print("Elapsed seconds:", round(time.perf_counter() - run_start, 2))
    print("Feature directory:", FEATURE_DIR)
    print("Manifest:", MANIFEST_PATH)
    print("Error log:", ERROR_LOG_PATH)


if __name__ == "__main__":
    main()
