import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_ROOT / "raw" / "YouTubeClips"
ANNOTATION_DIR = DATA_ROOT / "annotations"
FEATURE_DIR = DATA_ROOT / "features" / "vgg16_spatial"
MANIFEST_PATH = FEATURE_DIR / "manifest.jsonl"
ERROR_LOG_PATH = FEATURE_DIR / "errors.jsonl"

NUM_FRAMES = 8
BATCH_SIZE = 2
EXPECTED_FEATURE_SHAPE = (512, 7, 7)


def sample_frames(video_path, num_frames, preprocess):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")

    indices = np.linspace(
        0, frame_count - 1, num=min(num_frames, frame_count)
    ).round().astype(int)

    frames = []

    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {index}: {video_path.name}"
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb)))
    finally:
        cap.release()

    return torch.stack(frames), indices.tolist(), frame_count


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed():
    completed = {}

    if not MANIFEST_PATH.exists():
        return completed

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
                output_path = FEATURE_DIR / item["feature_file"]

                if output_path.is_file():
                    features = np.load(output_path, mmap_mode="r",
                                       allow_pickle=False)
                    expected = (item["num_frames"], *EXPECTED_FEATURE_SHAPE)

                    if features.shape == expected and np.isfinite(features).all():
                        completed[item["video_id"]] = item
            except Exception:
                # Invalid manifest entries are not treated as completed.
                continue

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many pending videos; 0 means all."
    )
    args = parser.parse_args()

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

    weights = VGG16_Weights.DEFAULT
    preprocess = weights.transforms()
    model = vgg16(weights=weights).features.eval().cpu()

    completed = load_completed()
    pending = [
        record for record in records
        if record["video_id"] not in completed
    ]

    if args.limit > 0:
        pending = pending[:args.limit]

    print("Device: CPU")
    print("Pretrained weights:", weights)
    print("Frame count target:", NUM_FRAMES)
    print("Frame batch size:", BATCH_SIZE)
    print("Annotated videos:", len(records))
    print("Previously validated features:", len(completed))
    print("Videos pending in this run:", len(pending))

    run_start = time.perf_counter()
    succeeded = 0
    failed = 0

    for position, record in enumerate(pending, start=1):
        video_path = VIDEO_DIR / record["video"]
        output_path = FEATURE_DIR / f"{record['video_id']}.npy"

        try:
            video_start = time.perf_counter()
            
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video: {video_path}")

            frame_batch, indices, source_frame_count = sample_frames(
                video_path, NUM_FRAMES, preprocess
            )

            feature_batches = []
            with torch.inference_mode():
                for start in range(0, len(frame_batch), BATCH_SIZE):
                    batch = frame_batch[start:start + BATCH_SIZE]
                    feature_batches.append(model(batch))

            features = torch.cat(feature_batches, dim=0).numpy()

            expected_shape = (
                len(indices), *EXPECTED_FEATURE_SHAPE
            )
            if features.shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape {features.shape}; expected {expected_shape}"
                )

            if not np.isfinite(features).all():
                raise ValueError("Features contain NaN or infinite values.")

            # Save to a temporary file first, then replace the destination.
            temp_path = output_path.with_suffix(".tmp.npy")
            np.save(temp_path, features)
            temp_path.replace(output_path)

            manifest_record = {
                "video_id": record["video_id"],
                "video": record["video"],
                "split": record["split"],
                "feature_file": output_path.name,
                "num_frames": len(indices),
                "sampled_frame_indices": indices,
                "source_frame_count": source_frame_count,
                "feature_shape": list(features.shape),
                "dtype": str(features.dtype),
                "backbone": "VGG-16",
                "weights": "VGG16_Weights.DEFAULT",
                "feature_layer": "features",
                "preprocessing": "VGG16_Weights.DEFAULT.transforms()",
                "elapsed_seconds": round(time.perf_counter() - video_start, 4),
            }

            append_jsonl(MANIFEST_PATH, manifest_record)
            succeeded += 1

        except Exception as exc:
            failed += 1
            append_jsonl(ERROR_LOG_PATH, {
                "video_id": record["video_id"],
                "video": record["video"],
                "split": record["split"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

        if position % 25 == 0 or position == len(pending):
            elapsed = time.perf_counter() - run_start
            print(
                f"Progress {position}/{len(pending)} | "
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
