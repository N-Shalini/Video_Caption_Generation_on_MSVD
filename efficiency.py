import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from generate_caption import load_model, greedy_decode, beam_search_decode

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "msvd_test.json"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_cpu_model():
    """Retrieve human-readable CPU model string across platforms."""
    # 1. Windows registry
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if name and name.strip():
                return name.strip()
        except Exception:
            pass

    # 2. Linux /proc/cpuinfo
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # 3. Fallbacks
    proc = platform.processor()
    if proc and proc.strip():
        return proc.strip()

    return os.environ.get("PROCESSOR_IDENTIFIER", "Unknown CPU")


def count_frozen_vgg16_parameters():
    """Return frozen VGG16 feature extractor parameter count (torchvision vgg16 features)."""
    try:
        from torchvision.models import vgg16
        features = vgg16(weights=None).features
        return sum(p.numel() for p in features.parameters())
    except Exception:
        # Canonical torchvision vgg16.features parameter count
        return 14714688


class ProcessMemorySampler:
    """Continuously samples process memory RSS in background to record peak usage."""

    def __init__(self, interval_sec=0.005):
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
        except Exception:
            pass
        return self.peak_rss


def main():
    parser = argparse.ArgumentParser(
        description="Profile model efficiency: parameter counts, decoding latency, and peak memory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=100,
        help="Number of test videos for latency benchmarking (default: 100).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of warm-up runs prior to latency measurements (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output JSON path (default: results/{checkpoint_stem}_efficiency.json).",
    )

    args = parser.parse_args()

    # OMP_NUM_THREADS plumbing for CPU benchmarking
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        try:
            torch.set_num_threads(int(omp_threads))
        except ValueError:
            pass

    device = torch.device("cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = checkpoint_path.stem
    output_path = (
        Path(args.output).resolve()
        if args.output
        else RESULTS_DIR / f"{checkpoint_stem}_efficiency.json"
    )

    # Start memory sampling
    sampler = ProcessMemorySampler(interval_sec=0.005)
    sampler.start()

    # Load test annotations and sample features
    if not DEFAULT_ANNOTATIONS.is_file():
        raise FileNotFoundError(f"Test annotations not found at: {DEFAULT_ANNOTATIONS}")

    with DEFAULT_ANNOTATIONS.open("r", encoding="utf-8") as f:
        records = json.load(f)

    selected_records = records[: args.num_videos]
    test_features = []
    for r in selected_records:
        vid = r["video_id"]
        feat_path = DEFAULT_FEATURE_DIR / f"{vid}.npy"
        if not feat_path.is_file():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        feat = np.load(feat_path, allow_pickle=False)
        test_features.append(feat)

    # 1. Load model and count parameters
    model, config, vocab, id_to_word, checkpoint = load_model(
        device=device, checkpoint_path=checkpoint_path
    )
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_params = sum(p.numel() for p in model.parameters())
    vgg16_features_params = count_frozen_vgg16_parameters()

    max_caption_length = config.get("max_caption_length", 30)

    # 2. Warm-up runs
    first_feat = test_features[0]
    for _ in range(args.warmup_runs):
        greedy_decode(
            model=model,
            features=first_feat,
            id_to_word=id_to_word,
            max_length=max_caption_length,
            device=device,
        )
        beam_search_decode(
            model=model,
            features=first_feat,
            id_to_word=id_to_word,
            max_length=max_caption_length,
            device=device,
            beam_width=5,
        )

    # 3. Greedy decoding latency
    greedy_latencies = []
    for feat in test_features:
        t0 = time.perf_counter()
        greedy_decode(
            model=model,
            features=feat,
            id_to_word=id_to_word,
            max_length=max_caption_length,
            device=device,
        )
        greedy_latencies.append(time.perf_counter() - t0)

    mean_greedy_sec = float(np.mean(greedy_latencies))
    mean_greedy_ms = mean_greedy_sec * 1000.0

    # 4. Beam-5 decoding latency
    beam5_latencies = []
    for feat in test_features:
        t0 = time.perf_counter()
        beam_search_decode(
            model=model,
            features=feat,
            id_to_word=id_to_word,
            max_length=max_caption_length,
            device=device,
            beam_width=5,
        )
        beam5_latencies.append(time.perf_counter() - t0)

    mean_beam5_sec = float(np.mean(beam5_latencies))
    mean_beam5_ms = mean_beam5_sec * 1000.0

    # Stop memory sampler
    peak_memory_bytes = sampler.stop()
    peak_memory_mb = round(peak_memory_bytes / (1024 * 1024), 2)

    cpu_model = get_cpu_model()

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_stem": checkpoint_stem,
        "cpu_model": cpu_model,
        "omp_num_threads": omp_threads,
        "torch_num_threads": torch.get_num_threads(),
        "trainable_parameters": trainable_params,
        "total_model_parameters": total_model_params,
        "frozen_vgg16_features_parameters": vgg16_features_params,
        "num_test_videos": len(test_features),
        "warmup_runs": args.warmup_runs,
        "greedy_latency_seconds_per_video": round(mean_greedy_sec, 6),
        "greedy_latency_ms_per_video": round(mean_greedy_ms, 3),
        "beam5_latency_seconds_per_video": round(mean_beam5_sec, 6),
        "beam5_latency_ms_per_video": round(mean_beam5_ms, 3),
        "peak_process_memory_bytes": peak_memory_bytes,
        "peak_process_memory_mb": peak_memory_mb,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"EFFICIENCY REPORT: {checkpoint_stem}")
    print(f"CPU Model: {cpu_model}")
    print(f"Threads: {torch.get_num_threads()} (OMP_NUM_THREADS={omp_threads})")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen VGG16 Features Parameters: {vgg16_features_params:,}")
    print(f"Test Videos Evaluated: {len(test_features)} (Warm-up: {args.warmup_runs})")
    print(f"Greedy Latency: {mean_greedy_ms:.2f} ms/video ({mean_greedy_sec:.4f} s/video)")
    print(f"Beam-5 Latency: {mean_beam5_ms:.2f} ms/video ({mean_beam5_sec:.4f} s/video)")
    print(f"Peak Process Memory: {peak_memory_mb:.2f} MB ({peak_memory_bytes:,} bytes)")
    print(f"Report written to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
