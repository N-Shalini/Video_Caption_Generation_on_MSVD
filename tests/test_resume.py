import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_refuse_overwrite_without_resume(tmp_path):
    run_name = "test_guard_refuse"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_file = ckpt_dir / f"{run_name}_best.pt"
    last_file = ckpt_dir / f"{run_name}_last.pt"

    # Create dummy checkpoint
    torch.save({"dummy": True}, last_file)

    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train_spatiotemporal.py"),
            "--run-name",
            run_name,
            "--epochs",
            "1",
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        assert res.returncode != 0
        assert "Refusing to overwrite existing checkpoint(s) without --resume" in res.stderr
        assert str(last_file) in res.stderr
    finally:
        if last_file.exists():
            last_file.unlink()
        if best_file.exists():
            best_file.unlink()


def test_resume_missing_checkpoint_fails():
    run_name = "test_nonexistent_run"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train_spatiotemporal.py"),
        "--run-name",
        run_name,
        "--resume",
    ]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert res.returncode != 0
    assert f"Cannot resume run '{run_name}': checkpoint not found" in res.stderr


def test_last_checkpoint_contains_all_state_and_resumes():
    run_name = "test_resume_integration"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    res_dir = PROJECT_ROOT / "results"
    best_file = ckpt_dir / f"{run_name}_best.pt"
    last_file = ckpt_dir / f"{run_name}_last.pt"
    hist_file = res_dir / f"{run_name}_history.json"
    cfg_file = res_dir / f"{run_name}_config_resolved.json"

    # Clean up before run
    for p in [best_file, last_file, hist_file, cfg_file]:
        if p.exists():
            p.unlink()

    try:
        # Phase 1: Train for 1 epoch
        cmd1 = [
            sys.executable,
            str(PROJECT_ROOT / "train_spatiotemporal.py"),
            "--run-name",
            run_name,
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-train-videos",
            "2",
            "--max-val-videos",
            "2",
        ]
        res1 = subprocess.run(cmd1, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        assert res1.returncode == 0, f"Phase 1 failed: {res1.stderr}"

        # Verify stdout formatting
        assert "train loss:" in res1.stdout
        assert "val loss:" in res1.stdout
        assert "val CIDEr:" in res1.stdout
        assert "BLEU-4:" in res1.stdout
        assert "METEOR:" in res1.stdout
        assert "ROUGE-L:" in res1.stdout
        assert "learning rate:" in res1.stdout
        assert "seconds:" in res1.stdout

        # Verify last_file exists and contains all required items
        assert last_file.is_file(), "last checkpoint must exist"
        ckpt = torch.load(last_file, map_location="cpu", weights_only=False)

        required_keys = [
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "best_metric_val",
            "epochs_without_improvement",
            "history",
        ]
        for key in required_keys:
            assert key in ckpt, f"Key '{key}' missing from _last.pt checkpoint"

        assert ckpt["epoch"] == 1
        assert len(ckpt["history"]) == 1

        # Verify history file
        assert hist_file.is_file(), "history json file must exist"
        with hist_file.open("r", encoding="utf-8") as f:
            hist_data = json.load(f)
        assert len(hist_data) == 1
        for field in [
            "epoch",
            "train_loss",
            "val_loss",
            "val_cider",
            "val_bleu4",
            "val_meteor",
            "val_rouge_l",
            "learning_rate",
            "seconds",
        ]:
            assert field in hist_data[0], f"Field '{field}' missing from history record"

        # Phase 2: Resume to 2 epochs
        cmd2 = [
            sys.executable,
            str(PROJECT_ROOT / "train_spatiotemporal.py"),
            "--run-name",
            run_name,
            "--resume",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--max-train-videos",
            "2",
            "--max-val-videos",
            "2",
        ]
        res2 = subprocess.run(cmd2, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        assert res2.returncode == 0, f"Phase 2 resume failed: {res2.stderr}"

        assert "Resuming at epoch 2 (completed epoch 1)" in res2.stdout
        assert "Epoch 02/02" in res2.stdout

        # Verify resumed checkpoint
        ckpt2 = torch.load(last_file, map_location="cpu", weights_only=False)
        assert ckpt2["epoch"] == 2
        assert len(ckpt2["history"]) == 2

        with hist_file.open("r", encoding="utf-8") as f:
            hist_data2 = json.load(f)
        assert len(hist_data2) == 2
        assert hist_data2[0]["epoch"] == 1
        assert hist_data2[1]["epoch"] == 2

    finally:
        for p in [best_file, last_file, hist_file, cfg_file]:
            if p.exists():
                p.unlink()
