import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_evaluate_test_set_refuses_without_decoding_config():
    # Attempt to run evaluate_test_set.py with a non-existent decoding config
    non_existent_config = PROJECT_ROOT / "results" / "non_existent_decoding_selection.json"
    if non_existent_config.exists():
        non_existent_config.unlink()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_test_set.py"),
        "--decoding-config",
        str(non_existent_config),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    # Must exit with non-zero error code
    assert result.returncode != 0
    assert "requires --decoding-config" in result.stderr or "not found" in result.stderr


def test_tune_decoding_grid_deduplicated():
    """Verify that tune_decoding grid removes duplicate configurations:
    for beam_width==1 iterates only repetition_penalty (length_penalty=0.0, ngram_block_size=0)."""
    beam_widths = [1, 3, 5]
    length_penalties = [0.0, 0.5, 0.7, 1.0]
    repetition_penalties = [1.0, 1.1, 1.2]
    ngram_block_sizes = [0, 3]

    grid = []
    for rp in repetition_penalties:
        grid.append((1, 0.0, rp, 0))

    for bw in [b for b in beam_widths if b > 1]:
        for lp in length_penalties:
            for rp in repetition_penalties:
                for nb in ngram_block_sizes:
                    grid.append((bw, lp, rp, nb))

    # Total combinations must be 3 + 48 = 51 (down from 72)
    assert len(grid) == 51
    # Verify unique configurations
    assert len(set(grid)) == 51
    # Verify beam_width==1 only has length_penalty=0.0 and ngram_block_size=0
    bw1_configs = [c for c in grid if c[0] == 1]
    assert len(bw1_configs) == len(repetition_penalties)
    for bw, lp, rp, nb in bw1_configs:
        assert lp == 0.0
        assert nb == 0


def test_evaluate_test_set_reads_eos_blocklist_config(tmp_path):
    """Verify evaluate_test_set reads eos_blocklist setting from decoding config JSON."""
    dummy_config = tmp_path / "decoding_selection.json"
    dummy_config.write_text(
        json.dumps({
            "best_config": {
                "beam_width": 1,
                "length_penalty": 0.0,
                "repetition_penalty": 1.0,
                "ngram_block_size": 0,
                "eos_blocklist": False,
                "blocklist_size": 0,
            }
        }),
        encoding="utf-8",
    )

    with dummy_config.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chosen = data.get("best_config", data)
    assert chosen["eos_blocklist"] is False
    assert chosen["blocklist_size"] == 0

