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
