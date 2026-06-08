# SPDX-License-Identifier: Apache-2.0
"""v3.11: Ensure no INSID3 or DINOv3 weight files are tracked in git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_WEIGHT_EXTS = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".engine", ".trt")


def _git_tracked_files():
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        return result.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("git ls-files not available")


def test_no_weight_binaries_tracked():
    tracked = _git_tracked_files()
    bad = [f for f in tracked if any(f.endswith(ext) for ext in _WEIGHT_EXTS)]
    assert not bad, f"Binary weight files tracked in git: {bad}"


def test_no_hf_cache_dirs_tracked():
    tracked = _git_tracked_files()
    hf_cache_entries = [f for f in tracked if ".cache/huggingface" in f or "hub/models--" in f]
    assert not hf_cache_entries, f"HF cache entries tracked in git: {hf_cache_entries}"


def test_insid3_runtime_has_no_hardcoded_weights():
    runtime_path = ROOT / "src" / "visionservex" / "insid3_runtime.py"
    assert runtime_path.exists(), "insid3_runtime.py must exist"
    text = runtime_path.read_text()
    # Code must not actively download/load bundled weights (patterns indicate hard-coded URLs)
    bad_patterns = ["wget ", "curl http", "torch.hub.load", "download_url(", "urlretrieve("]
    for pat in bad_patterns:
        assert pat not in text, (
            f"insid3_runtime.py contains suspicious pattern {pat!r} — no weights should be bundled"
        )


def test_insid3_all_policy_rows_no_ship():
    from visionservex.licensing.policy import POLICY

    bad = [mid for mid, pol in POLICY.items() if pol.family == "insid3" and pol.can_ship_weights]
    assert not bad, f"INSID3 rows with can_ship_weights=True: {bad}"


def test_gitignore_blocks_weight_extensions():
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        pytest.skip(".gitignore not found")
    text = gitignore.read_text()
    for ext in ("*.pt", "*.pth", "*.onnx", "*.safetensors"):
        assert ext in text, f".gitignore must block {ext}"
