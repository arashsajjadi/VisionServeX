# SPDX-License-Identifier: Apache-2.0
"""v3.20: no copyleft/non-commercial runtime path; no Ultralytics on runtime/training.

Weight-free.
"""

from __future__ import annotations

from pathlib import Path

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
_SRC = Path(__file__).resolve().parents[1] / "src" / "visionservex"
_RUNTIME_DIRS = [
    "engines",
    "core",
    "data",
    "runtime",
    "runtime_broker",
    "registry",
    "readiness",
    "licensing",
    "models",
    "model_zoo",
    "training",
]


def test_no_copyleft_or_noncommercial_is_commercial_safe():
    for mid, c in CAPS.items():
        if c["license_class"] in ("copyleft", "noncommercial"):
            assert not c["commercial_safe"], mid
            assert c["readiness_state"] in (
                taxonomy.LICENSE_BLOCKED,
                taxonomy.NON_COMMERCIAL_BLOCKED,
            ), mid


def test_classifier_detects_copyleft():
    assert taxonomy.classify_license("AGPL-3.0") == "copyleft"
    assert taxonomy.classify_license("GPL-3.0") == "copyleft"
    assert taxonomy.classify_license("Apache-2.0") == "permissive"


def test_no_ultralytics_on_runtime_or_training_path():
    offenders = []
    for d in _RUNTIME_DIRS:
        dd = _SRC / d
        if not dd.exists():
            continue
        for f in dd.rglob("*.py"):
            text = f.read_text()
            if "import ultralytics" in text or "from ultralytics" in text:
                offenders.append(str(f))
    assert not offenders, f"Ultralytics on runtime/training path: {offenders}"


def test_new_training_module_is_permissive_only():
    # The v3.20 embedding-finetune training module must not import any copyleft dep.
    text = (_SRC / "training" / "embedding_finetune.py").read_text()
    for bad in ("ultralytics", "mmcv", "mmdet"):
        assert bad not in text, bad
