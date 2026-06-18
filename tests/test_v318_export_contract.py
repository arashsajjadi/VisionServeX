# SPDX-License-Identifier: Apache-2.0
"""v3.18 export capability contract (weight-free).

The live ONNX export is exercised in the train-lifecycle live matrix; here we
only enforce that the advertised export capability is well-formed and consistent.
"""

from __future__ import annotations

from visionservex import VisionModel
from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
_KNOWN_FORMATS = {
    "onnx",
    "torchscript",
    "ts",
    "openvino",
    "trt",
    "tensorrt",
    "coreml",
    "hf_save_pretrained",
}


def test_export_supported_is_a_list_of_known_formats():
    for mid, cap in CAPS.items():
        exp = cap["export_supported"]
        assert isinstance(exp, list), mid
        for fmt in exp:
            assert fmt.lower() in _KNOWN_FORMATS, (mid, fmt)


def test_export_method_exists():
    assert hasattr(VisionModel, "export")


def test_models_with_export_advertise_export_syntax():
    for mid, cap in CAPS.items():
        if cap["export_supported"]:
            assert "export" in cap["validated_syntax"], mid


def test_train_ready_live_detectors_and_classifiers_export_onnx():
    # The live lifecycle proved ONNX export for the libreyolo + torchvision
    # TRAIN_READY_LIVE set — capability must reflect it.
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert any(f.lower() == "onnx" for f in cap["export_supported"]), mid
