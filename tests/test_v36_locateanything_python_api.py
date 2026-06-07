# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything Python API tests.

Tests VSX.locateanything() factory, _LocateAnythingHandle methods,
and that locate() raises VSXError when accept_noncommercial=False.
"""

from __future__ import annotations

import pytest


def test_vsx_locateanything_factory_exists() -> None:
    from visionservex.vsx import VSX

    h = VSX.locateanything("locate-anything-3b")
    assert h is not None
    assert h.family == "locate_anything"


def test_locateanything_handle_model_id() -> None:
    from visionservex.vsx import VSX

    h = VSX.locateanything("locate-anything-3b-v2")
    assert h.model_id == "locate-anything-3b-v2"


def test_explain_returns_required_keys() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    required = {
        "model_id", "family", "task", "state", "license", "default_safe",
        "commercial_safe", "install_extra", "auth_required", "byot",
        "warning", "sidecar_install", "limitations", "next_command", "tutorial",
    }
    for k in required:
        assert k in info, f"Missing key in explain(): {k!r}"


def test_explain_default_safe_is_false() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    assert info["default_safe"] is False


def test_explain_commercial_safe_is_false() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    assert info["commercial_safe"] is False


def test_explain_byot_true() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    assert info["byot"] is True


def test_locate_without_accept_noncommercial_raises_vsxerror() -> None:
    """locate() must raise VSXError if accept_noncommercial is not True."""
    from PIL import Image

    from visionservex.vsx import VSX, VSXError

    h = VSX.locateanything("locate-anything-3b")
    img = Image.new("RGB", (64, 64))
    with pytest.raises(VSXError) as exc_info:
        h.locate(img, text="cat", accept_noncommercial=False)
    err = exc_info.value
    assert err.state == "excluded_restricted"
    assert "accept_noncommercial" in str(err).lower() or "noncommercial" in str(err).lower()


def test_locate_with_accept_noncommercial_attempts_sidecar(capsys) -> None:
    """locate() with accept_noncommercial=True must print warning and attempt sidecar."""
    from PIL import Image

    from visionservex.vsx import VSX

    h = VSX.locateanything("locate-anything-3b")
    img = Image.new("RGB", (64, 64))
    try:
        h.locate(img, text="cat", accept_noncommercial=True)
    except Exception:
        pass  # RuntimeError from missing sidecar is expected
    captured = capsys.readouterr()
    # Warning must always be printed to stderr
    assert "WARNING" in captured.err or "NVIDIA" in captured.err


def test_locate_without_flag_prints_warning(capsys) -> None:
    from PIL import Image

    from visionservex.vsx import VSX, VSXError

    h = VSX.locateanything("locate-anything-3b")
    img = Image.new("RGB", (64, 64))
    with pytest.raises(VSXError):
        h.locate(img, text="cat", accept_noncommercial=False)
    captured = capsys.readouterr()
    # Warning still printed before the raise
    assert "WARNING" in captured.err or "NVIDIA" in captured.err


def test_install_extra_is_locateanything() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    assert "locateanything" in info["install_extra"]


def test_sidecar_install_references_eagle_repo() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    assert "NVlabs" in info["sidecar_install"] or "eagle" in info["sidecar_install"].lower()


def test_next_command_has_accept_noncommercial_flag() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    assert "--accept-noncommercial" in info["next_command"]
