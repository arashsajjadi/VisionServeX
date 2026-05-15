"""CLI smoke tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


def test_cli_version():
    r = runner.invoke(app, ["version", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "version" in data


def test_cli_doctor_json():
    r = runner.invoke(app, ["doctor", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert "devices" in data
    assert any(d["name"] == "cpu" for d in data["devices"])


def test_cli_list_models_json():
    r = runner.invoke(app, ["list-models", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert any(m["id"] == "mock-detect" for m in data)


def test_cli_info_unknown_model():
    r = runner.invoke(app, ["info", "no-such-model", "--json"])
    assert r.exit_code != 0
    # error is on stderr; the runner combines them by default in newer typer
    out = r.output + (r.stderr if r.stderr else "")
    assert "MODEL_NOT_FOUND" in out or "unknown model" in out


def test_cli_predict_smoke(tmp_path):
    from PIL import Image

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (64, 48), "red").save(img_path, "JPEG")
    out = tmp_path / "r.json"
    r = runner.invoke(
        app,
        ["predict", "mock-detect", str(img_path), "--save", str(out), "--json"],
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["kind"] == "detection"
