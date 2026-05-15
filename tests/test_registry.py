"""Registry validation."""

from __future__ import annotations

import pytest

from visionservex.registry import ModelEntry, ModelRegistry, RegistryError, default_registry


def test_default_registry_loads():
    reg = default_registry()
    assert len(reg) > 0
    ids = [e.id for e in reg.list()]
    assert "mock-detect" in ids
    assert "dfine-s" in ids
    assert "grounding-dino-tiny" in ids
    assert "rfdetr-small" in ids
    assert "sam2-hiera-tiny" in ids


def test_registry_get_unknown_raises():
    reg = default_registry()
    with pytest.raises(RegistryError):
        reg.get("does-not-exist")


def test_registry_filter_by_task():
    reg = default_registry()
    detect = reg.list(task="detect")
    assert all(e.task == "detect" for e in detect)
    assert "mock-detect" in [e.id for e in detect]


def test_register_duplicate_raises():
    reg = ModelRegistry()
    entry = ModelEntry(
        id="x", display_name="X", task="detect", family="f",
        license="Apache-2.0", upstream_url="https://example.com", engine="mock",
    )
    reg.register(entry)
    with pytest.raises(RegistryError):
        reg.register(entry)
    reg.register(entry, replace=True)


def test_invalid_id_rejected():
    with pytest.raises(ValueError):
        ModelEntry(
            id="bad id!",
            display_name="X",
            task="detect",
            family="f",
            license="Apache-2.0",
            upstream_url="https://example.com",
            engine="mock",
        )


def test_unknown_status_rejected():
    with pytest.raises(Exception):
        ModelEntry(
            id="x",
            display_name="X",
            task="detect",
            family="f",
            license="Apache-2.0",
            upstream_url="https://example.com",
            engine="mock",
            status="bogus",  # type: ignore[arg-type]
        )


def test_every_default_model_has_license():
    for e in default_registry().list():
        assert e.license, f"model {e.id} missing license"
        assert e.upstream_url, f"model {e.id} missing upstream_url"
        assert e.engine, f"model {e.id} missing engine"
