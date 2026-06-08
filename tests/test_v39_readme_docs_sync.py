# SPDX-License-Identifier: Apache-2.0
"""v3.9 — README and docs contain required BYOT statements and v3.9.0 marker."""

from __future__ import annotations

from pathlib import Path


def test_readme_mentions_v390():
    readme = Path("README.md").read_text()
    assert "3.9.0" in readme or "v3.9" in readme, "README missing v3.9.0 version marker"


def test_readme_contains_nonredistribution_statement():
    readme = Path("README.md").read_text()
    assert "never redistributes" in readme.lower() or "not redistribute" in readme.lower(), (
        "README missing non-redistribution statement"
    )


def test_readme_contains_sam3_mention():
    readme = Path("README.md").read_text()
    assert "SAM3" in readme or "sam3" in readme.lower()


def test_readme_contains_dinov3_mention():
    readme = Path("README.md").read_text()
    assert "DINOv3" in readme or "dinov3" in readme.lower()


def test_readme_contains_byot_section():
    readme = Path("README.md").read_text()
    assert "BYOT" in readme or "Bring Your Own Token" in readme


def test_changelog_contains_v390():
    changelog = Path("CHANGELOG.md").read_text()
    assert "3.9.0" in changelog, "CHANGELOG missing v3.9.0 entry"


def test_docs_byot_models_exists():
    doc = Path("docs/byot_models.md")
    assert doc.exists(), "Missing docs/byot_models.md"
    text = doc.read_text()
    assert "SAM3" in text or "sam3" in text.lower()
    assert "DINOv3" in text or "dinov3" in text.lower()


def test_docs_model_license_policy_exists():
    doc = Path("docs/model_license_policy.md")
    assert doc.exists()
    text = doc.read_text()
    assert "byot_license_required" in text


def test_docs_hf_connection_exists():
    doc = Path("docs/huggingface_connection.md")
    assert doc.exists()
    text = doc.read_text()
    assert "token" in text.lower()


def test_no_raw_token_in_readme():
    import re

    readme = Path("README.md").read_text()
    tokens = re.findall(r"hf_[A-Za-z0-9]{10,}", readme)
    real = [t for t in tokens if not t.startswith("hf_***")]
    assert not real, f"Raw token in README: {real}"
