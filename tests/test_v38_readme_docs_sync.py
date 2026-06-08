# SPDX-License-Identifier: Apache-2.0
"""v3.8 — README + docs are synced with the HF BYOT / license policy surface."""

from __future__ import annotations

from pathlib import Path

import pytest

README = Path("README.md")
DOCS = Path("docs")

REQUIRED_DOCS = [
    "huggingface_connection.md",
    "model_license_policy.md",
    "byot_models.md",
    "restricted_models.md",
    "commercial_safe_core.md",
    "anastig_saas_policy.md",
]


@pytest.mark.parametrize("name", REQUIRED_DOCS)
def test_required_doc_exists(name):
    assert (DOCS / name).exists(), f"docs/{name} missing"


def test_readme_has_byot_and_nonredistribution_statement():
    text = README.read_text()
    assert "pip install visionservex" in text
    assert "does not redistribute" in text.lower()
    assert "visionservex hf" in text  # HF connection surface documented
    # mentions both BYOT and restricted classes
    assert "BYOT" in text or "Bring Your Own Token" in text


def test_huggingface_connection_doc_covers_revocation_and_oauth():
    text = (DOCS / "huggingface_connection.md").read_text()
    for needle in ("revoke", "OAuth", "Anastig", "logout"):
        assert needle.lower() in text.lower(), f"huggingface_connection.md missing '{needle}'"


def test_restricted_doc_has_warning_classes():
    text = (DOCS / "restricted_models.md").read_text().lower()
    assert "non-commercial" in text
    assert "agpl" in text or "enterprise" in text
    assert "external api" in text or "api" in text
