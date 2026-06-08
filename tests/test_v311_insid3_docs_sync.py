# SPDX-License-Identifier: Apache-2.0
"""v3.11: INSID3 docs sync — insid3.md exists, README mentions INSID3, no token leaks."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
README = ROOT / "README.md"


def test_insid3_doc_exists():
    doc = DOCS / "insid3.md"
    assert doc.exists(), "docs/insid3.md must exist for v3.11.0"


def test_insid3_doc_mentions_apache_license():
    doc = DOCS / "insid3.md"
    if not doc.exists():
        pytest.skip("docs/insid3.md not yet created")
    text = doc.read_text()
    assert "Apache-2.0" in text, "docs/insid3.md must mention Apache-2.0 (INSID3 code license)"


def test_insid3_doc_mentions_dinov3_license():
    doc = DOCS / "insid3.md"
    if not doc.exists():
        pytest.skip("docs/insid3.md not yet created")
    text = doc.read_text()
    assert "DINOv3" in text, "docs/insid3.md must mention DINOv3"
    assert "Built with DINOv3" in text, (
        "docs/insid3.md must mention 'Built with DINOv3' attribution"
    )


def test_readme_mentions_insid3():
    if not README.exists():
        pytest.skip("README.md not found")
    text = README.read_text()
    assert "INSID3" in text or "insid3" in text.lower(), (
        "README.md must mention INSID3 after v3.11.0"
    )


def test_insid3_doc_no_token_literals():
    import re

    token_pattern = re.compile(r"hf_[A-Za-z0-9]{15,}")
    for doc_path in [DOCS / "insid3.md", DOCS / "insid3_anastig_workflow.md"]:
        if not doc_path.exists():
            continue
        text = doc_path.read_text()
        matches = token_pattern.findall(text)
        assert not matches, f"HF token literal in {doc_path.name}: {matches}"


def test_changelog_311_entry():
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("CHANGELOG.md not found")
    text = changelog.read_text()
    assert "3.11.0" in text, "CHANGELOG.md must have a 3.11.0 entry"


def test_byot_models_doc_mentions_insid3():
    byot_doc = DOCS / "byot_models.md"
    if not byot_doc.exists():
        pytest.skip("docs/byot_models.md not found")
    text = byot_doc.read_text()
    assert "insid3" in text.lower() or "INSID3" in text, (
        "docs/byot_models.md must mention INSID3 after v3.11.0"
    )
