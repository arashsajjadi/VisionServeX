# SPDX-License-Identifier: Apache-2.0
"""v3.18 catalog completeness — every model discovered exactly once, fully typed.

Weight-free.
"""

from __future__ import annotations

import collections

from visionservex.core.model import list_models, model_capabilities
from visionservex.registry import default_registry

ALL_IDS = list_models()
CAPS = {m: model_capabilities(m) for m in ALL_IDS}


def test_no_duplicate_model_ids():
    dupes = [m for m, n in collections.Counter(ALL_IDS).items() if n > 1]
    assert not dupes, f"duplicate model ids: {dupes}"


def test_list_models_equals_registry():
    reg_ids = sorted(e.id for e in default_registry().list())
    assert sorted(ALL_IDS) == reg_ids, "list_models() and the registry disagree"


def test_no_orphan_capability_or_registry_rows():
    reg_ids = {e.id for e in default_registry().list()}
    assert set(ALL_IDS) == reg_ids
    assert set(CAPS) == reg_ids


def test_every_model_has_task_family_license_readiness():
    for mid, cap in CAPS.items():
        assert cap["task"], f"{mid} missing task"
        assert cap["family"], f"{mid} missing family"
        assert cap["license"], f"{mid} missing license"
        assert cap["readiness"], f"{mid} missing readiness"
        assert cap["readiness_state"], f"{mid} missing readiness_state"


def test_filtered_list_models_is_a_subset():
    for task in {c["task"] for c in CAPS.values()}:
        sub = list_models(task=task)
        assert set(sub) <= set(ALL_IDS)
        assert all(CAPS[m]["task"] == task for m in sub)


def test_catalog_size_is_reasonable():
    # Guard against accidental mass-drop/duplication of the registry.
    assert len(ALL_IDS) >= 140, f"catalog unexpectedly small: {len(ALL_IDS)}"
