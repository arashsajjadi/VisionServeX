#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Notebook manifest consumption script — bridge between audit and future Colab.

Loads docs/audit/visionservex_notebook_input_manifest.json and proves it can
drive an external client. Run this before writing the Colab notebook.

Usage:
    python scripts/test_notebook_manifest_consumption.py
    python scripts/test_notebook_manifest_consumption.py --json
    python scripts/test_notebook_manifest_consumption.py --path path/to/manifest.json

Exit codes:
    0   manifest is valid and consumption test passed
    1   manifest is invalid or failed a structural check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"MANIFEST_NOT_FOUND: {p}", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text())


def check(condition: bool, msg: str, failures: list[str]) -> None:
    if not condition:
        failures.append(msg)


def main(manifest_path: str, as_json: bool = False) -> int:
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    report: dict = {}

    # ─── Package metadata ─────────────────────────────────────────────────────
    pkg = manifest.get("package", {})
    check(pkg.get("version"), "package.version missing", failures)
    check(pkg.get("load_matrix_rows", 0) >= 100, "too few load_matrix_rows", failures)
    report["package"] = pkg

    # ─── Model counts by category ─────────────────────────────────────────────
    models = manifest.get("models", [])
    check(len(models) >= 100, f"too few models: {len(models)}", failures)

    quick_models = [
        m
        for m in models
        if m.get("recommended_colab_mode") == "quick" and m.get("expected_load_mode") == "core_load"
    ]
    core_models = [m for m in models if m.get("expected_load_mode") == "core_load"]
    sidecar_models = [m for m in models if m.get("requires_sidecar")]
    gated_models = [m for m in models if m.get("requires_auth")]
    uc_models = [m for m in models if m.get("eligible_for_ultralytics_comparison")]

    report["model_counts"] = {
        "total": len(models),
        "core_load": len(core_models),
        "quick_colab": len(quick_models),
        "sidecar": len(sidecar_models),
        "gated": len(gated_models),
        "ultralytics_comparable": len(uc_models),
    }

    # ─── Every model must have non-empty smoke_command or expected_blocker ────
    broken = [
        m["model_id"]
        for m in models
        if not m.get("smoke_command") and not m.get("expected_blocker_code")
    ]
    check(
        not broken,
        f"models missing both smoke_command and expected_blocker_code: {broken[:5]}",
        failures,
    )

    # ─── Every model must have recommended_colab_mode ─────────────────────────
    no_mode = [m["model_id"] for m in models if not m.get("recommended_colab_mode")]
    check(not no_mode, f"models missing recommended_colab_mode: {no_mode[:5]}", failures)

    # ─── Gated/non-core/sidecar models must not be recommended as default ─────
    auto_run_bad = [
        m["model_id"]
        for m in models
        if (
            m.get("requires_auth")
            or m.get("license_risk") in ("gpl", "agpl", "restricted", "proprietary")
        )
        and m.get("recommended_colab_mode") not in ("sidecar", "full")
    ]
    check(
        not auto_run_bad,
        f"gated/non-core models should not be quick/balanced: {auto_run_bad[:5]}",
        failures,
    )

    # ─── Expected blockers ────────────────────────────────────────────────────
    blockers = manifest.get("expected_blockers", [])
    check(len(blockers) >= 5, f"too few expected_blockers: {len(blockers)}", failures)
    for b in blockers:
        check(b.get("code"), f"blocker missing code: {b}", failures)
    report["blockers"] = [b.get("code") for b in blockers]

    # ─── Notebook sections ────────────────────────────────────────────────────
    sections = manifest.get("notebook_sections", [])
    check(len(sections) >= 10, f"too few notebook_sections: {len(sections)}", failures)
    report["sections"] = [s.get("id") for s in sections]

    # ─── Sidecars ─────────────────────────────────────────────────────────────
    sidecars = manifest.get("sidecars", [])
    check(len(sidecars) >= 3, f"too few sidecars: {len(sidecars)}", failures)
    report["sidecars"] = [s.get("name") for s in sidecars]

    # ─── Optional extras ──────────────────────────────────────────────────────
    extras = manifest.get("optional_extras", [])
    check(len(extras) >= 3, f"too few optional_extras: {len(extras)}", failures)
    report["optional_extras"] = [e.get("name") for e in extras]

    # ─── Ultralytics comparison ───────────────────────────────────────────────
    uc = manifest.get("ultralytics_comparison", {})
    check(
        len(uc.get("eligible_models", [])) >= 5,
        "too few eligible_models for Ultralytics comparison",
        failures,
    )
    check(len(uc.get("caveats", [])) >= 3, "too few Ultralytics comparison caveats", failures)
    report["ultralytics_eligible"] = uc.get("eligible_models", [])

    # ─── Result ───────────────────────────────────────────────────────────────
    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    report["quick_models"] = [m["model_id"] for m in quick_models]

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Notebook Manifest Consumption Test — {report['verdict']}")
        print(f"{'=' * 60}")
        print(f"  Total models:       {report['model_counts']['total']}")
        print(f"  Core runnable:      {report['model_counts']['core_load']}")
        print(f"  Quick Colab:        {report['model_counts']['quick_colab']}")
        print(f"  Sidecar:            {report['model_counts']['sidecar']}")
        print(f"  Gated:              {report['model_counts']['gated']}")
        print(f"  Ultralytics-comp:   {report['model_counts']['ultralytics_comparable']}")
        print(f"\n  Sections: {', '.join(report['sections'][:5])}...")
        print(f"  Blockers: {', '.join(report['blockers'][:3])}...")
        print(f"  Sidecars: {', '.join(report['sidecars'])}")
        print(f"\n  Quick-mode models ({len(quick_models)}):")
        for m in quick_models[:8]:
            mid = m["model_id"] if isinstance(m, dict) else m
            print(f"    - {mid}")
        if len(quick_models) > 8:
            print(f"    ... and {len(quick_models) - 8} more")
        if failures:
            print(f"\n  FAILURES ({len(failures)}):")
            for f in failures:
                print(f"    ✗ {f}")
        else:
            print("\n  ✓ All checks passed")
        print()

    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test notebook manifest consumption")
    parser.add_argument("--path", default="docs/audit/visionservex_notebook_input_manifest.json")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    sys.exit(main(args.path, args.as_json))
