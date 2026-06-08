#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.8 release security guard — fails hard on any policy/secret violation.

Checks:
  1. No real HF token (`hf_<20+ chars>`) in any tracked file or the working diff.
  2. No weight binaries tracked by git (.onnx/.pt/.pth/.ckpt/.safetensors/.bin/.engine/.trt).
  3. No non-commercial / enterprise / AGPL model is default_safe in the policy.
  4. No HF token referenced in GitHub Actions workflows.
  5. README carries the BYOT / non-redistribution statement.

Writes notebook/99_final_report/reports/v38_security_scan.json and exits non-zero
on any failure.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPORT = Path("notebook/99_final_report/reports/v38_security_scan.json")
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
WEIGHT_EXT = (".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".engine", ".trt")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def main() -> int:
    findings: dict[str, object] = {}
    failures: list[str] = []

    tracked = [f for f in _git("ls-files").splitlines() if f]
    diff = _git("diff", "--", ".") + _git("diff", "--cached", "--", ".")

    # 1a. CRITICAL: the REAL active token must never appear anywhere.
    # Read it in-memory only (never printed); skip if not logged in.
    real_token = None
    try:
        from visionservex import hf_auth as _H

        real_token = _H.hf_get_token()
    except Exception:
        real_token = None
    real_token_hits: list[str] = []
    if real_token:
        for f in tracked:
            p = Path(f)
            if not p.is_file():
                continue
            try:
                if real_token in p.read_text(encoding="utf-8", errors="ignore"):
                    real_token_hits.append(f)
            except OSError:
                continue
        if real_token in diff:
            real_token_hits.append("<working-diff>")
    findings["real_token_present_locally"] = bool(real_token)
    findings["real_token_hits"] = real_token_hits
    if real_token_hits:
        failures.append(f"REAL HF token leaked into: {real_token_hits}")

    # 1b. token-pattern scan over shippable files (tests legitimately hold fake
    # redaction fixtures, so they are excluded from the PATTERN check only).
    token_hits: list[str] = []
    for f in tracked:
        if f.startswith("tests/"):
            continue
        p = Path(f)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if TOKEN_RE.search(text):
            token_hits.append(f)
    findings["token_pattern_hits_nontest"] = token_hits
    if token_hits:
        failures.append(f"token-shaped string in shippable file(s): {token_hits}")

    # 2. weight binaries tracked
    bin_hits = [f for f in tracked if f.endswith(WEIGHT_EXT) or f.startswith("artifacts/")]
    findings["weight_binaries_tracked"] = bin_hits
    if bin_hits:
        failures.append(f"weight binaries tracked: {bin_hits}")

    # 3. policy invariants
    from visionservex.licensing import policy as P

    bad_default = [
        r.model_id
        for r in P.iter_policies()
        if r.default_safe and r.final_policy != "commercial_safe_core"
    ]
    agpl_default = [
        r.model_id for r in P.iter_policies() if r.default_safe and "AGPL" in r.weights_license
    ]
    findings["noncore_default_safe"] = bad_default
    findings["agpl_default_safe"] = agpl_default
    if bad_default:
        failures.append(f"non-core model is default_safe: {bad_default}")
    if agpl_default:
        failures.append(f"AGPL model is default_safe: {agpl_default}")
    findings["any_can_ship_weights"] = [r.model_id for r in P.iter_policies() if r.can_ship_weights]
    if findings["any_can_ship_weights"]:
        failures.append("a policy row has can_ship_weights=True")

    # 4. no HF token in GitHub Actions
    wf_hits = []
    for wf in (
        Path(".github/workflows").glob("*.y*ml") if Path(".github/workflows").exists() else []
    ):
        text = wf.read_text(encoding="utf-8", errors="ignore")
        if TOKEN_RE.search(text) or "HF_TOKEN" in text or "HUGGINGFACE_HUB_TOKEN" in text:
            wf_hits.append(wf.name)
    findings["github_actions_hf_token_refs"] = wf_hits
    if wf_hits:
        failures.append(f"HF token referenced in workflows: {wf_hits}")

    # 5. README statement
    readme = Path("README.md").read_text(encoding="utf-8", errors="ignore")
    has_stmt = "does not redistribute" in readme.lower()
    findings["readme_nonredistribution_statement"] = has_stmt
    if not has_stmt:
        failures.append("README missing the non-redistribution statement")

    result = {"ok": not failures, "failures": failures, "findings": findings}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": result["ok"],
                "failures": failures,
                "token_hits": len(token_hits),
                "weight_binaries": len(bin_hits),
            },
            indent=2,
        )
    )
    if failures:
        print("SECURITY GUARD: FAIL")
        return 1
    print("SECURITY GUARD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
