# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Order-preserving parallel executor for medical batch segmentation.

Policy (deliberately conservative — segmentation models are large and GPU-bound):
* Output order ALWAYS matches input order, regardless of completion order.
* ``strict=True`` forces sequential execution so a stop is deterministic.
* A per-item exception is captured and isolated; it never corrupts siblings.
* The caller chooses ``workers``; GPU callers must keep it at 1 (one model per
  device — never duplicate a giant model on one GPU). CPU may use >1 only when
  the per-item ``fn`` is self-contained / thread-safe.

Import-light: only stdlib. No torch, no model code.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ItemResult:
    index: int
    item: Any
    status: str  # "ok" | "failed" | "skipped"
    value: Any = None
    error: str | None = None
    error_code: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "item": str(self.item),
            "status": self.status,
            "error": self.error,
            "error_code": self.error_code,
            **({"extra": self.extra} if self.extra else {}),
        }


def _run_one(index: int, item: Any, fn: Callable[[Any, int], Any]) -> ItemResult:
    try:
        value = fn(item, index)
    except Exception as exc:
        code = getattr(exc, "code", None)
        return ItemResult(index, item, "failed", error=str(exc), error_code=code)
    extra = value.get("extra", {}) if isinstance(value, dict) else {}
    return ItemResult(index, item, "ok", value=value, extra=extra)


def run_ordered(
    items: list[Any],
    fn: Callable[[Any, int], Any],
    *,
    workers: int = 1,
    continue_on_error: bool = True,
    strict: bool = False,
) -> list[ItemResult]:
    """Run ``fn(item, index)`` over ``items``; return results in input order.

    ``strict`` overrides ``continue_on_error`` and runs sequentially so the stop
    point is deterministic. In parallel mode every item runs to completion and
    failures are reported per-item (never raised).
    """
    n = len(items)
    results: list[ItemResult | None] = [None] * n

    # strict ⇒ sequential so "stop on first failure" is deterministic.
    if strict or workers <= 1:
        for i, it in enumerate(items):
            r = _run_one(i, it, fn)
            results[i] = r
            if r.status == "failed" and (strict or not continue_on_error):
                for j in range(i + 1, n):
                    results[j] = ItemResult(j, items[j], "skipped")
                break
        return [r for r in results if r is not None]

    # parallel, continue-on-error: order preserved by index assignment.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, i, it, fn): i for i, it in enumerate(items)}
        for fut, i in futures.items():
            results[i] = fut.result()
    return [r for r in results if r is not None]


__all__ = ["ItemResult", "run_ordered"]
