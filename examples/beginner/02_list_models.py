# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 02 — list models in the bundled registry."""

from __future__ import annotations

from visionservex.registry import default_registry


def main() -> None:
    reg = default_registry()
    print(f"{len(reg)} models registered:\n")
    print(f"{'id':32s} {'task':22s} {'license':12s} {'status':12s} impl")
    print("-" * 90)
    for e in reg.list():
        print(f"{e.id:32s} {e.task:22s} {e.license:12s} {e.status:12s} {e.implementation_status}")


if __name__ == "__main__":
    main()
