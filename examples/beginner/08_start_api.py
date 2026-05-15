# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 08 — start the local HTTP API.

Equivalent to ``visionservex serve``.
"""

from __future__ import annotations

import uvicorn

from visionservex.config import get_settings
from visionservex.server.app import create_app


def main() -> None:
    settings = get_settings()
    app = create_app(settings)
    print(f"VisionServeX listening on http://{settings.server.host}:{settings.server.port}")
    print("Try:")
    print(
        f"  curl -F 'image=@examples/images/street.jpg' -F 'model_id=mock-detect' "
        f"http://{settings.server.host}:{settings.server.port}/detect"
    )
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
