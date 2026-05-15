"""Programmatically start the FastAPI server.

Equivalent to ``visionservex serve``. Useful for embedding the server inside
a larger application.
"""

from __future__ import annotations

import uvicorn

from visionservex.config import get_settings
from visionservex.server.app import create_app


def main() -> None:
    settings = get_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
