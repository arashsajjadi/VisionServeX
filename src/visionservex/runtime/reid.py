# SPDX-License-Identifier: Apache-2.0
"""Optional ReID adapters for VisionServeX video-search.

Only one backend is wired today: Torchreid / OSNet (MIT). FastReID is
declared but routed as expert sidecar — its old environment makes a
permissive optional extra impossible.

All adapters expose::

    extract(images: list[PIL.Image | np.ndarray]) -> numpy.ndarray  # (N, dim)

Structured errors:

- TORCHREID_REQUIRED         : package not installed
- REID_CHECKPOINT_REQUIRED   : checkpoint path missing
- REID_API_UNSUPPORTED       : installed package's API has shifted
- REID_UNAVAILABLE           : adapter name is unknown
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REID_PACKAGES: dict[str, str] = {
    "osnet": "torchreid",
    "torchreid-osnet": "torchreid",
    "fastreid": "fastreid",
}

_INSTALL_HINTS: dict[str, str] = {
    "osnet": (
        "pip install torchreid  # or: "
        "pip install git+https://github.com/KaiyangZhou/deep-person-reid"
    ),
    "torchreid-osnet": (
        "pip install torchreid  # or: "
        "pip install git+https://github.com/KaiyangZhou/deep-person-reid"
    ),
    "fastreid": (
        "git clone https://github.com/JDAI-CV/fast-reid && pip install -e . "
        "(expert sidecar; old environment)"
    ),
}


class ReIDUnavailableError(Exception):
    """Raised when a ReID backend cannot be used (missing pkg or checkpoint)."""

    def __init__(self, name: str, code: str, install: str) -> None:
        super().__init__(f"{code}: {name}")
        self.name = name
        self.code = code
        self.install = install

    def to_dict(self) -> dict:
        return {"code": self.code, "reid": self.name, "install": self.install}


class ReIDAPIUnsupportedError(Exception):
    """Raised when a ReID package is installed but its API has shifted."""

    def __init__(
        self,
        name: str,
        code: str,
        installed_version: str | None,
        available_attrs: list[str],
    ) -> None:
        super().__init__(f"{code}: {name}")
        self.name = name
        self.code = code
        self.installed_version = installed_version
        self.available_attrs = available_attrs

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "reid": self.name,
            "installed_version": self.installed_version,
            "available_attrs": self.available_attrs,
        }


def _probe(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def list_reid() -> dict[str, dict[str, Any]]:
    """Return registry of known ReID backends with install status."""
    return {
        "cosine-embedding": {
            "installed": True,
            "license": "Apache-2.0",
            "core_safe": True,
            "install": None,
            "description": "Built-in cosine similarity on detector embedder (SigLIP2/DINOv2).",
        },
        "osnet": {
            "installed": _probe("torchreid"),
            "license": "MIT",
            "core_safe": True,
            "install": _INSTALL_HINTS["osnet"],
            "code": "TORCHREID_REQUIRED",
            "description": "OSNet via Torchreid (lightweight person ReID backbone).",
        },
        "fastreid": {
            "installed": _probe("fastreid"),
            "license": "Apache-2.0",
            "core_safe": False,
            "install": _INSTALL_HINTS["fastreid"],
            "code": "FASTREID_REQUIRED",
            "description": "FastReID (Apache-2.0 but old env — expert sidecar only).",
        },
    }


def build_reid_extractor(
    name: str,
    *,
    model_path: str | Path | None = None,
    device: str = "cpu",
    model_name: str = "osnet_x1_0",
) -> Any:
    """Return a callable adapter; raises a structured error if unavailable."""
    if name in ("cosine-embedding", "", None):
        return None
    pkg = _REID_PACKAGES.get(name)
    if not pkg:
        raise ReIDUnavailableError(
            name,
            "REID_UNAVAILABLE",
            f"Unknown ReID backend {name!r}. Available: {list(_REID_PACKAGES)}",
        )
    if not _probe(pkg):
        raise ReIDUnavailableError(
            name,
            "TORCHREID_REQUIRED" if pkg == "torchreid" else "FASTREID_REQUIRED",
            _INSTALL_HINTS.get(name, f"pip install {pkg}"),
        )
    if pkg == "torchreid":
        return _TorchreidOSNetAdapter(
            model_path=str(model_path) if model_path is not None else None,
            device=device,
            model_name=model_name,
        )
    # fastreid is intentionally not built here.
    raise ReIDUnavailableError(
        name,
        "FASTREID_REQUIRED",
        _INSTALL_HINTS.get(name, f"pip install {pkg}"),
    )


class _TorchreidOSNetAdapter:
    """Torchreid FeatureExtractor wrapper.

    Reference: https://kaiyangzhou.github.io/deep-person-reid/
    """

    def __init__(
        self,
        *,
        model_path: str | None,
        device: str = "cpu",
        model_name: str = "osnet_x1_0",
    ) -> None:
        # torchreid 0.2.5 ships FeatureExtractor at torchreid.reid.utils;
        # newer / legacy installs expose it at torchreid.utils. Try both.
        FeatureExtractor = None  # type: ignore[assignment]
        for path in ("torchreid.utils", "torchreid.reid.utils"):
            try:
                FeatureExtractor = __import__(path, fromlist=["FeatureExtractor"]).FeatureExtractor
                break
            except (ImportError, AttributeError):
                continue
        if FeatureExtractor is None:  # pragma: no cover - probed earlier
            raise ReIDUnavailableError(
                "osnet",
                "TORCHREID_REQUIRED",
                _INSTALL_HINTS["osnet"],
            )

        # Torchreid requires a local checkpoint. Tell the user clearly when it
        # is missing — silently downloading would surprise users on machines
        # without network access or when running tests.
        if model_path is None or not Path(model_path).exists():
            raise ReIDUnavailableError(
                "osnet",
                "REID_CHECKPOINT_REQUIRED",
                (
                    "OSNet checkpoint not found. Download from "
                    "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html "
                    "(e.g. osnet_x1_0_imagenet.pth) and pass --model-path PATH."
                ),
            )

        try:
            self._extractor = FeatureExtractor(
                model_name=model_name,
                model_path=str(model_path),
                device=device,
            )
        except (TypeError, AttributeError) as exc:
            import torchreid as _tr  # type: ignore

            raise ReIDAPIUnsupportedError(
                "osnet",
                "REID_API_UNSUPPORTED",
                getattr(_tr, "__version__", None),
                [a for a in dir(_tr) if not a.startswith("_")][:20],
            ) from exc

    def extract(self, images: list[Any]) -> Any:
        """Return an ndarray of shape (N, dim) of L2-normalized embeddings.

        ``images`` may contain strings, numpy arrays, or PIL.Image objects.
        Torchreid's FeatureExtractor accepts only strings or numpy arrays,
        so PIL.Image inputs are converted in-place.
        """
        import numpy as np

        normalized_images: list[Any] = []
        for img in images:
            if isinstance(img, str):
                normalized_images.append(img)
                continue
            if isinstance(img, np.ndarray):
                normalized_images.append(img)
                continue
            # PIL.Image or anything else with a numpy conversion path.
            normalized_images.append(np.asarray(img))

        feats = self._extractor(normalized_images)
        try:
            arr = feats.detach().cpu().numpy()
        except AttributeError:
            arr = np.asarray(feats)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


__all__ = [
    "ReIDAPIUnsupportedError",
    "ReIDUnavailableError",
    "build_reid_extractor",
    "list_reid",
]
