# SPDX-License-Identifier: Apache-2.0
"""Anomalib version-dispatched adapter for PatchCore and other algorithms.

Handles anomalib 1.x vs 2.x API differences. Falls back to CLI if Python API
is unavailable or broken.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def detect_anomalib_version() -> str | None:
    """Return "1.x", "2.x", or None if anomalib is not installed."""
    try:
        import anomalib  # type: ignore

        raw = getattr(anomalib, "__version__", "") or ""
        major = raw.split(".")[0] if raw else ""
        if major == "1":
            return "1.x"
        if major == "2":
            return "2.x"
        # Unknown version — return best guess
        return f"{major}.x" if major else None
    except ImportError:
        return None


def get_anomalib_capabilities() -> dict[str, Any]:
    """Return a dict describing what anomalib features are available."""
    import shutil

    version = detect_anomalib_version()
    installed = version is not None

    engine_available = False
    if installed:
        try:
            importlib.import_module("anomalib.engine")
            importlib.import_module("anomalib.models")
            engine_available = True
        except ImportError:
            pass

    cli_available = shutil.which("anomalib") is not None

    return {
        "installed": installed,
        "version": version,
        "api_version": version,
        "engine_available": engine_available,
        "cli_available": cli_available,
    }


class AnomalibUnavailableError(Exception):
    """Raised when anomalib is not installed at all."""

    code = "ANOMALIB_REQUIRED"
    fix = "pip install 'visionservex[anomaly]'"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "fix": self.fix,
        }


class AnomalibUnsupportedVersionError(Exception):
    """Raised when anomalib is installed but its Python API is not compatible."""

    code = "ANOMALIB_API_UNSUPPORTED"

    def __init__(
        self,
        message: str,
        *,
        installed_version: str | None = None,
        available_modules: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.installed_version = installed_version
        self.available_modules = available_modules or []
        self.fix = (
            "Upgrade anomalib: pip install -U anomalib  "
            "or use the CLI fallback by ensuring `anomalib` is on PATH."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "installed_version": self.installed_version,
            "available_modules": self.available_modules,
            "fix": self.fix,
        }


class PatchCoreAdapter:
    """High-level adapter for PatchCore training, inference, and benchmarking.

    Attempts the anomalib Python Engine API first (1.x / 2.x) and falls back
    to the anomalib CLI subprocess when the API is unavailable.
    """

    def train(
        self,
        data_dir: str | Path,
        out_dir: str | Path,
        *,
        max_epochs: int = 1,
        image_size: int = 256,
    ) -> dict[str, Any]:
        """Train a PatchCore model on images in ``data_dir``.

        Returns a dict with at least ``status`` and ``out_dir``.
        Raises ``AnomalibUnavailableError`` if anomalib is not installed.
        Raises ``AnomalibUnsupportedVersionError`` if neither Python API nor
        CLI is functional.
        """
        data_dir = Path(data_dir)
        out_dir = Path(out_dir)

        if detect_anomalib_version() is None:
            raise AnomalibUnavailableError(
                f"anomalib is not installed. Install with: {AnomalibUnavailableError.fix}"
            )

        try:
            from anomalib.data import Folder  # type: ignore
            from anomalib.engine import Engine  # type: ignore
            from anomalib.models import Patchcore  # type: ignore

            model_instance = Patchcore()
            # anomalib's Folder uses ``root`` as the parent dir and
            # ``normal_dir`` as a path relative to that root. Splitting the
            # caller-supplied ``data_dir`` avoids the double-prefix bug.
            data_dir_resolved = data_dir.resolve()
            folder_root = str(data_dir_resolved.parent)
            normal_relative = data_dir_resolved.name
            # anomalib 2.x Folder requires a `name` field. 1.x has
            # `task="classification"` and no `name`. Try 2.x first.
            try:
                dm = Folder(
                    name="visionservex-anomaly",
                    root=folder_root,
                    normal_dir=normal_relative,
                    train_batch_size=4,
                    eval_batch_size=4,
                    num_workers=0,
                )
            except TypeError:
                try:
                    dm = Folder(
                        root=folder_root,
                        normal_dir=normal_relative,
                        task="classification",
                    )
                except TypeError:
                    dm = Folder(root=folder_root, normal_dir=normal_relative)

            # anomalib 2.x Engine drops max_epochs — Lightning Trainer kwarg.
            try:
                engine = Engine(
                    default_root_dir=str(out_dir),
                    max_epochs=max_epochs,
                )
            except TypeError:
                engine = Engine(default_root_dir=str(out_dir))
            engine.fit(model=model_instance, datamodule=dm)
            return {
                "status": "trained",
                "engine": "anomalib.Engine",
                "anomalib_version": detect_anomalib_version(),
                "out_dir": str(out_dir),
            }
        except ImportError as exc:
            caps = get_anomalib_capabilities()
            raise AnomalibUnsupportedVersionError(
                f"anomalib Engine/Patchcore API not importable: {exc}",
                installed_version=caps.get("version"),
            ) from exc
        except AttributeError as exc:
            caps = get_anomalib_capabilities()
            raise AnomalibUnsupportedVersionError(
                f"anomalib API attribute error: {exc}",
                installed_version=caps.get("version"),
            ) from exc
        except Exception as exc:
            # Try CLI fallback before giving up
            cli_result = self._try_cli_train(data_dir)
            if cli_result is not None:
                return cli_result
            return {"status": "error", "error": str(exc)[:300]}

    def _try_cli_train(self, data_dir: Path) -> dict[str, Any] | None:
        """Attempt anomalib CLI training. Returns result dict or None."""
        import shutil
        import subprocess

        if shutil.which("anomalib"):
            result = subprocess.run(
                ["anomalib", "train", "--model", "Patchcore", "--data.root", str(data_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "status": "cli_fallback",
                "returncode": result.returncode,
                "out": result.stdout[:500],
            }
        return None

    def predict(
        self,
        model_dir: str | Path,
        image_path: str | Path,
        *,
        heatmap_out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run PatchCore inference on a single image.

        Falls back to CLI if the Python Engine API is unavailable.
        Raises ``AnomalibUnavailableError`` if anomalib is not installed.
        Raises ``AnomalibUnsupportedVersionError`` if neither path works.
        """
        model_dir = Path(model_dir)
        image_path = Path(image_path)

        if detect_anomalib_version() is None:
            raise AnomalibUnavailableError(
                f"anomalib is not installed. Install with: {AnomalibUnavailableError.fix}"
            )

        try:
            from anomalib.engine import Engine  # type: ignore
            from anomalib.models import Patchcore  # type: ignore

            model_instance = Patchcore()
            engine = Engine(default_root_dir=str(model_dir))

            # anomalib 2.x: pass `data_path=` to engine.predict and load the
            # most recent .ckpt the train step wrote. 1.x accepts a path list.
            ckpt = self._latest_checkpoint(model_dir)
            try:
                result = engine.predict(
                    model=model_instance,
                    data_path=str(image_path),
                    ckpt_path=str(ckpt) if ckpt else None,
                    return_predictions=True,
                )
            except TypeError:
                result = engine.predict(
                    model=model_instance,
                    dataset=[str(image_path)],
                    ckpt_path=str(ckpt) if ckpt else None,
                )

            out: dict[str, Any] = {
                "status": "predicted",
                "engine": "anomalib.Engine",
                "anomalib_version": detect_anomalib_version(),
                "image_path": str(image_path),
                "ckpt_path": str(ckpt) if ckpt else None,
            }
            # Try to extract one scalar anomaly score per image.
            if result:
                try:
                    first = result[0]
                    if isinstance(first, list):
                        first = first[0] if first else None
                    score = getattr(first, "pred_score", None)
                    if score is None and isinstance(first, dict):
                        score = first.get("pred_score") or first.get("anomaly_score")
                    if score is not None:
                        try:
                            out["pred_score"] = float(score)
                        except (TypeError, ValueError):
                            out["pred_score_repr"] = repr(score)[:80]
                except (IndexError, AttributeError):
                    pass

            if heatmap_out and result:
                out["heatmap_out"] = str(heatmap_out)
            return out
        except ImportError as exc:
            caps = get_anomalib_capabilities()
            raise AnomalibUnsupportedVersionError(
                f"anomalib Engine API not importable: {exc}",
                installed_version=caps.get("version"),
            ) from exc
        except AttributeError as exc:
            caps = get_anomalib_capabilities()
            raise AnomalibUnsupportedVersionError(
                f"anomalib API attribute error: {exc}",
                installed_version=caps.get("version"),
            ) from exc
        except Exception as exc:
            # Try CLI fallback
            cli_result = self._try_cli_predict(image_path)
            if cli_result is not None:
                return cli_result
            return {"status": "error", "error": str(exc)[:300]}

    @staticmethod
    def _latest_checkpoint(model_dir: Path) -> Path | None:
        """Return the most recently-modified .ckpt under ``model_dir``."""
        candidates = sorted(
            Path(model_dir).rglob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _try_cli_predict(self, image_path: Path) -> dict[str, Any] | None:
        """Attempt anomalib CLI prediction. Returns result dict or None."""
        import shutil
        import subprocess

        if shutil.which("anomalib"):
            result = subprocess.run(
                ["anomalib", "predict", "--data.path", str(image_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "status": "cli_fallback",
                "returncode": result.returncode,
                "out": result.stdout[:500],
            }
        return None

    def benchmark(
        self,
        dataset_dir: str | Path,
        out_json: str | Path,
        *,
        max_images: int = 50,
    ) -> dict[str, Any]:
        """Thin benchmark wrapper: trains then predicts on up to ``max_images`` images.

        Results are written to ``out_json`` and also returned as a dict.
        """
        import json

        dataset_dir = Path(dataset_dir)
        out_json = Path(out_json)
        out_dir = out_json.parent / "patchcore_bench"

        train_result = self.train(dataset_dir, out_dir)

        images = sorted(
            p for p in dataset_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )[:max_images]

        predictions: list[dict[str, Any]] = []
        for img in images:
            pred = self.predict(out_dir, img)
            predictions.append(pred)

        payload = {
            "train": train_result,
            "n_images": len(predictions),
            "predictions": predictions,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        return payload


__all__ = [
    "AnomalibUnavailableError",
    "AnomalibUnsupportedVersionError",
    "PatchCoreAdapter",
    "detect_anomalib_version",
    "get_anomalib_capabilities",
]
