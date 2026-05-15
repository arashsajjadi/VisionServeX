# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Python client for the VisionServeX local HTTP gateway.

Usage::

    from visionservex import Client

    client = Client("http://127.0.0.1:8080")
    result = client.predict("dfine-n", "image.jpg")
    result = client.detect("rfdetr-nano", "image.jpg")
    result = client.segment("sam2-hiera-tiny", "image.jpg", box=[10, 20, 200, 300])
    result = client.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
    result = client.classify("swinv2-tiny", "image.jpg")

All methods accept:
- A file path (str/Path) → multipart upload
- PIL Image → encoded internally
- bytes → multipart upload
- base64 string → JSON body (via ``image_b64``)
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

_RETRY_STATUS = {503}


class GatewayError(RuntimeError):
    """Raised when the gateway returns an error envelope."""

    def __init__(
        self, code: str, message: str, hint: str = "", details: dict | None = None
    ) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.hint = hint
        self.details = details or {}


class ClientResult:
    """Thin wrapper around the server prediction response."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def model_id(self) -> str:
        return self._data.get("model_id", "")

    @property
    def task(self) -> str:
        return self._data.get("task", "")

    @property
    def device(self) -> str:
        return self._data.get("device", "")

    @property
    def precision(self) -> str:
        return self._data.get("precision", "fp32")

    @property
    def backend(self) -> str:
        return self._data.get("backend", "")

    @property
    def latency_ms(self) -> float:
        return float(self._data.get("latency_ms", 0.0))

    @property
    def results(self) -> list[dict[str, Any]]:
        return self._data.get("results", [])

    @property
    def warnings(self) -> list[str]:
        return self._data.get("warnings", [])

    @property
    def request_id(self) -> str:
        return self._data.get("request_id", "")

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return (
            f"<ClientResult model={self.model_id!r} task={self.task!r} "
            f"device={self.device!r} latency={self.latency_ms:.1f}ms "
            f"n_results={len(self.results)}>"
        )


class Client:
    """Synchronous HTTP client for VisionServeX gateway.

    Args:
        base_url: Gateway URL, e.g. ``"http://127.0.0.1:8080"``.
        api_key:  Bearer token when auth is enabled.
        timeout:  Per-request timeout in seconds.
        max_retries: Number of retries for 503 SERVER_BUSY responses.
        retry_delay: Initial retry delay in seconds (doubles each retry).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ------ low-level ------

    def _headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {}
        if self._api_key:
            hdrs["Authorization"] = f"Bearer {self._api_key}"
        return hdrs

    def _get(self, path: str) -> dict[str, Any]:
        import httpx

        r = httpx.get(f"{self.base_url}{path}", headers=self._headers(), timeout=self._timeout)
        return self._raise_or_json(r)

    def _post_multipart(
        self,
        path: str,
        image_bytes: bytes,
        filename: str,
        extra_fields: dict[str, str],
    ) -> dict[str, Any]:
        import httpx

        attempt = 0
        delay = self._retry_delay
        while True:
            r = httpx.post(
                f"{self.base_url}{path}",
                files={"image": (filename, io.BytesIO(image_bytes), "image/jpeg")},
                data=extra_fields,
                headers=self._headers(),
                timeout=self._timeout,
            )
            if r.status_code in _RETRY_STATUS and attempt < self._max_retries:
                attempt += 1
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            return self._raise_or_json(r)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        attempt = 0
        delay = self._retry_delay
        while True:
            r = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            if r.status_code in _RETRY_STATUS and attempt < self._max_retries:
                attempt += 1
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            return self._raise_or_json(r)

    @staticmethod
    def _raise_or_json(response: Any) -> dict[str, Any]:
        data = response.json()
        if response.status_code >= 400:
            err = data.get("error", {})
            raise GatewayError(
                code=err.get("code", f"HTTP_{response.status_code}"),
                message=err.get("message", response.text[:200]),
                hint=err.get("hint", ""),
                details=err.get("details", {}),
            )
        return data

    @staticmethod
    def _prepare_image(image: Any) -> tuple[bytes, str]:
        """Return (bytes, filename) for any supported image input."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            return p.read_bytes(), p.name
        if isinstance(image, bytes):
            return image, "image.jpg"
        # PIL Image
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image.jpg"

    # ------ metadata ------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def models(self) -> list[dict[str, Any]]:
        return self._get("/models").get("models", [])

    def model_info(self, model_id: str) -> dict[str, Any]:
        return self._get(f"/models/{model_id}")

    def devices(self) -> list[dict[str, Any]]:
        return self._get("/devices").get("devices", [])

    def gateway_status(self) -> dict[str, Any]:
        return self._get("/gateway/status")

    def metrics(self) -> dict[str, Any]:
        return self._get("/metrics")

    # ------ lifecycle ------

    def pull(self, model_id: str, *, wait: bool = True) -> dict[str, Any]:
        """Trigger a weight download on the server side."""
        return self._post_json(f"/models/{model_id}/pull", {})

    def load(self, model_id: str) -> dict[str, Any]:
        return self._post_json(f"/models/{model_id}/load", {})

    def unload(self, model_id: str) -> dict[str, Any]:
        return self._post_json(f"/models/{model_id}/unload", {})

    def warmup(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        return self._post_json("/gateway/warmup", {"model_ids": model_ids or []})

    # ------ inference ------

    def predict(
        self,
        model_id: str,
        image: Any,
        *,
        prompts: list[str] | None = None,
        wait_for_download: bool = True,
    ) -> ClientResult:
        """Generic predict — routes to the right task automatically."""
        img_bytes, fname = self._prepare_image(image)
        fields: dict[str, str] = {"model_id": model_id}
        if prompts:
            fields["prompts"] = ",".join(prompts)
        data = self._post_multipart(
            f"/predict?wait_for_download={'true' if wait_for_download else 'false'}",
            img_bytes,
            fname,
            fields,
        )
        return ClientResult(data)

    def detect(self, model_id: str, image: Any) -> ClientResult:
        img_bytes, fname = self._prepare_image(image)
        data = self._post_multipart("/detect", img_bytes, fname, {"model_id": model_id})
        return ClientResult(data)

    def classify(self, model_id: str, image: Any) -> ClientResult:
        img_bytes, fname = self._prepare_image(image)
        data = self._post_multipart("/classify", img_bytes, fname, {"model_id": model_id})
        return ClientResult(data)

    def segment(
        self,
        model_id: str,
        image: Any,
        *,
        box: list[float] | None = None,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
    ) -> ClientResult:
        """Foundation segmentation with optional box or point prompts."""
        img_bytes, fname = self._prepare_image(image)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        payload: dict[str, Any] = {"model_id": model_id, "image_b64": b64, "prompts": []}
        if box:
            payload["options"] = {"boxes": [box]}
        if points:
            payload["options"] = {"points": points, "point_labels": point_labels or []}
        data = self._post_json("/segment/b64", payload)
        if "error" in data:
            # Fall back to multipart for standard segment endpoint
            data = self._post_multipart("/segment", img_bytes, fname, {"model_id": model_id})
        return ClientResult(data)

    def open_vocab_detect(
        self,
        model_id: str,
        image: Any,
        prompts: list[str],
    ) -> ClientResult:
        img_bytes, _ = self._prepare_image(image)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data = self._post_json(
            "/open-vocab/detect",
            {"model_id": model_id, "image_b64": b64, "prompts": prompts},
        )
        return ClientResult(data)

    def grounded_segment(
        self,
        model_id: str,
        image: Any,
        prompt: str,
    ) -> ClientResult:
        img_bytes, _ = self._prepare_image(image)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data = self._post_json(
            "/grounded-segment",
            {
                "model_id": model_id,
                "image_b64": b64,
                "prompts": [p.strip() for p in prompt.split(",") if p.strip()],
            },
        )
        return ClientResult(data)

    def pose(self, model_id: str, image: Any) -> ClientResult:
        img_bytes, fname = self._prepare_image(image)
        data = self._post_multipart("/pose", img_bytes, fname, {"model_id": model_id})
        return ClientResult(data)

    def obb(self, model_id: str, image: Any) -> ClientResult:
        """Oriented bounding box detection."""
        img_bytes, fname = self._prepare_image(image)
        data = self._post_multipart("/obb", img_bytes, fname, {"model_id": model_id})
        return ClientResult(data)

    def batch_detect(self, model_id: str, images: list[Any]) -> list[ClientResult]:
        return [self.detect(model_id, img) for img in images]

    # ------ jobs ------

    def job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def job_events(self, job_id: str, *, poll_interval_s: float = 0.5):  # yields dict
        """Poll job events until terminal state. Yields each status snapshot."""
        terminal = {"completed", "failed", "cancelled"}
        while True:
            data = self.job_status(job_id)
            yield data
            if data.get("status") in terminal:
                break
            time.sleep(poll_interval_s)

    def poll_job(
        self, job_id: str, *, timeout_s: float = 600.0, interval_s: float = 1.0
    ) -> dict[str, Any]:
        """Poll until job reaches a terminal state."""
        deadline = time.time() + timeout_s
        terminal = {"completed", "failed", "cancelled"}
        while time.time() < deadline:
            data = self.job_status(job_id)
            if data.get("status") in terminal:
                return data
            time.sleep(interval_s)
        raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        import httpx

        r = httpx.delete(
            f"{self.base_url}/jobs/{job_id}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        return self._raise_or_json(r)


# ---------------------------------------------------------------------------
# AsyncClient
# ---------------------------------------------------------------------------


class AsyncClient:
    """Asynchronous HTTP client for VisionServeX gateway.

    Requires ``httpx[asyncio]`` (already a dependency).

    Example::

        import asyncio
        from visionservex import AsyncClient

        async def main():
            client = AsyncClient("http://127.0.0.1:8080")
            result = await client.detect("dfine-n", "image.jpg")
            print(result)

        asyncio.run(main())
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {}
        if self._api_key:
            hdrs["Authorization"] = f"Bearer {self._api_key}"
        return hdrs

    @staticmethod
    def _raise_or_json(response: Any) -> dict[str, Any]:
        return Client._raise_or_json(response)

    @staticmethod
    def _prepare_image(image: Any) -> tuple[bytes, str]:
        return Client._prepare_image(image)

    async def _get(self, path: str) -> dict[str, Any]:

        import httpx

        async with httpx.AsyncClient(headers=self._headers(), timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}{path}")
            return self._raise_or_json(r)

    async def _post_multipart(
        self,
        path: str,
        image_bytes: bytes,
        filename: str,
        extra_fields: dict[str, str],
    ) -> dict[str, Any]:
        import asyncio

        import httpx

        attempt = 0
        delay = self._retry_delay
        async with httpx.AsyncClient(headers=self._headers(), timeout=self._timeout) as c:
            while True:
                r = await c.post(
                    f"{self.base_url}{path}",
                    files={"image": (filename, io.BytesIO(image_bytes), "image/jpeg")},
                    data=extra_fields,
                )
                if r.status_code in _RETRY_STATUS and attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                return self._raise_or_json(r)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        import httpx

        attempt = 0
        delay = self._retry_delay
        async with httpx.AsyncClient(headers=self._headers(), timeout=self._timeout) as c:
            while True:
                r = await c.post(f"{self.base_url}{path}", json=payload)
                if r.status_code in _RETRY_STATUS and attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                return self._raise_or_json(r)

    async def health(self) -> dict[str, Any]:
        return await self._get("/health")

    async def models(self) -> list[dict[str, Any]]:
        return (await self._get("/models")).get("models", [])

    async def detect(self, model_id: str, image: Any) -> ClientResult:
        img_bytes, fname = self._prepare_image(image)
        data = await self._post_multipart("/detect", img_bytes, fname, {"model_id": model_id})
        return ClientResult(data)

    async def classify(self, model_id: str, image: Any, *, top_k: int = 5) -> ClientResult:
        img_bytes, fname = self._prepare_image(image)
        data = await self._post_multipart(
            "/classify", img_bytes, fname, {"model_id": model_id, "top_k": str(top_k)}
        )
        return ClientResult(data)

    async def segment(
        self,
        model_id: str,
        image: Any,
        *,
        box: list[float] | None = None,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
    ) -> ClientResult:
        img_bytes, _ = self._prepare_image(image)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        payload: dict[str, Any] = {"model_id": model_id, "image_b64": b64, "prompts": []}
        if box:
            payload["options"] = {"boxes": [box]}
        data = await self._post_json("/segment/b64", payload)
        return ClientResult(data)

    async def grounded_segment(self, model_id: str, image: Any, prompt: str) -> ClientResult:
        img_bytes, _ = self._prepare_image(image)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data = await self._post_json(
            "/grounded-segment",
            {
                "model_id": model_id,
                "image_b64": b64,
                "prompts": [p.strip() for p in prompt.split(",") if p.strip()],
            },
        )
        return ClientResult(data)

    async def predict(self, model_id: str, image: Any, **kwargs: Any) -> ClientResult:
        img_bytes, fname = self._prepare_image(image)
        fields: dict[str, str] = {"model_id": model_id}
        data = await self._post_multipart("/predict", img_bytes, fname, fields)
        return ClientResult(data)

    async def batch_detect(self, model_id: str, images: list[Any]) -> list[ClientResult]:
        import asyncio

        tasks = [self.detect(model_id, img) for img in images]
        return list(await asyncio.gather(*tasks))


__all__ = ["AsyncClient", "Client", "ClientResult", "GatewayError"]
