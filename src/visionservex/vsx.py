"""VSX — a single, memorable, Ultralytics-style facade over VisionServeX (v3.1).

    from visionservex import VSX

    model = VSX("dfine-x"); result = model("image.jpg")
    sam   = VSX.sam("sam2.1-hiera-small"); sam.segment("image.jpg", box=[10,20,200,220])
    dino  = VSX.dino("dinov2-base"); dino.embed("image.jpg")
    pipe  = VSX.pipeline("grounding-dino-swin-t+sam2.1-hiera-small"); pipe.run("image.jpg", text="defect")
    tool  = VSX.cv2("opencv-mser-proposals"); tool.run("image.jpg")

Every handle exposes ``.status()`` (honest current state) and ``.explain()`` (a
dict describing task, license, auth, install extra, output schema, limitations,
and the exact next command). ``run``/``segment``/``embed`` route to the real
engines for commercial-safe runnable models; for gated / non-commercial / not-yet-
released targets they raise a structured error carrying the exact lawful next step.
"""

from __future__ import annotations

from typing import Any

# Compact, embedded honesty table so status()/explain() work from the installed
# wheel without the reports/ directory. Mirrors v31_sam_family_matrix /
# v31_dino_family_matrix. (state, license, auth_env, note)
_SAM_FACTS: dict[str, dict[str, str]] = {
    "_runnable": "sam-vit-base sam-vit-large sam-vit-huge sam2-hiera-tiny sam2-hiera-small "
    "sam2-hiera-base-plus sam2-hiera-large sam2.1-hiera-tiny sam2.1-hiera-small "
    "sam2.1-hiera-base-plus sam2.1-hiera-large mobilesam efficientsam medsam",
    "_legal_review": "hq-sam hq-sam2 light-hq-sam tinysam q-tinysam",
    "_auth": "sam3-base",
    "_excluded": "edge-sam edgesam",
    "_sidecar": "medsam2",
}
_DINO_FACTS = {
    "_runnable_embed": "dinov2-small dinov2-base dinov2-large dinov2-giant",
    "_runnable_detect": "grounding-dino-swin-t grounding-dino-swin-b grounding-dino-tiny "
    "grounding-dino-original-swin-t grounding-dino-original-swin-b",
    "_legal_review": "dinov3-vits16 dinov3-vitb16 dinov3-vitl16 dinov3-vit7b16",
    "_auth": "grounding-dino-1.5 grounding-dino-1.6",
    "_external_api": "grounding-dino-1.5-pro grounding-dino-1.6-pro dino-x-api",
}
_ALIAS = {"sam-vit-b": "sam-vit-base", "sam-vit-l": "sam-vit-large", "sam-vit-h": "sam-vit-huge"}


def _in(mid: str, csv: str) -> bool:
    return mid in csv.split()


class VSXError(RuntimeError):
    """Structured error carrying the exact lawful next command."""

    def __init__(self, message: str, *, state: str, next_command: str):
        super().__init__(message)
        self.state = state
        self.next_command = next_command


def _load_image(path):
    from PIL import Image

    if isinstance(path, str):
        return Image.open(path).convert("RGB")
    return path


class _Base:
    family = "model"

    def __init__(self, model_id: str):
        self.model_id = _ALIAS.get(model_id, model_id)

    def status(self) -> str:
        return self.explain()["state"]

    def explain(self) -> dict[str, Any]:  # overridden
        raise NotImplementedError


class _SAMHandle(_Base):
    family = "sam"

    def explain(self) -> dict[str, Any]:
        m = self.model_id
        if _in(m, _SAM_FACTS["_excluded"]):
            state, lic, nxt = (
                "excluded_restricted",
                "S-Lab License 1.0 (NON-COMMERCIAL)",
                "# EdgeSAM is non-commercial; external baseline only",
            )
        elif _in(m, _SAM_FACTS["_auth"]):
            state, lic, nxt = (
                "auth_required",
                "SAM License (Meta custom, HF-gated)",
                f"export HF_TOKEN=... && visionservex sam status {m}  # request HF gated access first",
            )
        elif _in(m, _SAM_FACTS["_legal_review"]):
            state, lic, nxt = (
                "legal_review_required",
                "Apache-2.0 code; HQSeg-44K/SA-1B training-data review",
                f"visionservex legal review {m}",
            )
        elif _in(m, _SAM_FACTS["_sidecar"]):
            state, lic, nxt = (
                "sidecar_required",
                "Apache-2.0",
                f"visionservex sidecar create {m} --execute",
            )
        elif _in(m, _SAM_FACTS["_runnable"]):
            state, lic, nxt = (
                "benchmark_passed",
                "Apache-2.0 (commercial-safe)",
                f"visionservex sam run {m} image.jpg --box 10,20,200,220 --out runs/{m}",
            )
        else:
            state, lic, nxt = "legal_review_required", "unknown", f"visionservex sam status {m}"
        return {
            "model_id": m,
            "family": "sam",
            "task": "promptable_segmentation",
            "state": state,
            "license": lic,
            "install_extra": "visionservex[sam2,promptable]",
            "prompt_types": ["box", "points", "mask"],
            "auth_required": state == "auth_required",
            "output_schema": {
                "masks": "list[HxW uint8]",
                "scores": "list[float]",
                "polygons": "optional",
            },
            "limitations": "video/text prompts require SAM2-video / SAM3 (gated)",
            "next_command": nxt,
            "tutorial": f"notebook/tutorials/sam_family/{m}.ipynb",
        }

    def segment(self, image, box=None, points=None, labels=None, **kw):
        info = self.explain()
        if info["state"] != "benchmark_passed":
            raise VSXError(
                f"{self.model_id} is {info['state']} — cannot run directly",
                state=info["state"],
                next_command=info["next_command"],
            )
        from visionservex.core.model import VisionModel

        with VisionModel(self.model_id) as model:
            return model.predict(_load_image(image), box=box, points=points, **kw)

    def track(self, video, frame=0, box=None, **kw):
        raise VSXError(
            f"{self.model_id}: video tracking requires the SAM2-video runtime",
            state="sidecar_required",
            next_command=f"visionservex sam video {self.model_id} {video} --box ... --out runs/video",
        )


class _DINOHandle(_Base):
    family = "dino"

    def explain(self) -> dict[str, Any]:
        m = self.model_id
        if _in(m, _DINO_FACTS["_external_api"]):
            state, lic, task, nxt = (
                "external_api_only",
                "proprietary/closed (API)",
                "open_vocab_detect",
                f"export DEEPDATASPACE_API_KEY=... && visionservex dino api {m} image.jpg --text '...'",
            )
        elif _in(m, _DINO_FACTS["_auth"]):
            state, lic, task, nxt = (
                "auth_required",
                "API/gated",
                "open_vocab_detect",
                f"export DEEPDATASPACE_API_KEY=... && visionservex dino status {m}",
            )
        elif _in(m, _DINO_FACTS["_legal_review"]):
            state, lic, task, nxt = (
                "legal_review_required",
                "DINOv3 License (Meta custom, HF-gated)",
                "embed",
                f"export HF_TOKEN=... && visionservex dino status {m}  # DINOv3 custom license, request access",
            )
        elif _in(m, _DINO_FACTS["_runnable_detect"]):
            state, lic, task, nxt = (
                "benchmark_passed",
                "Apache-2.0",
                "open_vocab_detect",
                f"visionservex dino detect {m} image.jpg --text 'defect' --out boxes.json",
            )
        elif _in(m, _DINO_FACTS["_runnable_embed"]):
            state, lic, task, nxt = (
                "benchmark_passed",
                "Apache-2.0",
                "embed",
                f"visionservex dino embed {m} image.jpg --out embedding.npy",
            )
        else:
            state, lic, task, nxt = (
                "legal_review_required",
                "unknown",
                "embed",
                f"visionservex dino status {m}",
            )
        return {
            "model_id": m,
            "family": "dino",
            "task": task,
            "state": state,
            "license": lic,
            "install_extra": "visionservex[foundation,open-vocab]",
            "auth_required": state in ("auth_required", "external_api_only"),
            "output_schema": (
                {"embedding": "np.ndarray[D]"}
                if task == "embed"
                else {"boxes": "xyxy", "scores": "list", "labels": "list"}
            ),
            "limitations": "DINOv3 is custom-licensed (gated); GroundingDINO 1.5/1.6 are API/token-gated",
            "next_command": nxt,
            "tutorial": f"notebook/tutorials/dino_family/{m}.ipynb",
        }

    def embed(self, image, **kw):
        info = self.explain()
        if info["task"] != "embed" or info["state"] != "benchmark_passed":
            raise VSXError(
                f"{self.model_id} embed not available ({info['state']})",
                state=info["state"],
                next_command=info["next_command"],
            )
        from visionservex.core.model import VisionModel

        with VisionModel(self.model_id) as model:
            return model.predict(_load_image(image))

    def detect(self, image, text: str, **kw):
        info = self.explain()
        if info["task"] != "open_vocab_detect" or info["state"] != "benchmark_passed":
            raise VSXError(
                f"{self.model_id} detect not available ({info['state']})",
                state=info["state"],
                next_command=info["next_command"],
            )
        from visionservex.core.model import VisionModel

        with VisionModel(self.model_id) as model:
            return model.predict(_load_image(image), text=text, **kw)


class _PipelineHandle:
    family = "pipeline"

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.detector, self.segmenter = pipeline_id.split("+")

    def explain(self) -> dict[str, Any]:
        det = _DINOHandle(self.detector).explain()
        seg = _SAMHandle(self.segmenter).explain()
        if (
            det["state"] in ("external_api_only", "auth_required")
            or seg["state"] == "auth_required"
        ):
            state = "auth_required"
        elif det["state"] == "legal_review_required" or seg["state"] == "legal_review_required":
            state = "legal_review_required"
        elif det["state"] == "benchmark_passed" and seg["state"] == "benchmark_passed":
            state = "pipeline_demo_ready"
        else:
            state = "blocked_on_part"
        return {
            "pipeline_id": self.pipeline_id,
            "kind": "text_to_mask",
            "state": state,
            "detector": self.detector,
            "detector_state": det["state"],
            "segmenter": self.segmenter,
            "segmenter_state": seg["state"],
            "license": f"{det['license']} + {seg['license']}",
            "next_command": f"visionservex pipeline run {self.pipeline_id} image.jpg --text 'defect' --out runs/out",
            "limitations": "gated component => the pipeline inherits auth_required",
            "tutorial": f"notebook/tutorials/pipelines/{self.pipeline_id.replace('+', '_')}.ipynb",
        }

    def status(self) -> str:
        return self.explain()["state"]

    def run(self, image, text: str, **kw):
        info = self.explain()
        if info["state"] != "pipeline_demo_ready":
            raise VSXError(
                f"pipeline {self.pipeline_id} is {info['state']}",
                state=info["state"],
                next_command=info["next_command"],
            )
        from visionservex.core.model import VisionModel

        with VisionModel(self.detector) as det:
            boxes = det.predict(_load_image(image), text=text, **kw)
        return {
            "pipeline_id": self.pipeline_id,
            "detector_result": boxes,
            "note": "run the segmenter on each box via VSX.sam(segmenter).segment(image, box=b)",
        }


class _Cv2Handle:
    family = "cv2"

    def __init__(self, tool_id: str):
        self.tool_id = tool_id

    def explain(self) -> dict[str, Any]:
        from visionservex.cv2_pro import tool_available

        ok, why = tool_available(self.tool_id)
        return {
            "tool_id": self.tool_id,
            "family": "cv2-pro",
            "state": "tool_available" if ok else "dependency_required",
            "license": "OpenCV Apache-2.0",
            "install_extra": "visionservex[cv2-pro]" if not ok else "base",
            "reason": why,
            "next_command": f"visionservex cv2-pro run {self.tool_id} image.jpg --out out.json",
        }

    def status(self) -> str:
        return self.explain()["state"]

    def run(self, image, **params):
        from visionservex.cv2_pro import run_tool

        img = _load_image(image)
        import numpy as np

        return run_tool(self.tool_id, np.asarray(img)[..., ::-1], **params)  # PIL RGB -> BGR


class VSX:
    """Top-level facade. ``VSX(model_id)`` for prediction; classmethods for families."""

    def __init__(self, model_id: str):
        from visionservex.core.model import VisionModel

        self.model_id = model_id
        self._model = VisionModel(model_id)

    def __call__(self, image, **kw):
        return self._model.predict(_load_image(image), **kw)

    @staticmethod
    def sam(model_id: str) -> _SAMHandle:
        return _SAMHandle(model_id)

    @staticmethod
    def dino(model_id: str) -> _DINOHandle:
        return _DINOHandle(model_id)

    @staticmethod
    def pipeline(pipeline_id: str) -> _PipelineHandle:
        return _PipelineHandle(pipeline_id)

    @staticmethod
    def cv2(tool_id: str) -> _Cv2Handle:
        return _Cv2Handle(tool_id)

    @staticmethod
    def interactive(model_id: str):
        from visionservex.smart_annotation import refine  # classic interactive refiners

        return refine
