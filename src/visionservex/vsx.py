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

# LocateAnything-3B (NVIDIA License — non-commercial only, BYOT, --accept-noncommercial required).
# VisionServeX does NOT ship or mirror the weights.
_LOCATEANYTHING_FACTS: dict[str, str] = {
    "_model_ids": (
        "locate-anything-3b locate-anything-3b-v2 locate-anything-3b-grounded "
        "locate-anything-3b-coco locate-anything-3b-lvis locate-anything-3b-objects365 "
        "locate-anything-3b-open-vocab locate-anything-3b-caption "
        "locate-anything-3b-video locate-anything-3b-ft"
    ),
    "_license": "NVIDIA License (non-commercial only)",
    "_default_safe": "false",
    "_commercial_safe": "false",
    "_sidecar_install": (
        "git clone https://github.com/NVlabs/Eagle.git eagle && "
        "cd eagle/Embodied && pip install -e ."
    ),
    "_warning": (
        "WARNING: LocateAnything-3B pretrained weights are released under the NVIDIA License "
        "for non-commercial use only. Do not use this model for commercial products, paid SaaS, "
        "client work, production annotation, or redistribution unless you have written commercial "
        "permission from NVIDIA. VisionServeX does not ship or mirror the weights. "
        "Use is BYOT/user-local-cache only."
    ),
}


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

    def segment(self, image, box=None, points=None, labels=None, text=None, **kw):
        # SAM 3 / 3.1 are gated concept/text-prompt models (BYOT runtime).
        if self.model_id.startswith("sam3"):
            from visionservex.byot_runtime import sam3_segment

            res = sam3_segment(self.model_id, image, text=text, **kw)
            if isinstance(res, dict) and res.get("status") == "blocked":
                raise VSXError(res["reason"], state=res["state"],
                               next_command=res.get("next_command", ""))
            return res
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

    def track(self, frames, box=None, max_frames: int = 8, **kw):
        """SAM2 video object tracking (transformers backend).

        ``frames`` may be a list of PIL frames OR a path to a video file
        (.mp4/.avi/...), which is decoded to frames automatically.
        """
        if not (self.model_id.startswith("sam2") or "video" in self.model_id):
            raise VSXError(
                f"{self.model_id}: video tracking is a SAM2 capability",
                state="not_applicable",
                next_command="use a sam2/sam2.1 model: VSX.sam('sam2.1-hiera-small').track(frames, box=...)",
            )
        if isinstance(frames, str):
            import imageio.v3 as iio
            from PIL import Image

            arr = iio.imread(frames, plugin="pyav")
            frames = [Image.fromarray(f).convert("RGB") for f in arr[:max_frames]]
        from visionservex.sam2_runtime import track_video

        return track_video(self.model_id, frames, box=box, **kw)

    def to_onnx(self, out_path: str):
        """Export the SAM mask decoder to ONNX (commercial-safe SAM variants only)."""
        from visionservex.onnx_export import export_sam_decoder_onnx, onnx_eligible

        if self.model_id not in onnx_eligible():
            raise VSXError(
                f"{self.model_id}: not ONNX-export-eligible (commercial-safe SAM only)",
                state="not_applicable",
                next_command="eligible: mobilesam, sam-vit-b (Apache-2.0 local export)",
            )
        return export_sam_decoder_onnx(self.model_id, out_path)


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
        # DINOv3 is a gated custom-license model (BYOT runtime).
        if self.model_id.startswith("dinov3"):
            from visionservex.byot_runtime import dinov3_embed

            res = dinov3_embed(self.model_id, image, **kw)
            if isinstance(res, dict) and res.get("status") == "blocked":
                raise VSXError(res["reason"], state=res["state"],
                               next_command=res.get("next_command", ""))
            return res
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


class _LocateAnythingHandle(_Base):
    """Handle for NVIDIA LocateAnything-3B family (non-commercial, BYOT only).

    Every execution path prints the NVIDIA non-commercial warning.
    Requires ``--accept-noncommercial`` at the CLI level; ``.locate()`` requires
    ``accept_noncommercial=True`` to be passed explicitly.
    """

    family = "locate_anything"

    def explain(self) -> dict[str, Any]:
        m = self.model_id
        # v3.7: the model's OWN native license is NVIDIA non-commercial (not an
        # inherited dependency) -> excluded_restricted, same class as EdgeSAM.
        # BYOT sidecar path remains documented but it never enters commercial-safe core.
        state = "excluded_restricted"
        return {
            "model_id": m,
            "family": "locate_anything",
            "task": "open_vocab_detect_and_ground",
            "state": state,
            "license": _LOCATEANYTHING_FACTS["_license"],
            "default_safe": False,
            "commercial_safe": False,
            "install_extra": "visionservex[locateanything]",
            "auth_required": False,
            "byot": True,
            "warning": _LOCATEANYTHING_FACTS["_warning"],
            "sidecar_install": _LOCATEANYTHING_FACTS["_sidecar_install"],
            "limitations": (
                "Non-commercial NVIDIA License. BYOT/user-local-cache only. "
                "Requires --accept-noncommercial flag. Not included in default-safe core."
            ),
            "next_command": (
                f"visionservex locate-anything run {m} image.jpg "
                f"--text 'cat' --accept-noncommercial --out result.json"
            ),
            "tutorial": f"notebook/tutorials/locate_anything/{m}.ipynb",
        }

    def locate(self, image, text: str, *, accept_noncommercial: bool = False, **kw):
        """Run LocateAnything-3B grounded detection.

        ``accept_noncommercial=True`` is required; omitting it raises VSXError to
        ensure the caller has acknowledged the NVIDIA non-commercial license.
        """
        import sys

        print(_LOCATEANYTHING_FACTS["_warning"], file=sys.stderr)
        if not accept_noncommercial:
            raise VSXError(
                "LocateAnything-3B requires accept_noncommercial=True — "
                "acknowledge the NVIDIA non-commercial license before running.",
                state="excluded_restricted",
                next_command=(
                    f"VSX.locateanything('{self.model_id}').locate("
                    "image, text='...', accept_noncommercial=True)"
                ),
            )
        from visionservex.locate_anything_runtime import run_locate_anything

        return run_locate_anything(self.model_id, _load_image(image), text=text, **kw)


class _InteractiveHandle(_Base):
    """Click-based interactive segmentation (ritm/clickseg/simpleclick/focalclick + classic).

    ``VSX.interactive("ritm")(image, positive_points=[(x,y)], negative_points=[(x,y)])``.
    Named deep models are BYOT (checkpoint) with honest license states; the classic
    refiners (grabcut/watershed/...) run immediately, commercial-safe, CPU, weight-free.
    """

    family = "interactive"

    def explain(self) -> dict[str, Any]:
        from visionservex.interactive_runtime import explain as _ex
        return _ex(self.model_id)

    def __call__(self, image, positive_points=None, negative_points=None, **kw):
        from visionservex.interactive_runtime import run_interactive
        res = run_interactive(self.model_id, _load_image(image),
                              positive_points=positive_points,
                              negative_points=negative_points, **kw)
        if isinstance(res, dict) and res.get("status") == "blocked":
            raise VSXError(res["reason"], state=res["state"], next_command=res["next_command"])
        return res

    # alias matching the table's spec name
    def run(self, image, positive_points=None, negative_points=None, **kw):
        return self.__call__(image, positive_points, negative_points, **kw)


class _RFDetrSegHandle(_Base):
    """RF-DETR instance segmentation (Apache-2.0). ``VSX.rfdetr_seg("rfdetr-seg-small").segment_instances(img)``."""

    family = "rf-detr"

    def explain(self) -> dict[str, Any]:
        from visionservex.rfdetr_seg_runtime import explain as _ex
        return _ex(self.model_id)

    def segment_instances(self, image, threshold: float = 0.3, **kw):
        from visionservex.rfdetr_seg_runtime import segment_instances
        return segment_instances(self.model_id, _load_image(image), threshold=threshold, **kw)


class _HFNamespace:
    """``VSX.hf`` — Hugging Face connection (BYOT). All secrets redacted."""

    def status(self) -> dict[str, Any]:
        from visionservex import hf_auth as H

        out = {
            "logged_in": H.hf_is_logged_in(),
            "token_source": H.hf_token_source(),
            "token_redacted": H.hf_get_token(redact=True),
        }
        if out["logged_in"]:
            out.update(H.hf_whoami())
        return out

    def whoami(self, redact: bool = True) -> dict[str, Any]:
        from visionservex import hf_auth as H

        return H.hf_whoami(redact=redact)

    def is_logged_in(self) -> bool:
        from visionservex import hf_auth as H

        return H.hf_is_logged_in()

    def check_model(self, model_id: str) -> dict[str, Any]:
        from visionservex import hf_auth as H

        return H.hf_model_access_status(model_id)

    def logout(self) -> dict[str, Any]:
        from visionservex import hf_auth as H

        return H.hf_logout_local()


class _ModelHandle(_Base):
    """``VSX.model(model_id)`` — license/policy + lawful BYOT pull for any model."""

    family = "model"

    def license(self) -> dict[str, Any]:
        from visionservex.licensing import policy as P

        pol = P.get_policy(self.model_id)
        if pol is None:
            return {"model_id": self.model_id, "final_policy": "not_in_policy_table"}
        return pol.as_row()

    def explain(self) -> dict[str, Any]:
        from visionservex import hf_auth as H
        from visionservex.licensing import policy as P

        pol = P.get_policy(self.model_id)
        out: dict[str, Any] = {"model_id": P.resolve_model_id(self.model_id)}
        if pol is not None:
            out["policy"] = pol.as_row()
            out["access"] = H.hf_model_access_status(self.model_id)
            out["state"] = out["access"].get("state")
        else:
            out["state"] = "not_in_policy_table"
        return out

    def status(self) -> str:
        return self.explain().get("state", "unknown")

    def access(self) -> dict[str, Any]:
        from visionservex import hf_auth as H

        return H.hf_model_access_status(self.model_id)

    def pull(self, *, accept_upstream_license: bool = False,
             research_only: bool = False, accept_noncommercial: bool = False):
        """Lawful BYOT download into the user's HF cache. Never ships weights.

        Raises :class:`VSXError` with the exact next step when the license policy
        is not satisfied (gated without acceptance, non-commercial, enterprise…).
        """
        from visionservex import hf_auth as H
        from visionservex.licensing import policy as P

        pol = P.get_policy(self.model_id)
        canonical = P.resolve_model_id(self.model_id)
        if pol is None:
            raise VSXError(f"{self.model_id}: not in license policy",
                           state="unknown_model",
                           next_command=f"visionservex model license {self.model_id}")
        if pol.final_policy == "byot_license_required" and not accept_upstream_license:
            raise VSXError(
                f"{canonical} is gated/BYOT — pass accept_upstream_license=True after "
                f"accepting the upstream license. {pol.warning_text}",
                state="byot_license_required",
                next_command=f"accept at {pol.upstream_url}; then .pull(accept_upstream_license=True)",
            )
        try:
            H.hf_require_user_accepted_license(
                canonical, research_only=research_only,
                accept_noncommercial=accept_noncommercial)
        except H.HFLicenseError as exc:
            raise VSXError(str(exc), state=exc.state, next_command=exc.next_command) from exc
        if not pol.hf_repo:
            raise VSXError(f"{canonical}: no Hugging Face repo to pull (sidecar/manual).",
                           state="manual_download_required",
                           next_command=pol.exact_next_command)
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=pol.hf_repo, token=H.hf_get_token())


class VSX:
    """Top-level facade. ``VSX(model_id)`` for prediction; classmethods for families."""

    #: ``VSX.hf`` — Hugging Face connection namespace (status/whoami/check_model).
    hf = _HFNamespace()

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self._model = None
        if model_id is not None:
            from visionservex.core.model import VisionModel

            self._model = VisionModel(model_id)

    def __call__(self, image, **kw):
        if self._model is None:
            raise ValueError("VSX was constructed without a model_id; pass one: VSX('dfine-x')")
        return self._model.predict(_load_image(image), **kw)

    @staticmethod
    def model(model_id: str) -> _ModelHandle:
        """License/policy + lawful BYOT pull handle for any model id."""
        return _ModelHandle(model_id)

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
    def locateanything(model_id: str) -> _LocateAnythingHandle:
        """Return a handle for NVIDIA LocateAnything-3B (non-commercial, BYOT).

        Prints the NVIDIA non-commercial license warning on every call to
        ``.locate()``. Requires ``accept_noncommercial=True`` at call time.
        """
        return _LocateAnythingHandle(model_id)

    @staticmethod
    def interactive(model_id: str) -> _InteractiveHandle:
        """Click-based interactive segmentation handle.

        ``VSX.interactive("ritm")(image, positive_points=[(x,y)], negative_points=[(x,y)])``.
        Named deep models (ritm/clickseg/simpleclick/focalclick) carry honest BYOT/legal
        states; classic refiners (grabcut/watershed/...) run immediately and commercial-safe.
        """
        return _InteractiveHandle(model_id)

    @staticmethod
    def rfdetr_seg(model_id: str) -> _RFDetrSegHandle:
        """RF-DETR instance-segmentation handle (Apache-2.0, all 6 seg variants)."""
        return _RFDetrSegHandle(model_id)

    def segment_instances(self, image, threshold: float = 0.3, **kw):
        """Instance segmentation for ``VSX("rfdetr-seg-small").segment_instances(image)``.

        Routes RF-DETR-Seg model IDs to the rfdetr engine; raises for non-instance-seg models.
        """
        from visionservex.rfdetr_seg_runtime import segment_instances, variants
        mid = self.model_id
        if mid not in variants():
            raise VSXError(
                f"{mid} is not an instance-segmentation model; "
                f"use one of {variants()} or VSX.sam(...)/VSX.rfdetr_seg(...)",
                state="not_applicable",
                next_command="visionservex segment-instances image.jpg --model rfdetr-seg-small --out out/",
            )
        return segment_instances(mid, _load_image(image), threshold=threshold, **kw)

