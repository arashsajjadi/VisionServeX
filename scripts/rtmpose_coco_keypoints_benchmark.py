#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2.53.0 RTMPose COCO keypoints benchmark (GT-box top-down mode).

Evaluates RTMPose models using COCO person ground-truth bounding boxes
as pose input (top-down GT-box mode). This is a valid pose-estimator
benchmark, NOT end-to-end person detection + pose.

Key distinction:
  box_source = gt-person-boxes  ← COCO GT boxes used as person crop input
  NOT end-to-end detection+pose pipeline

Metrics: OKS AP, OKS AP50, OKS AP75, OKS AR
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

COCO_KP_DIR = "/home/arash/datasets/coco_keypoints_val_mini_vsx"
COCO_KP_ANN = f"{COCO_KP_DIR}/annotations/person_keypoints_val2017_mini.json"
COCO_KP_IMG = f"{COCO_KP_DIR}/images"

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rtmpose_bench.json"
MODELS_ARG = sys.argv[2].split(",") if len(sys.argv) > 2 else None

CONFIGS_DIR = Path("/tmp/rtmpose_coco_configs")

MMPOSE_MODEL_CONFIGS = {
    "rtmpose-t": {
        "config": "/home/arash/miniconda3/envs/vsx-openmmlab-py310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/body8/rtmpose-t_8xb256-420e_body8-256x192.py",
        "checkpoint": str(
            CONFIGS_DIR
            / "rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
        ),
        "input_size": (256, 192),
        "config_source": "github:open-mmlab/mmpose/configs/body_2d_keypoint/rtmpose/coco/",
    },
    "rtmpose-s": {
        "config": str(CONFIGS_DIR / "rtmpose-s_8xb256-420e_coco-256x192.py"),
        "checkpoint": str(
            CONFIGS_DIR / "rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth"
        ),
        "input_size": (256, 192),
    },
    "rtmpose-m": {
        "config": str(CONFIGS_DIR / "rtmpose-m_8xb256-420e_coco-256x192.py"),
        "checkpoint": str(
            CONFIGS_DIR / "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
        ),
        "input_size": (256, 192),
    },
    "rtmpose-l": {
        "config": str(CONFIGS_DIR / "rtmpose-l_8xb256-420e_coco-256x192.py"),
        "checkpoint": str(
            CONFIGS_DIR / "rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"
        ),
        "input_size": (256, 192),
    },
    # 384x288 configs not available in mim registry for this mmpose version.
    # Marked as RTMPOSE_CONFIG_NOT_IN_MIM_REGISTRY.
    "rtmpose-m-384x288": {
        "config": str(CONFIGS_DIR / "rtmpose-m_8xb256-420e_aic-coco-384x288.py"),
        "checkpoint": str(
            CONFIGS_DIR / "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth"
        ),
        "input_size": (384, 288),
        "config_source": "github:open-mmlab/mmpose/configs/body_2d_keypoint/rtmpose/coco/",
    },
    "rtmpose-l-384x288": {
        "config": str(CONFIGS_DIR / "rtmpose-l_8xb256-420e_aic-coco-384x288.py"),
        "checkpoint": str(
            CONFIGS_DIR / "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth"
        ),
        "input_size": (384, 288),
        "config_source": "github:open-mmlab/mmpose/configs/body_2d_keypoint/rtmpose/coco/",
    },
}

DEFAULT_MODELS = list(MMPOSE_MODEL_CONFIGS.keys())


def _load_dataset():
    from pycocotools.coco import COCO

    coco_gt = COCO(COCO_KP_ANN)
    img_ids = list(coco_gt.imgs.keys())
    return coco_gt, img_ids


def _run_model(model_id: str, coco_gt, img_ids: list) -> dict:
    result = {
        "model_id": model_id,
        "status": "failed",
        "code": "",
        "box_source": "gt-person-boxes",
        "oks_ap": None,
        "oks_ap50": None,
        "oks_ap75": None,
        "oks_ar": None,
        "latency_ms_p50": None,
        "fps": None,
        "n_images": len(img_ids),
        "n_person_instances": 0,
        "error": None,
    }
    if model_id not in MMPOSE_MODEL_CONFIGS:
        result["code"] = "UNKNOWN_MODEL_ID"
        return result
    cfg = MMPOSE_MODEL_CONFIGS[model_id]

    if cfg.get("unavailable"):
        result["code"] = cfg["unavailable"]
        result["error"] = (
            f"RTMPose config for {model_id} ({cfg['input_size']}) not available in "
            f"mim registry for mmpose 1.3.x. Config would need manual download from "
            f"OpenMMLab GitHub. Run: mim download mmpose --config {model_id.replace('-', '_').replace('rtmpose_', 'rtmpose-')} --dest /tmp/rtmpose_coco_configs"
        )
        print(f"  [{model_id}] NOT AVAILABLE: {cfg['unavailable']}")
        return result

    try:
        from mmpose.apis import inference_topdown, init_model
    except ImportError as e:
        result["code"] = "MMPOSE_IMPORT_FAILED"
        result["error"] = str(e)
        return result

    import torch

    device = "cpu"  # mmcv ABI incompatible with cu130; GPU latency measured separately
    result["device"] = device

    try:
        pose_model = init_model(
            cfg["config"],
            cfg["checkpoint"],
            device=device,
            cfg_options={"model.test_cfg.flip_test": False},
        )
        print(f"  [{model_id}] loaded on {device}")
    except Exception as e:
        err = str(e)[:300]
        if "checkpoint" in err.lower() or "download" in err.lower():
            result["code"] = "CHECKPOINT_REQUIRED"
        else:
            result["code"] = "MODEL_LOAD_FAILED"
        result["error"] = err
        print(f"  [{model_id}] LOAD FAILED: {err[:80]}")
        return result

    preds_coco = []
    latencies = []
    n_instances = 0

    img_dir = Path(COCO_KP_IMG)
    for img_id in img_ids:
        img_info = coco_gt.imgs[img_id]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        ann_ids = coco_gt.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco_gt.loadAnns(ann_ids)
        # GT bounding boxes for top-down inference
        bboxes = [a["bbox"] for a in anns if a.get("num_keypoints", 0) > 0]
        if not bboxes:
            continue

        # Convert COCO xywh to xyxy for MMPose
        bboxes_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in bboxes]

        try:
            t0 = time.perf_counter()
            results = inference_topdown(pose_model, str(img_path), bboxes_xyxy, bbox_format="xyxy")
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            n_instances += len(results)
            for det_result in results:
                pred_instances = det_result.pred_instances
                if pred_instances is None:
                    continue
                kps = pred_instances.keypoints
                scores = pred_instances.keypoint_scores
                if kps is None or scores is None:
                    continue
                # mmpose 1.3.x returns numpy arrays (not tensors)
                import numpy as _np

                kps_np = kps.cpu().numpy() if hasattr(kps, "cpu") else _np.asarray(kps)
                scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else _np.asarray(scores)
                for i in range(kps_np.shape[0]):
                    kp_flat = []
                    for j in range(17):
                        kp_flat.extend(
                            [
                                float(kps_np[i, j, 0]),
                                float(kps_np[i, j, 1]),
                                float(scores_np[i, j]),
                            ]
                        )
                    preds_coco.append(
                        {
                            "image_id": img_id,
                            "category_id": 1,
                            "keypoints": kp_flat,
                            "score": float(scores_np[i].mean()),
                        }
                    )
        except Exception:
            continue

    result["n_person_instances"] = n_instances
    if not preds_coco:
        result["code"] = "NO_PREDICTIONS"
        result["error"] = "No valid keypoint predictions"
        return result

    try:
        from pycocotools.cocoeval import COCOeval

        coco_dt = coco_gt.loadRes(preds_coco)
        ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        lats = sorted(latencies)
        p50 = lats[len(lats) // 2] if lats else None
        fps = 1000.0 / (sum(latencies) / len(latencies)) if latencies else None
        result.update(
            {
                "status": "ok",
                "code": "OK",
                "oks_ap": round(float(ev.stats[0]), 4),
                "oks_ap50": round(float(ev.stats[1]), 4),
                "oks_ap75": round(float(ev.stats[2]), 4),
                "oks_ar": round(float(ev.stats[5]), 4),
                "latency_ms_p50": round(p50, 1) if p50 else None,
                "fps": round(fps, 1) if fps else None,
            }
        )
        print(
            f"  [{model_id}] OKS AP={ev.stats[0]:.4f} "
            f"AP50={ev.stats[1]:.4f} AP75={ev.stats[2]:.4f} "
            f"AR={ev.stats[5]:.4f} lat={p50:.0f}ms"
        )
    except Exception as e:
        result["code"] = "EVAL_FAILED"
        result["error"] = str(e)[:300]

    try:
        del pose_model
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main():
    models_to_run = MODELS_ARG or DEFAULT_MODELS
    print(f"RTMPose benchmark: {len(models_to_run)} models")
    print(f"Dataset: {COCO_KP_ANN}")
    print("Box source: gt-person-boxes (top-down GT-box evaluation)")
    coco_gt, img_ids = _load_dataset()
    print(f"Images with keypoint annotations: {len(img_ids)}")
    results = []
    for i, mid in enumerate(models_to_run):
        print(f"\n[{i + 1}/{len(models_to_run)}] {mid}")
        r = _run_model(mid, coco_gt, img_ids)
        results.append(r)
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "benchmark_type": "rtmpose_coco_keypoints_gt_box_topdown",
                "box_source": "gt-person-boxes",
                "evaluation_note": "GT person boxes used as pose input. This evaluates the pose estimator only, not end-to-end detection+pose.",
                "dataset": "coco-keypoints:" + COCO_KP_DIR,
                "n_images": len(img_ids),
                "models": results,
            },
            f,
            indent=2,
        )
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nSummary: {len(ok)}/{len(results)} benchmark_passed")
    if ok:
        print(f"Best OKS AP: {max(r['oks_ap'] for r in ok):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
