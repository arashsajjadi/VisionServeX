# Model licenses

> This document is informational, not legal advice. Always verify upstream
> license, dataset license, and per-checkpoint license before commercial
> deployment.

VisionServeX defaults to permissively-licensed models (Apache-2.0, MIT,
BSD). Models with uncertain licensing or restricted upstream weights are
marked `license_uncertain=true` and may be `experimental`, `optional`, or
`external` in the registry.

## Authoritative table

| Model id              | Upstream                                                | License    | Uncertain | Notes                                                  |
| --------------------- | ------------------------------------------------------- | ---------- | --------- | ------------------------------------------------------ |
| dfine-small           | https://github.com/Peterande/D-FINE                     | Apache-2.0 | no        |                                                        |
| dfine-medium          | https://github.com/Peterande/D-FINE                     | Apache-2.0 | no        |                                                        |
| rfdetr-base           | https://github.com/roboflow/rf-detr                     | Apache-2.0 | no        |                                                        |
| rfdetr-large          | https://github.com/roboflow/rf-detr                     | Apache-2.0 | no        |                                                        |
| rfdetr-seg            | https://github.com/roboflow/rf-detr                     | Apache-2.0 | no        |                                                        |
| co-dino-inst          | https://github.com/Sense-X/Co-DETR                      | Apache-2.0 | yes       | Some checkpoints research-only; review.                |
| rtmpose-m             | https://github.com/open-mmlab/mmpose                    | Apache-2.0 | no        | Requires MMPose toolchain.                             |
| rtmdet-r-m            | https://github.com/open-mmlab/mmrotate                  | Apache-2.0 | yes       | OBB; verify upstream checkpoints.                      |
| rtmdet-r2-m           | https://github.com/open-mmlab/mmrotate                  | Apache-2.0 | yes       | Experimental.                                          |
| swinv2-base           | https://github.com/microsoft/Swin-Transformer           | MIT        | no        | Many HF checkpoints; verify per-checkpoint license.    |
| internimage-base      | https://github.com/OpenGVLab/InternImage                | MIT        | yes       | Custom CUDA ops; verify checkpoint license.            |
| sam2-hiera-base       | https://github.com/facebookresearch/sam2                | Apache-2.0 | no        |                                                        |
| sam2-hiera-large      | https://github.com/facebookresearch/sam2                | Apache-2.0 | no        |                                                        |
| grounding-dino-tiny   | https://github.com/IDEA-Research/GroundingDINO          | Apache-2.0 | no        |                                                        |
| grounding-dino-base   | https://github.com/IDEA-Research/GroundingDINO          | Apache-2.0 | no        |                                                        |
| grounding-dino-1.5    | https://github.com/IDEA-Research/Grounding-DINO-1.5-API | Custom     | yes       | API-gated. Disabled by default. Review upstream terms. |
| grounded-sam2         | https://github.com/IDEA-Research/Grounded-SAM-2         | Apache-2.0 | no        | Composes Grounding DINO + SAM 2.                       |
| seem-base             | https://github.com/UX-Decoder/SEEM                      | Apache-2.0 | yes       | Expert install; review checkpoints.                    |
| oneformer-coco        | https://github.com/SHI-Labs/OneFormer                   | MIT        | no        |                                                        |
| mock-*                | this repo                                               | Apache-2.0 | no        | Deterministic test fixtures.                           |

## Commercial-use considerations

For each model you deploy, confirm at least:

1. **Code license** of the upstream repository.
2. **Checkpoint license** of the specific weights you ship.
3. **Dataset license** for any training data used to produce the weights you
   use (sometimes more restrictive than the code).
4. **Trademark and attribution** requirements from the upstream project.

If anything is unclear, treat the model as `license_uncertain=true` and
either obtain explicit permission upstream, retrain on permissively-licensed
data, or pick a different model.

## VisionServeX itself

VisionServeX is licensed under Apache-2.0. The framework does not include
any AGPL-licensed code, does not depend on any AGPL package by default, and
does not import code from AGPL ecosystems for documentation, examples, or
branding.
