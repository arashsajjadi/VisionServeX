# Partial models final resolution (v3.20)

| Model | State | Resolution |
|---|---|---|
| `rtmdet-r2-s` | `PARTIAL_IMPLEMENTATION_BLOCKED` → hidden | OpenMMLab Docker sidecar (built+CPU-proven this sprint, `openmmlab_sidecar_final_attempt.md`). In-process engine is a stub; GPU blocked on RTX 5080 sm_120. Not visible to Anastig. |
| `rtmpose-s` | `PARTIAL_IMPLEMENTATION_BLOCKED` → hidden | same — OpenMMLab sidecar path; cached `rtmpose-*` checkpoints ready. |

Both remain `hide` in the default package (no in-process engine; sidecar is opt-in).
The earlier `maxvit-tiny-tf-224` partial was fully resolved in v3.19
(`INFERENCE_READY_LIVE` via timm TimmWrapper). **No partial model is user-visible.**
