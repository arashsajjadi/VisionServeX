# VisionServeX Surveillance Domain

Appearance-based video search and multi-object tracking. **Local-only. No face recognition. No biometric identity.**

---

## Privacy Notice

VisionServeX surveillance commands:
- Run entirely locally by default (binds to 127.0.0.1)
- Use appearance similarity only — no face recognition or biometric identification
- Do not retain embeddings or identities across sessions
- Are not designed for surveillance of individuals without consent

---

## Architecture

```
Video / Frame folder
     |
     v
[Detector] (OWLv2, Grounding DINO, ...)
     |
  detections
     |
     v
[Tracker] (simple-iou, ByteTrack, BoT-SORT, OC-SORT)
     |
  tracks
     |
     v
[ReID Embedder] (SigLIP2, OSNet, FastReID)
     |
  embeddings
     |
     v
[Index] (FAISS-backed)
     |
     v
[Text Query] --> ranked results
```

---

## Tracker Matrix

| Tracker | Status | Install | Algorithm | License |
|---------|--------|---------|-----------|---------|
| `simple-iou` | **runnable (built-in)** | None | IoU-based SORT-lite | Apache-2.0 |
| `bytetrack` | optional_extra | `pip install bytetracker` | BYTE — high/low confidence tracklets | Apache-2.0 |
| `bot-sort` | optional_extra | Clone BoT-SORT | Re-identification + camera motion compensation | Apache-2.0 |
| `ocsort` | optional_extra | `pip install ocsort` | Observation-centric SORT | MIT |

### Install tracker backends

```bash
# See what is available and get install commands:
visionservex video-search install-help
visionservex video-search install-help --tracker bytetrack
visionservex video-search install-help --tracker bot-sort

# Check tracker status:
visionservex video-search doctor --tracker bytetrack

# ByteTrack:
pip install bytetracker

# BoT-SORT:
git clone https://github.com/NirAharon/BoT-SORT && pip install -e .

# OC-SORT:
pip install ocsort
```

---

## ReID Matrix

| Backend | Status | Install | Approach | License |
|---------|--------|---------|----------|---------|
| `cosine-siglip2` | **runnable (built-in)** | None | Cosine similarity on SigLIP2 embeddings | Apache-2.0 |
| `osnet` | optional_extra | `pip install torchreid` | OSNet lightweight person backbone | MIT |
| `fastreid` | optional_extra | Clone FastReID | Strong baseline (ResNet/ViT) | Apache-2.0 |

### Install ReID backends

```bash
# See what is available:
visionservex video-search install-help --reid osnet
visionservex video-search install-help --reid fastreid

# OSNet (Torchreid):
pip install torchreid
# or:
pip install git+https://github.com/KaiyangZhou/deep-person-reid

# FastReID:
git clone https://github.com/JDAI-CV/fast-reid && pip install -e .
```

---

## Quick Start

### Index a video

```bash
pip install 'visionservex[hf]'

# Index video with default tracker and embedder:
visionservex video-search index video.mp4 \
  --out /tmp/vsx_index \
  --detector owlv2-base-patch16 \
  --embedder siglip2-base-patch16-224 \
  --prompt "person" \
  --sample-fps 1.0 \
  --tracker simple-iou

# With ByteTrack (if installed):
visionservex video-search index video.mp4 \
  --out /tmp/vsx_index \
  --tracker bytetrack
```

### Query the index

```bash
visionservex video-search query /tmp/vsx_index \
  --text "person in red jacket" \
  --top-k 10

# Save timeline HTML:
visionservex video-search query /tmp/vsx_index \
  --text "person with backpack" \
  --top-k 20 \
  --out /tmp/timeline.html

# JSON output:
visionservex video-search query /tmp/vsx_index \
  --text "person running" \
  --json
```

### Inspect an index

```bash
visionservex video-search inspect /tmp/vsx_index
visionservex video-search inspect /tmp/vsx_index --json
```

### Cleanup

```bash
visionservex video-search cleanup /tmp/vsx_index --yes
```

---

## Doctor and List Commands

```bash
# List available trackers:
visionservex video-search trackers

# List ReID backends:
visionservex video-search reid-models

# Check specific backend:
visionservex video-search doctor --tracker bytetrack
visionservex video-search doctor --reid osnet

# Get install help:
visionservex video-search install-help
visionservex video-search install-help --tracker bot-sort
```

---

## Structured Errors

If a tracker is not installed:
```json
{
  "code": "BYTETRACK_REQUIRED",
  "message": "bytetrack not installed",
  "install": "pip install bytetracker  # or: git clone https://github.com/ifzhang/ByteTrack"
}
```

---

## Performance Notes

- `simple-iou` is built-in and has zero extra dependencies. Suitable for non-overlapping, well-separated objects.
- `bytetrack` handles crowded scenes better by tracking low-confidence detections.
- `bot-sort` adds camera motion compensation — useful for moving camera footage.
- `cosine-siglip2` ReID is built-in and works well for person/vehicle re-identification when using SigLIP2 embeddings.
- For higher-accuracy ReID at scale, `osnet` (via torchreid) is the recommended upgrade path.
