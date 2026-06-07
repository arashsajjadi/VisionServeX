"""CV2-Pro — professional, commercial-safe OpenCV tools (v3.1).

Weight-free, CPU-first region-proposal / segmentation-refinement / background-
subtraction / ONNX-runner tools built on OpenCV (Apache-2.0). Region-proposal
tools that need ``cv2.ximgproc`` (selective search) require the optional
``opencv-contrib-python-headless`` (extra ``cv2-pro``); everything else runs on
the base ``opencv-python-headless``. These are TOOLS, not pretrained models —
they are tracked in ``v31_cv2_pro_tool_ledger.csv``, never in the model leaderboard.
"""

from __future__ import annotations

from .tools import TOOL_LICENSE, list_tools, run_tool, tool_available

__all__ = ["TOOL_LICENSE", "list_tools", "run_tool", "tool_available"]
