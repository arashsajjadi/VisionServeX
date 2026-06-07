"""Smart-annotation toolkit (V3).

Commercial-safe, CPU-only *classic* (no-weight) interactive segmentation/refinement
tools that turn a coarse user prompt (box / points / scribble / polygon / mask hint)
into a clean instance mask. These are NOT pretrained models — there are no model
weights, so the only legal surface is the dependency license (all permissive:
OpenCV Apache-2.0, scikit-image / scikit-learn / scipy / numpy BSD-3). They are
tracked in ``smart_tool_coverage_ledger.csv``, separate from the pretrained model
leaderboard, per V3 gate V3-13.

Public API::

    from visionservex.smart_annotation import Prompt, refine, list_methods

    result = refine(image, Prompt(box=(10, 20, 200, 220)), method="classic-grabcut")
    result.mask        # H x W uint8 {0,1}
    result.to_contract_dict()
"""

from __future__ import annotations

from .classic import METHOD_LICENSE, list_methods, refine
from .contracts import OUTPUT_CONTRACT_KEYS, Prompt, RefineResult

__all__ = [
    "METHOD_LICENSE",
    "OUTPUT_CONTRACT_KEYS",
    "Prompt",
    "RefineResult",
    "list_methods",
    "refine",
]
