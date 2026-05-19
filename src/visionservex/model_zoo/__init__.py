"""Source-grounded model zoo manifest.

Every model entry in this manifest cites its official source(s) and
documents its license, install path, checkpoint location, and current
runnable status. Models that cannot be installed or run cleanly get
explicit blocker entries — they are never marked runnable.

Modules:
- sources.py: per-model URL/license metadata
- domain_zoo.py: domain → models mapping with recipes
- manifest.py: combined runtime queries
"""

from visionservex.model_zoo.domain_zoo import (
    DOMAIN_ZOO,
    DomainRecipe,
    list_domains,
    recommend_for_domain,
)

# v2.40: merge the extended block so the canonical manifest covers all 140 ledger rows.
from visionservex.model_zoo.extended_manifest_v240 import apply_v240_extension as _apply_v240
from visionservex.model_zoo.manifest import (
    SOURCE_MANIFEST,
    ModelSource,
    get_model_source,
    list_all_models,
    verify_manifest,
)

_apply_v240()

__all__ = [
    "DOMAIN_ZOO",
    "SOURCE_MANIFEST",
    "DomainRecipe",
    "ModelSource",
    "get_model_source",
    "list_all_models",
    "list_domains",
    "recommend_for_domain",
    "verify_manifest",
]
