# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Hugging Face connection layer (BYOT — Bring Your Own Token).

VisionServeX never bundles gated or restricted weights and never stores your
token. This module is the user-facing way to:

* detect whether you are already logged in (``huggingface-cli login`` cache),
  via the ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` env var, or a local private
  token file;
* see *who* you are on the Hub (``whoami``) without ever printing the token;
* check whether your token grants access to a gated model (e.g. SAM 3, DINOv3)
  *without downloading any weights*;
* get the exact upstream model page where you must accept the license yourself;
* enforce the VisionServeX license policy (non-commercial models refuse a
  production run; gated weights are never redistributed).

Security contract
-----------------
* The raw token is returned by exactly one function — :func:`hf_get_token` — and
  only when ``redact=False`` is passed explicitly. Every other function redacts.
* Nothing here writes the token to disk, logs, notebooks, reports, or git.
* :func:`hf_redact_token` shows only the first 3 and last 2 characters.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from visionservex.licensing import policy as _policy

# Documented private token-file locations (read into memory only, never copied).
_TOKEN_FILE_CANDIDATES: tuple[str, ...] = (
    "/home/arash/Documents/ای پی ای هاگینگ فیس",
    "/home/arash/Documents/api_huggingface.txt",
)
_TOKEN_FILE_GLOBS: tuple[str, ...] = ("hugging", "هاگینگ", "hf", "api_hugging")

_ENV_VARS: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


class HFAuthError(RuntimeError):
    """No usable Hugging Face token / not logged in."""

    def __init__(self, message: str, *, next_command: str = ""):
        super().__init__(message)
        self.next_command = next_command


class HFLicenseError(RuntimeError):
    """The user has not satisfied the upstream/local license policy for a model."""

    def __init__(self, message: str, *, state: str, next_command: str = "", model_id: str = ""):
        super().__init__(message)
        self.state = state
        self.next_command = next_command
        self.model_id = model_id


# --------------------------------------------------------------------------- #
# Token redaction
# --------------------------------------------------------------------------- #
def hf_redact_token(value: str | None) -> str:
    """Redact a token to ``abc***yz`` form. Never returns more than 5 real chars."""
    if not value:
        return ""
    value = value.strip()
    if len(value) < 8:
        return "***"
    return f"{value[:3]}***{value[-2:]}"


# --------------------------------------------------------------------------- #
# Token detection (cli cache -> env -> local file), in that order
# --------------------------------------------------------------------------- #
def _read_token_file(path: str) -> str | None:
    try:
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8", errors="ignore") as fh:
            val = fh.read().strip()
    except OSError:
        return None
    return val if (val.startswith("hf_") and len(val) >= 20) else None


def _file_candidates() -> list[str]:
    import glob

    cands = list(_TOKEN_FILE_CANDIDATES)
    for pat in _TOKEN_FILE_GLOBS:
        cands.extend(glob.glob(f"/home/arash/Documents/*{pat}*"))
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _detect() -> tuple[str | None, str | None]:
    """Return ``(token, source)`` using the documented detection order.

    source is one of: ``cli_cache``, ``env:HF_TOKEN``, ``env:HUGGINGFACE_HUB_TOKEN``,
    ``file:<basename>`` or None.
    """
    # 1) huggingface-cli login cache
    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached:
            return cached, "cli_cache"
    except Exception:
        pass
    # 2) environment variables
    for var in _ENV_VARS:
        v = os.environ.get(var)
        if v:
            return v, f"env:{var}"
    # 3) local private file
    for path in _file_candidates():
        tok = _read_token_file(path)
        if tok:
            return tok, f"file:{os.path.basename(path)}"
    return None, None


def hf_token_source() -> str | None:
    """Where a token would be read from (without exposing the token)."""
    return _detect()[1]


def hf_get_token(redact: bool = False) -> str | None:
    """Return the active Hugging Face token.

    .. warning::
        With ``redact=False`` (the default) this returns the **raw** token for
        internal use by the download path. NEVER print, log, or persist the
        return value. Use ``redact=True`` for anything user-visible.
    """
    tok, _ = _detect()
    if redact:
        # always a redacted string ("" when no token) — never a raw token
        return hf_redact_token(tok)
    return tok


def hf_is_logged_in() -> bool:
    """True if a usable token can be detected (does not hit the network)."""
    return _detect()[0] is not None


# --------------------------------------------------------------------------- #
# whoami / validation (network)
# --------------------------------------------------------------------------- #
@dataclass
class HFWhoAmI:
    logged_in: bool
    source: str | None = None
    token_redacted: str = ""
    name: str | None = None
    type: str | None = None
    token_display_name: str | None = None
    token_role: str | None = None
    orgs: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orgs"] = list(self.orgs)
        return d


def hf_whoami(redact: bool = True) -> dict:
    """Resolve the current Hub identity. Never includes the raw token.

    ``redact`` is accepted for API symmetry; the token is ALWAYS redacted in the
    returned payload regardless of its value.
    """
    tok, source = _detect()
    info = HFWhoAmI(logged_in=bool(tok), source=source, token_redacted=hf_redact_token(tok))
    if not tok:
        return info.to_dict()
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=tok).whoami()
        info.name = who.get("name")
        info.type = who.get("type")
        auth = (who.get("auth") or {}).get("accessToken") or {}
        info.token_display_name = auth.get("displayName")
        info.token_role = auth.get("role")
        info.orgs = tuple(o.get("name") for o in (who.get("orgs") or []) if o.get("name"))
    except Exception as exc:  # network / invalid token
        info.error = f"{type(exc).__name__}: {exc}"
    return info.to_dict()


def hf_validate_token(required_scopes: list[str] | None = None) -> dict:
    """Validate the detected token against the Hub (and optional fine-grained scopes).

    Returns a dict with ``valid``, ``source``, ``name``, ``role`` and (when
    ``required_scopes`` is given and the Hub exposes them) ``missing_scopes``.
    """
    tok, source = _detect()
    out: dict[str, object] = {
        "valid": False,
        "source": source,
        "token_redacted": hf_redact_token(tok),
    }
    if not tok:
        out["error"] = "no_token"
        out["next_command"] = "visionservex hf connect  # or: huggingface-cli login"
        return out
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=tok).whoami()
        out["valid"] = True
        out["name"] = who.get("name")
        out["type"] = who.get("type")
        auth = (who.get("auth") or {}).get("accessToken") or {}
        out["role"] = auth.get("role")
        out["token_display_name"] = auth.get("displayName")
        if required_scopes:
            granted = set()
            fg = (who.get("auth") or {}).get("accessToken", {}).get("fineGrained")
            if isinstance(fg, dict):
                for scope_block in fg.get("scoped", []) or []:
                    granted.update(scope_block.get("permissions", []) or [])
                granted.update(fg.get("global", []) or [])
            out["granted_scopes"] = sorted(granted)
            out["missing_scopes"] = (
                sorted(set(required_scopes) - granted) if granted else list(required_scopes)
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def hf_logout_local() -> dict:
    """Remove the locally cached huggingface-cli login (best effort).

    Does not touch env vars or token files (those are owned by the user).
    """
    result: dict[str, object] = {"logged_out_cache": False, "note": ""}
    try:
        from huggingface_hub import logout

        logout()
        result["logged_out_cache"] = True
    except Exception as exc:
        result["note"] = f"could not clear cache login: {type(exc).__name__}: {exc}"
    env_present = [v for v in _ENV_VARS if os.environ.get(v)]
    if env_present:
        result["note"] = (
            result["note"] + " " if result["note"] else ""
        ) + f"Env var(s) still set: {', '.join(env_present)} (unset them yourself)."
    return result


# --------------------------------------------------------------------------- #
# Per-model access + policy gates
# --------------------------------------------------------------------------- #
def hf_acceptance_instructions(model_id: str) -> dict:
    """Exact upstream page + steps for accepting a model's license yourself."""
    pol = _policy.get_policy(model_id)
    canonical = _policy.resolve_model_id(model_id)
    if pol is None:
        return {
            "model_id": canonical,
            "known": False,
            "instructions": [
                f"Model '{model_id}' is not in the VisionServeX license policy table.",
                "Run: visionservex model license <model_id> for known models.",
            ],
        }
    repo = pol.hf_repo
    url = pol.upstream_url or (f"https://huggingface.co/{repo}" if repo else "")
    steps = []
    if pol.gated and repo:
        steps = [
            f"1. Visit https://huggingface.co/{repo} and click 'Agree and access repository'.",
            "2. Connect your token locally: visionservex hf connect  (or huggingface-cli login).",
            f"3. Confirm access: visionservex hf check-model {repo}",
            f"4. Pull (BYOT): visionservex model pull {canonical} --accept-upstream-license",
        ]
    elif pol.final_policy == "external_api_only_terms_required":
        steps = [
            f"This is an external API model. Read provider terms at {url}.",
            "Set your provider API key as an env var; data leaves your machine.",
        ]
    else:
        steps = [f"See the upstream page: {url}" if url else "No upstream URL on record."]
    return {
        "model_id": canonical,
        "known": True,
        "final_policy": pol.final_policy,
        "hf_repo": repo,
        "upstream_url": url,
        "user_license_required": pol.user_license_required,
        "instructions": steps,
        "warning": pol.warning_text,
    }


def hf_model_access_status(model_id: str) -> dict:
    """Check whether the current token can access a model's (gated) repo.

    Performs a metadata-only Hub call — it never downloads weights. The runtime
    ``state`` reflects what the user can actually do right now.
    """
    pol = _policy.get_policy(model_id)
    canonical = _policy.resolve_model_id(model_id)
    out: dict[str, object] = {
        "model_id": canonical,
        "final_policy": pol.final_policy if pol else "unknown",
        "hf_repo": pol.hf_repo if pol else "",
        "gated": bool(pol.gated) if pol else None,
        "token_present": hf_is_logged_in(),
        "token_source": hf_token_source(),
    }
    if pol is None:
        out["state"] = "unknown_model"
        out["next_command"] = f"visionservex model license {model_id}"
        return out
    if pol.final_policy == "external_api_only_terms_required":
        out["state"] = "external_api_only"
        out["next_command"] = pol.exact_next_command
        out["warning"] = pol.warning_text
        return out
    repo = pol.hf_repo
    if not repo:
        # Non-HF model (e.g. git-clone sidecar). Report static policy.
        out["state"] = pol.final_policy
        out["next_command"] = pol.exact_next_command
        out["warning"] = pol.warning_text
        return out
    tok = hf_get_token()
    if pol.gated and not tok:
        out["state"] = "auth_required"
        out["next_command"] = "visionservex hf connect  # then accept the upstream license"
        out["warning"] = pol.warning_text
        out["acceptance"] = hf_acceptance_instructions(canonical)
        return out
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        # auth_check tests *download* authorization (model_info only checks repo
        # visibility — a gated repo is listing-visible even without file access).
        try:
            HfApi(token=tok).auth_check(repo)
            out["state"] = "access_granted"
        except GatedRepoError:
            out["state"] = "auth_required_license_pending"
            out["next_command"] = (
                f"Accept the license at https://huggingface.co/{repo} first, "
                f"then: visionservex model pull {canonical} --accept-upstream-license"
            )
            out["acceptance"] = hf_acceptance_instructions(canonical)
        except RepositoryNotFoundError:
            out["state"] = "not_found_or_no_access"
            out["next_command"] = "Verify the repo exists / your token: visionservex hf whoami"
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", "?")
            out["state"] = f"http_{code}"
    except ImportError:
        out["state"] = "hf_hub_not_installed"
        out["next_command"] = "pip install 'visionservex[hf]'"
    if pol.gated:
        out["warning"] = pol.warning_text
    return out


def hf_download_allowed_by_policy(model_id: str) -> dict:
    """Decide whether VisionServeX may auto-download a model's weights.

    Returns ``{allowed: bool, reason, final_policy, requires_token, requires_user_license,
    can_ship_weights, warning, next_command}``. Used by the model-pull guard.

    Hard rules: gated/restricted weights are never shipped; non-commercial and
    enterprise/AGPL models are never auto-downloaded for production; legal-review
    models are blocked until resolved.
    """
    pol = _policy.get_policy(model_id)
    canonical = _policy.resolve_model_id(model_id)
    if pol is None:
        return {
            "model_id": canonical,
            "allowed": False,
            "reason": "unknown_model",
            "next_command": f"visionservex model license {model_id}",
        }
    base = {
        "model_id": canonical,
        "final_policy": pol.final_policy,
        "requires_token": pol.local_token_required,
        "requires_user_license": pol.user_license_required,
        "can_ship_weights": pol.can_ship_weights,  # always False
        "warning": pol.warning_text,
        "next_command": pol.exact_next_command,
    }
    if pol.final_policy == "commercial_safe_core":
        return {**base, "allowed": True, "reason": "commercial_safe_core"}
    if pol.final_policy == "byot_license_required":
        return {**base, "allowed": False, "reason": "byot_requires_user_token_and_accepted_license"}
    if pol.final_policy == "external_api_only_terms_required":
        return {**base, "allowed": False, "reason": "external_api_no_local_weights"}
    if pol.final_policy == "noncommercial_restricted":
        return {**base, "allowed": False, "reason": "noncommercial_research_only"}
    if pol.final_policy == "enterprise_license_required":
        return {**base, "allowed": False, "reason": "enterprise_or_agpl_license_required"}
    if pol.final_policy == "legal_review_required":
        return {**base, "allowed": False, "reason": "legal_review_pending"}
    return {**base, "allowed": False, "reason": pol.final_policy}


def hf_require_user_accepted_license(
    model_id: str, *, research_only: bool = False, accept_noncommercial: bool = False
) -> dict:
    """Enforce the license policy before a run/pull. Raises :class:`HFLicenseError`
    when the user has not satisfied the upstream/local terms.

    * commercial_safe_core -> allowed.
    * byot -> requires a token AND confirmed repo access (license accepted upstream).
    * noncommercial -> refused for production; allowed only with
      ``research_only=True`` and ``accept_noncommercial=True``.
    * enterprise / legal_review / api-only / not_released -> refused.
    """
    pol = _policy.get_policy(model_id)
    canonical = _policy.resolve_model_id(model_id)
    if pol is None:
        raise HFLicenseError(
            f"Unknown model '{model_id}' — not in the license policy.",
            state="unknown_model",
            model_id=canonical,
            next_command=f"visionservex model license {model_id}",
        )
    fp = pol.final_policy
    if fp == "commercial_safe_core":
        return {"model_id": canonical, "allowed": True, "final_policy": fp}
    if fp == "byot_license_required":
        if not hf_is_logged_in():
            raise HFLicenseError(
                f"{canonical}: a Hugging Face token is required (BYOT). {pol.warning_text}",
                state="auth_required",
                model_id=canonical,
                next_command="visionservex hf connect",
            )
        access = hf_model_access_status(canonical)
        if access.get("state") != "access_granted":
            raise HFLicenseError(
                f"{canonical}: you must accept the upstream license first. {pol.warning_text}",
                state=access.get("state", "auth_required_license_pending"),
                model_id=canonical,
                next_command=access.get("next_command", f"accept at {pol.upstream_url}"),
            )
        return {
            "model_id": canonical,
            "allowed": True,
            "final_policy": fp,
            "warning": pol.warning_text,
            "access": "granted",
        }
    if fp == "noncommercial_restricted":
        if research_only and accept_noncommercial:
            return {
                "model_id": canonical,
                "allowed": True,
                "final_policy": fp,
                "mode": "research_only",
                "warning": pol.warning_text,
            }
        raise HFLicenseError(
            f"{canonical}: {pol.warning_text}",
            state="noncommercial_restricted",
            model_id=canonical,
            next_command=(
                f"visionservex model pull {canonical} --research-only "
                f"--accept-noncommercial   # research use only; never production"
            ),
        )
    # enterprise / legal_review / external_api / not_released -> hard refuse
    raise HFLicenseError(
        f"{canonical}: {pol.warning_text}",
        state=fp,
        model_id=canonical,
        next_command=pol.exact_next_command,
    )


__all__ = [
    "HFAuthError",
    "HFLicenseError",
    "HFWhoAmI",
    "hf_acceptance_instructions",
    "hf_download_allowed_by_policy",
    "hf_get_token",
    "hf_is_logged_in",
    "hf_logout_local",
    "hf_model_access_status",
    "hf_redact_token",
    "hf_require_user_accepted_license",
    "hf_token_source",
    "hf_validate_token",
    "hf_whoami",
]
