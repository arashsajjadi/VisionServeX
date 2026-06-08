# Anastig SaaS policy (hosted VisionServeX)

When VisionServeX powers a hosted, multi-tenant SaaS (Anastig), the BYOT and
license-policy guarantees must hold **per tenant**. This document is the design
contract for that.

## Principles

1. **OAuth per user — never a shared platform token.**
   Each user connects their own Hugging Face account via "Sign in with Hugging
   Face" (OAuth). The platform never holds a single token that downloads gated
   weights on everyone's behalf.

2. **Encrypted, per-user secret storage.**
   - Tokens / OAuth grants are encrypted at rest (envelope encryption, KMS-managed
     data keys), scoped to the owning user.
   - Secrets are never written to logs, shared buckets, notebooks, reports, or CI.
   - Every log line redacts tokens to `hf_***xx` form (the same
     `visionservex.hf_auth.hf_redact_token` contract used locally).
   - Token rotation / deletion removes the encrypted secret and invalidates any
     cached access immediately.

3. **License acceptance is per user.**
   A user may run a gated model only if *their* Hub account has accepted *that*
   model's upstream license. The platform never accepts a license on a user's
   behalf, and never serves one tenant's weights cache to another tenant.

4. **Policy enforced server-side.**
   The `v38_license_policy_matrix` (`production_allowed`, `commercial_safe`,
   `default_safe`) is enforced in the API layer:
   - `commercial_safe_core` → allowed for all plans.
   - `byot_license_required` → allowed only for users who connected a token and
     accepted the upstream license.
   - `noncommercial_restricted` → blocked for any paid/commercial workload; at most
     a clearly-labelled research mode for eligible users.
   - `enterprise_license_required` / `legal_review_required` /
     `external_api_only_terms_required` → blocked from production until resolved /
     licensed / provider-keyed by the tenant.

5. **No weight redistribution.**
   The platform downloads weights into per-user caches on the user's behalf. It
   never re-exports, mirrors, or bundles gated/restricted weights, and never ships
   them in container images.

6. **External-API models keep data-egress explicit.**
   For `external_api_only_terms_required` models, the tenant supplies their own
   provider API key, and the UI states clearly that data leaves the environment
   and is governed by the provider's terms.

## Isolation

- Per-tenant weight caches (no cross-tenant cache sharing).
- Per-tenant audit log of which models were pulled / run, with redacted identities.
- Resource guards (RAM/VRAM/disk) per worker, mirroring the local
  `resource_guard` budgets, to prevent one tenant's job from starving others.

## Mapping to the local implementation

| Local (today) | SaaS (Anastig) |
|---|---|
| `huggingface-cli login` / `visionservex hf connect` | "Sign in with Hugging Face" OAuth |
| HF cache on disk | encrypted per-user secret store + per-user cache |
| `--accept-upstream-license` flag | per-user license-acceptance record |
| `hf_redact_token` in logs | platform-wide redaction middleware |
| `visionservex.licensing.policy` | same policy table enforced in the API gateway |
