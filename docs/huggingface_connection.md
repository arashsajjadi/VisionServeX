# Connecting your Hugging Face account (BYOT)

VisionServeX uses a **Bring Your Own Token (BYOT)** model for any gated or
license-required model (e.g. SAM 3 / SAM 3.1, DINOv3). You connect *your own*
Hugging Face account, you accept the upstream license *yourself*, and the weights
are downloaded into *your own* Hugging Face cache.

> **VisionServeX does not redistribute gated or restricted model weights.**
> A Hugging Face token grants *you* access under the terms *you* accepted — it
> never grants redistribution rights, and VisionServeX never bundles weights into
> the PyPI wheel, the GitHub repo, or any Docker image.

---

## 1. Local CLI token flow (today)

### Create a token

1. Go to <https://huggingface.co/settings/tokens>.
2. Create a token. A **read** (or fine-grained read) scope is enough to download
   models you have been granted access to.

### Connect it

Pick one of:

```bash
# (a) standard Hugging Face login — stores the token in the HF cache
huggingface-cli login

# (b) from an environment variable (never printed)
export HF_TOKEN=hf_...
visionservex hf connect --token-env HF_TOKEN

# (c) from a private local file (never printed, never copied)
visionservex hf connect --token-file /path/to/token.txt
```

VisionServeX detects a token in this order: **(1)** the `huggingface-cli login`
cache, **(2)** the `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` environment variable,
**(3)** a private local file. The token is stored only in the standard Hugging
Face cache — **never** in this repository, in notebooks, in reports, or in CI.

### Check the connection

```bash
visionservex hf status            # shows login state; the token is always redacted
visionservex hf whoami            # shows your username / org
visionservex hf status --json     # machine-readable
```

Every command redacts the token to `hf_***xx` form — only the first 3 and last 2
characters are ever shown.

---

## 2. Accept the upstream license yourself

Gated models require you to accept the model owner's license on the Hub **before**
your token can download the weights.

```bash
# 1) See exactly what is required and where:
visionservex model license sam3-base
visionservex hf check-model facebook/sam3        # access status, no download

# 2) Accept on the model page (one click):
#    https://huggingface.co/facebook/sam3
#    https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

# 3) Pull (BYOT) once accepted:
visionservex model pull sam3-base --accept-upstream-license
```

`--accept-upstream-license` is your explicit confirmation that you have read and
accepted the upstream terms. Without it, gated pulls are refused with the exact
next step.

### Python API

```python
from visionservex import VSX

VSX.hf.status()                  # {'logged_in': True, 'name': ..., 'token_redacted': 'hf_***xx'}
VSX.hf.whoami()
VSX.model("sam3-base").license()       # full policy row
VSX.model("sam3-base").access()        # 'access_granted' / 'auth_required' / ...
VSX.model("sam3-base").pull(accept_upstream_license=True)
```

---

## 3. Running restricted (non-commercial) models locally

Some models are **non-commercial / research-only** (e.g. EdgeSAM, LocateAnything,
the larger Depth-Anything V2 weights). They are **never** production-allowed and
**never** in the commercial-safe core. They can be run locally for research only,
and only with an explicit acknowledgement:

```bash
visionservex model pull edge-sam --research-only --accept-noncommercial
```

```python
VSX.model("edge-sam").pull(research_only=True, accept_noncommercial=True)
```

Enterprise / AGPL models (FastSAM, Ultralytics YOLO-seg) and `legal_review`
models (HQ-SAM, TinySAM, OneFormer, InternImage) are refused for production until
an enterprise license is obtained or the review is resolved. See
[restricted_models.md](restricted_models.md) and
[model_license_policy.md](model_license_policy.md).

---

## 4. Revoking / disconnecting your token

```bash
visionservex hf logout            # clears the cached HF login on this machine
```

Then, to fully revoke:

- Delete or rotate the token at <https://huggingface.co/settings/tokens>.
- Unset any `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` env vars you exported.
- Remove any private token file you created.

`visionservex hf logout` only clears the local cache login; the environment
variables and any token files remain yours to remove.

---

## 5. Future: web OAuth flow

The CLI BYOT flow is the supported path today. The intended web experience:

- A future VisionServeX web app will use **Hugging Face OAuth** ("Sign in with
  Hugging Face") so a user authorizes access through the Hub itself rather than
  pasting a token.
- The user still owns the token/grant and still accepts each upstream model
  license on the Hub.
- VisionServeX still never redistributes weights; it downloads them, on the
  user's behalf, into a per-user cache.

---

## 6. Anastig SaaS policy (per-user secret storage)

When VisionServeX powers a hosted SaaS (Anastig), the BYOT principle is preserved
per tenant:

- **OAuth per user**, not a shared platform token. Each user connects their own
  Hugging Face account.
- **Encrypted per-user secret storage**: tokens are stored encrypted at rest
  (e.g. envelope encryption with a KMS-managed key), scoped to the user, never
  logged, never written to shared storage, and redacted in every log line.
- **License acceptance is per user**: a user may only run a gated model if *their*
  account has accepted *that* model's upstream license. The platform never
  accepts a license on a user's behalf and never shares one user's weights cache
  with another tenant.
- **No commercial use of non-commercial models** on behalf of paying customers —
  the policy matrix (`production_allowed=False`) is enforced server-side.
- **Token revocation** propagates immediately: deleting the connection removes the
  encrypted secret and invalidates cached access.

See [anastig_saas_policy.md](anastig_saas_policy.md) for the full design.
