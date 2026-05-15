# VisionServeX Threat Model

## Operating Modes and Threat Surface

### Mode 1: `local_private` (default)

**Threat surface: minimal.**

- Server binds to `127.0.0.1` — no network exposure.
- Attacker must have local code execution to reach the API.
- If they do: inference results and model outputs are accessible.
- Mitigation: local-only default is the strongest practical boundary.

**Threats we protect against:**
- Remote network attackers: ✅ blocked (loopback only)
- SSRF via image URL: ✅ disabled by default
- Path traversal: ✅ `safe_join` guards
- Decompression bombs: ✅ pixel/size limits
- Model weight leakage: ✅ weights in user cache, not served

**Threats we do NOT protect against:**
- Malicious local process with file access
- Side-channel via inference timing (not mitigated)
- Model inversion (inherent to ML systems)

---

### Mode 2: `lan_private`

**Threat surface: local network.**

- Bind to LAN interface.
- Any device on the same network can reach the API unless auth is on.
- **Auth required.** TLS strongly recommended.
- CORS allowlist required.

**Additional mitigations needed:**
- Enable `VISIONSERVEX_AUTH__ENABLED=true`
- Use TLS cert for the LAN interface
- Restrict CORS to known client origins

---

### Mode 3: `cloudflare_private`

**Threat surface: public internet via Cloudflare.**

- `cloudflared` creates an outbound tunnel — origin port stays closed.
- All traffic passes through Cloudflare edge.
- Cloudflare Access provides identity layer before reaching origin.
- **Auth required.** Service tokens recommended for programmatic clients.

**Mitigations in place:**
- API key mandatory (auth enabled by default in this mode)
- Cloudflare Access for browser users
- Service tokens for programmatic clients (CF-Access-Client-Id/Secret)
- Optional mTLS for highest-value clients
- Tunnel ingress has catch-all 404 (no accidental path leakage)
- Rate limiting in VisionServeX middleware
- SSRF guard on URL inputs

**What we cannot protect against:**
- Cloudflare compromise (out of scope; Cloudflare is a trusted intermediary)
- Leaked API keys (rotate keys, use secret managers)
- Model extraction via repeated API calls

---

### Mode 4: `production_multi_user`

**All of cloudflare_private plus:**
- Per-request audit logs
- Encrypted job store (Fernet/AES-128 for metadata at rest)
- Retention policy
- TLS required
- Auth required

---

## What VisionServeX CANNOT protect

1. **Model inversion / membership inference**: inherent to ML models.
   The model itself may leak information about training data.

2. **Side-channel attacks via timing**: inference latency can leak some
   information about input complexity. We do not add timing noise.

3. **Physical access to the inference machine**: if an attacker has physical
   or hypervisor-level access, plaintext tensors are accessible.

4. **True end-to-end encryption**: the server must see plaintext image
   tensors to run inference. We do not and cannot claim E2E encryption.

5. **Post-inference data use by the model provider**: if you use an
   external model whose weights were trained on private data, that is
   out of scope for VisionServeX.

## Key Management

Encryption keys must NEVER appear in:
- Source code
- Git history
- Log files
- API responses

Use: `visionservex security keygen` and store in a secrets manager or
environment variable outside of the repository.
