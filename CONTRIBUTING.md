# Contributing to VisionServeX

Thank you for considering a contribution. VisionServeX aims to be a friendly,
secure, and predictable framework for serving permissive-license computer
vision models.

## Quick start

```bash
git clone <your-fork>
cd VisionServeX
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,server]"
pre-commit install
pytest
```

## Coding standards

- Python 3.10+.
- `ruff format` and `ruff check` must pass.
- `mypy src/visionservex` should report no new errors.
- Public APIs should have type hints and short, English-only docstrings.
- Avoid adding new top-level dependencies; prefer optional extras.
- New engines must implement the `BaseEngine` interface.
- New models must be registered with full metadata in
  `src/visionservex/registry/models.yaml`, including license and status.

## Security

Security defaults are non-negotiable:

- Server binds to `127.0.0.1` by default.
- Public exposure must require explicit opt-in and API key.
- Never log secrets. Use `visionservex.utils.logging.redact`.
- Path inputs must be validated against traversal.
- Remote URL inputs must pass the SSRF guard.

If you find a security issue, please follow `SECURITY.md` rather than opening a
public issue.

## Tests

- Unit tests live in `tests/`.
- Use the `MockEngine` for engine tests rather than downloading real weights.
- New CLI commands need at least one smoke test under
  `tests/test_cli.py`.

## Pull requests

- One logical change per PR.
- Update `CHANGELOG.md` under an `## [Unreleased]` section.
- Update relevant docs if behavior changes.
- Be explicit about licensing of any code you copy; do not include AGPL or
  unclear-license code.

## Code of conduct

Be kind and assume good faith. Discrimination, harassment, and personal
attacks are not tolerated.
