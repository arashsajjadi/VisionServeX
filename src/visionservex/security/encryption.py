# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Optional encryption-at-rest for sensitive job store fields.

Uses the ``cryptography`` package (Fernet / AES-128-CBC + HMAC-SHA256).
If ``cryptography`` is not installed, encryption is unavailable and the
caller receives a clear error rather than silently storing plaintext.

IMPORTANT:
VisionServeX cannot provide end-to-end encryption in the strict sense.
The inference server must see plaintext image tensors to run models.
This module only encrypts *stored metadata* at rest in the SQLite job store.
"""

from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

_MISSING_HINT = (
    "pip install cryptography  # then set VISIONSERVEX_ENCRYPTION_KEY or "
    "VISIONSERVEX_PRIVACY__ENCRYPTION_KEY_FILE"
)


class EncryptionUnavailableError(RuntimeError):
    pass


class EncryptionKeyError(RuntimeError):
    pass


def _load_fernet():
    try:
        from cryptography.fernet import Fernet, InvalidToken  # type: ignore

        return Fernet, InvalidToken
    except ImportError as exc:
        raise EncryptionUnavailableError(
            f"cryptography package is required for encryption-at-rest. {_MISSING_HINT}"
        ) from exc


def load_key(
    *,
    key_env: str = "VISIONSERVEX_ENCRYPTION_KEY",
    key_file: str | None = None,
) -> bytes:
    """Load a Fernet key from env var or file.  Never logs the key value."""
    raw: bytes | None = None

    # 1. Environment variable (highest priority)
    env_val = os.environ.get(key_env)
    if env_val:
        raw = env_val.encode("utf-8")

    # 2. Key file
    if raw is None and key_file:
        kf = Path(key_file).expanduser()
        if not kf.exists():
            raise EncryptionKeyError(
                f"Encryption key file not found: {kf}. "
                f"Generate one with: visionservex security keygen --out {kf}"
            )
        # Warn if permissions are too open
        mode = kf.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            import warnings

            warnings.warn(
                f"Key file {kf} is readable by group/others (permissions: {oct(mode)})."
                " Set to 0600 for security.",
                stacklevel=2,
            )
        raw = kf.read_bytes().strip()

    if raw is None:
        raise EncryptionKeyError(
            f"No encryption key found. Set {key_env} environment variable or "
            f"VISIONSERVEX_PRIVACY__ENCRYPTION_KEY_FILE, or generate a key with: "
            f"visionservex security keygen"
        )

    # Validate it's a valid Fernet key (32 url-safe base64 bytes)
    try:
        decoded = base64.urlsafe_b64decode(raw + b"==")
        if len(decoded) != 32:
            raise ValueError
    except Exception as exc:
        raise EncryptionKeyError(
            "Invalid Fernet key: must be 32 bytes encoded as URL-safe base64. "
            "Generate a new one with: visionservex security keygen"
        ) from exc

    return raw


def generate_key() -> bytes:
    """Generate a new Fernet key (32 random bytes, URL-safe base64)."""
    Fernet, _ = _load_fernet()
    return Fernet.generate_key()


class FieldEncryptor:
    """Encrypt/decrypt individual string fields using Fernet (AES-128-CBC + HMAC)."""

    def __init__(self, key: bytes) -> None:
        Fernet, self._InvalidToken = _load_fernet()
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """Return a base64-encoded ciphertext string."""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Return the original plaintext string or raise EncryptionKeyError."""
        try:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except self._InvalidToken as exc:
            raise EncryptionKeyError(
                "Decryption failed: wrong key or corrupted ciphertext."
            ) from exc


def get_encryptor(privacy_config) -> FieldEncryptor | None:
    """Return a FieldEncryptor if encryption is configured, else None."""
    if not privacy_config.encrypt_job_store:
        return None
    key = load_key(
        key_env=privacy_config.encryption_key_env,
        key_file=privacy_config.encryption_key_file,
    )
    return FieldEncryptor(key)


__all__ = [
    "EncryptionKeyError",
    "EncryptionUnavailableError",
    "FieldEncryptor",
    "generate_key",
    "get_encryptor",
    "load_key",
]
