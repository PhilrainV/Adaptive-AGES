import base64
import hashlib

from cryptography.fernet import Fernet

from app.core.config import get_settings


def _fernet() -> Fernet:
    secret = get_settings().jwt_secret.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


def encrypt_secret(value: str | None) -> str | None:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8") if value else None


def decrypt_secret(value: str | None) -> str | None:
    return _fernet().decrypt(value.encode("utf-8")).decode("utf-8") if value else None
