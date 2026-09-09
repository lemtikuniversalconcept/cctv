from __future__ import annotations

import os
from typing import Any

try:
    import boto3
    from botocore.config import Config
except Exception:  # pragma: no cover - optional dependency until configured
    boto3 = None  # type: ignore
    Config = None  # type: ignore


# Cloudflare R2 is S3-compatible - boto3's generic "s3" client works against it by pointing
# endpoint_url at the account's R2 endpoint instead of AWS. Same pattern as osint's own
# operations/blob_storage.py and relationship_api's BLOB_STORAGE_* config: each service gets its
# own copy of the credentials so one service's storage outage can't take another down.
_R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL", "").strip()
_R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "").strip()
_R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
_R2_BUCKET = os.getenv("R2_BUCKET", "").strip()

_client: Any | None = None
_client_attempted = False


def _get_client() -> Any | None:
    global _client, _client_attempted
    if _client_attempted:
        return _client
    _client_attempted = True
    if boto3 is None or not (_R2_ENDPOINT and _R2_ACCESS_KEY and _R2_SECRET_KEY and _R2_BUCKET):
        return None
    try:
        _client = boto3.client(
            "s3",
            endpoint_url=_R2_ENDPOINT,
            aws_access_key_id=_R2_ACCESS_KEY,
            aws_secret_access_key=_R2_SECRET_KEY,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
    except Exception:
        _client = None
    return _client


def r2_configured() -> bool:
    return _get_client() is not None


def upload_bytes(key: str, content: bytes, content_type: str = "application/octet-stream") -> str | None:
    """Best-effort upload. Returns the object key on success, None on ANY failure - missing
    config, network error, auth error. Never raises; a frame that can't be archived should still
    get analyzed, not block the request."""
    client = _get_client()
    if client is None:
        return None
    try:
        client.put_object(Bucket=_R2_BUCKET, Key=key, Body=content, ContentType=content_type)
        return key
    except Exception:
        return None


def presigned_read_url(key: str, expires_seconds: int = 3600) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": _R2_BUCKET, "Key": key}, ExpiresIn=expires_seconds
        )
    except Exception:
        return None
