"""Posting content hashing and the S3 raw-landing-zone key convention (PLAN.md Phase 1.3).

`compute_posting_hash` is the identity used for dedup/change-detection (Phase 12) and, by
default, as the DynamoDB `posting_id` — see common/dynamodb.py.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime


def compute_posting_hash(source: str, source_url: str, description_text: str) -> str:
    """Deterministic content hash for a posting.

    Hashing `source_url` + `description_text` (rather than the full raw payload) means an
    edited/re-posted listing at the same URL with changed content produces a different
    hash, which is exactly the "detect and diff re-posted/edited listings" behavior spec
    section 5.1 asks for.
    """
    normalized = f"{source}\n{source_url}\n{description_text}".strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def raw_posting_key(source: str, when: date | datetime | str, posting_hash: str) -> str:
    """Build the immutable S3 landing-zone key: raw/{source}/{date}/{posting_hash}.json"""
    if isinstance(when, (date, datetime)):
        date_str = when.strftime("%Y-%m-%d")
    else:
        date_str = when
    return f"raw/{source}/{date_str}/{posting_hash}.json"


def structured_posting_key(source: str, when: date | datetime | str, posting_hash: str) -> str:
    """Build the extraction Lambda's output key: structured/{source}/{date}/{posting_hash}.json

    Interim persistence for the extracted Posting record, mirroring the raw/ convention.
    Phase 5 introduces DynamoDB as the real structured store; until then this S3 prefix
    lets extraction (Phase 3) be built, deployed, and tested independently.
    """
    if isinstance(when, (date, datetime)):
        date_str = when.strftime("%Y-%m-%d")
    else:
        date_str = when
    return f"structured/{source}/{date_str}/{posting_hash}.json"
