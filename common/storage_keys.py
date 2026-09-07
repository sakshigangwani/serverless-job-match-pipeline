"""Posting content hashing and the S3 raw-landing-zone key convention (PLAN.md Phase 1.3).

`compute_posting_hash` is the identity used for dedup/change-detection (Phase 12) and, by
default, as the DynamoDB `posting_id` — see common/dynamodb.py.

Phase 3 and Phase 4 originally used interim S3 prefixes (structured/, embeddings/) for
their output, before DynamoDB existed. Phase 5 replaced both with DynamoDB writes, so
those key builders were removed — raw/ (below) and the candidate embeddings cache remain
the only S3 conventions.
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


# Fixed key: the candidate profile changes rarely (only when a resume is updated), so its
# embeddings are computed once and cached here rather than per-posting (PLAN.md Phase 4.1).
# Cache invalidation is manual for now — delete this object to force a recompute after
# changing the candidate profile.
CANDIDATE_EMBEDDINGS_KEY = "candidate/embeddings.json"
