"""DynamoDB item shape for postings (PLAN.md Phase 1.3, table created in Phase 5).

This module is the single place that defines the table's key schema and the item <->
Posting conversion, so every Lambda that touches DynamoDB (extract, embed, threshold,
query_api, skill_gap) agrees on the same shape.

Key schema:
- Partition key: `posting_id` (string) — set to the posting's content hash, so writing the
  same posting twice overwrites rather than duplicates (dedup, spec section 5.1).
- GSI `ScoreIndex`: partition key `gsi_pk` (constant "POSTING"), sort key `score` (Number).
  Querying this GSI with ScanIndexForward=False returns postings ranked highest-score-first
  without a table scan — this is what the query API (Phase 8) and skill-gap analysis
  (Phase 14) rely on. A posting without a `score` yet (before Phase 7 runs) is simply
  absent from this GSI — DynamoDB excludes items missing a scalar-typed GSI key attribute
  from that index rather than erroring, which is exactly the "not ready to rank yet"
  behavior this pipeline wants.

`embedding` (see pack_embedding/unpack_embedding) is a sibling item attribute, not a
Posting field — it isn't valid JSON-round-trippable binary data, so it's kept out of the
pydantic model and handled only at the DynamoDB item level.
"""
from __future__ import annotations

import struct
from datetime import datetime
from decimal import Decimal
from typing import Any

from boto3.dynamodb.types import Binary

from common.models import Posting

TABLE_NAME = "JobPulsePostings"
PARTITION_KEY = "posting_id"
GSI_NAME = "ScoreIndex"
GSI_PARTITION_KEY = "gsi_pk"
GSI_PARTITION_VALUE = "POSTING"
GSI_SORT_KEY = "score"
EMBEDDING_ATTRIBUTE = "embedding"


_DECIMAL_FIELDS = ("score", "embedding_similarity")


def to_dynamodb_item(posting: Posting) -> dict[str, Any]:
    """Convert a Posting into a DynamoDB-ready item (Decimal scores, ISO-8601 timestamps)."""
    item: dict[str, Any] = posting.model_dump(mode="python")
    item["ingested_at"] = posting.ingested_at.isoformat()
    for field in _DECIMAL_FIELDS:
        value = item[field]
        item[field] = Decimal(str(value)) if value is not None else None
    item[GSI_PARTITION_KEY] = GSI_PARTITION_VALUE
    return item


def from_dynamodb_item(item: dict[str, Any]) -> Posting:
    """Convert a raw DynamoDB item back into a validated Posting."""
    data = {k: v for k, v in item.items() if k != GSI_PARTITION_KEY}
    if isinstance(data.get("ingested_at"), str):
        data["ingested_at"] = datetime.fromisoformat(data["ingested_at"])
    for field in _DECIMAL_FIELDS:
        if isinstance(data.get(field), Decimal):
            data[field] = float(data[field])
    return Posting.model_validate(data)


def pack_embedding(vector: list[float]) -> Binary:
    """Pack a float embedding vector into DynamoDB's compact Binary type.

    A JSON array of floats is verbose (~15-20 bytes/float as text); packing each as a
    32-bit float is a flat 4 bytes/float — a 4-5x size reduction that matters given
    DynamoDB's 400KB item limit and size-based cost, especially at 1024+ dimensions
    (spec section 2.1 calls for storing embeddings "as compact base64/float arrays" —
    boto3's Binary type is that compact representation; the base64 encoding itself only
    happens implicitly on DynamoDB's HTTP wire format, not something this code does).
    """
    return Binary(struct.pack(f"{len(vector)}f", *vector))


def unpack_embedding(value: Binary | bytes) -> list[float]:
    """Inverse of pack_embedding. Note: round-tripping through 32-bit floats loses the
    precision Python's native 64-bit floats have — expect exact values back only to
    float32 precision (~7 significant digits), which is more than sufficient for cosine
    similarity or re-ranker features.
    """
    raw = bytes(value)
    count = len(raw) // struct.calcsize("f")
    return list(struct.unpack(f"{count}f", raw))
