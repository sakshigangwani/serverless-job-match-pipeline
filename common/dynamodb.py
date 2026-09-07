"""DynamoDB item shape for postings (PLAN.md Phase 1.3).

The table itself is created in Phase 5 (CDK); this module is the single place that defines
its key schema and the item <-> Posting conversion, so every Lambda that touches DynamoDB
(extract, embed, threshold, query_api, skill_gap) agrees on the same shape.

Key schema:
- Partition key: `posting_id` (string) — set to the posting's content hash, so writing the
  same posting twice overwrites rather than duplicates (dedup, spec section 5.1).
- GSI `ScoreIndex`: partition key `gsi_pk` (constant "POSTING"), sort key `score` (Number).
  Querying this GSI with ScanIndexForward=False returns postings ranked highest-score-first
  without a table scan — this is what the query API (Phase 8) and skill-gap analysis
  (Phase 14) rely on.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from common.models import Posting

TABLE_NAME = "JobPulsePostings"
PARTITION_KEY = "posting_id"
GSI_NAME = "ScoreIndex"
GSI_PARTITION_KEY = "gsi_pk"
GSI_PARTITION_VALUE = "POSTING"
GSI_SORT_KEY = "score"


def to_dynamodb_item(posting: Posting) -> dict[str, Any]:
    """Convert a Posting into a DynamoDB-ready item (Decimal scores, ISO-8601 timestamps)."""
    item: dict[str, Any] = posting.model_dump(mode="python")
    item["ingested_at"] = posting.ingested_at.isoformat()
    item["score"] = Decimal(str(posting.score)) if posting.score is not None else None
    item[GSI_PARTITION_KEY] = GSI_PARTITION_VALUE
    return item


def from_dynamodb_item(item: dict[str, Any]) -> Posting:
    """Convert a raw DynamoDB item back into a validated Posting."""
    data = {k: v for k, v in item.items() if k != GSI_PARTITION_KEY}
    if isinstance(data.get("ingested_at"), str):
        data["ingested_at"] = datetime.fromisoformat(data["ingested_at"])
    if isinstance(data.get("score"), Decimal):
        data["score"] = float(data["score"])
    return Posting.model_validate(data)
