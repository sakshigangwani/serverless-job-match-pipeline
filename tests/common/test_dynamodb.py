from datetime import datetime, timezone
from decimal import Decimal

from common.dynamodb import (
    GSI_PARTITION_KEY,
    GSI_PARTITION_VALUE,
    PARTITION_KEY,
    from_dynamodb_item,
    to_dynamodb_item,
)
from common.models import Posting


def _sample_posting(**overrides) -> Posting:
    defaults = {
        "posting_id": "deadbeef",
        "source": "remoteok",
        "source_url": "https://example.com/job/1",
        "posting_hash": "deadbeef",
        "ingested_at": datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc),
        "title": "Backend Engineer",
        "required_skills": ["Python", "AWS"],
        "score": 0.87,
    }
    defaults.update(overrides)
    return Posting(**defaults)


def test_to_item_sets_gsi_partition_for_score_ranked_queries():
    item = to_dynamodb_item(_sample_posting())

    assert item[GSI_PARTITION_KEY] == GSI_PARTITION_VALUE
    assert item[PARTITION_KEY] == "deadbeef"


def test_to_item_converts_score_to_decimal_for_dynamodb():
    item = to_dynamodb_item(_sample_posting(score=0.87))

    assert isinstance(item["score"], Decimal)
    assert item["score"] == Decimal("0.87")


def test_to_item_handles_unscored_posting():
    item = to_dynamodb_item(_sample_posting(score=None))

    assert item["score"] is None


def test_item_round_trip_preserves_posting_data():
    original = _sample_posting()
    item = to_dynamodb_item(original)
    restored = from_dynamodb_item(item)

    assert restored == original


def test_from_item_ignores_gsi_partition_key():
    item = to_dynamodb_item(_sample_posting())
    restored = from_dynamodb_item(item)

    assert not hasattr(restored, GSI_PARTITION_KEY)
