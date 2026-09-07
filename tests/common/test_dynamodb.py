from datetime import datetime, timezone
from decimal import Decimal

import pytest
from boto3.dynamodb.types import Binary

from common.dynamodb import (
    GSI_PARTITION_KEY,
    GSI_PARTITION_VALUE,
    PARTITION_KEY,
    from_dynamodb_item,
    pack_embedding,
    to_dynamodb_item,
    unpack_embedding,
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


def test_to_item_converts_embedding_similarity_to_decimal():
    item = to_dynamodb_item(_sample_posting(embedding_similarity=0.732))

    assert isinstance(item["embedding_similarity"], Decimal)
    assert item["embedding_similarity"] == Decimal("0.732")


def test_to_item_handles_missing_embedding_similarity():
    item = to_dynamodb_item(_sample_posting(embedding_similarity=None))

    assert item["embedding_similarity"] is None


def test_item_round_trip_preserves_embedding_similarity():
    original = _sample_posting(embedding_similarity=0.5)
    restored = from_dynamodb_item(to_dynamodb_item(original))

    assert restored == original
    assert isinstance(restored.embedding_similarity, float)


def test_pack_embedding_returns_dynamodb_binary_type():
    packed = pack_embedding([0.1, 0.2, 0.3])

    assert isinstance(packed, Binary)


def test_pack_embedding_is_compact_4_bytes_per_float():
    vector = [0.1] * 1024

    packed = pack_embedding(vector)

    assert len(bytes(packed)) == 1024 * 4


def test_embedding_round_trip_preserves_values_to_float32_precision():
    original = [0.1, -2.5, 3.14159, 0.0, 100.0]

    restored = unpack_embedding(pack_embedding(original))

    assert restored == pytest.approx(original, rel=1e-6)


def test_unpack_embedding_accepts_raw_bytes_not_just_binary_wrapper():
    packed = pack_embedding([1.0, 2.0])

    restored = unpack_embedding(bytes(packed))

    assert restored == pytest.approx([1.0, 2.0])
