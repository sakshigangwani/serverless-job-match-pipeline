from datetime import datetime, timezone
from decimal import Decimal

import boto3
import pytest
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.types import Binary
from moto import mock_aws

from common.dynamodb import (
    GSI_NAME,
    GSI_PARTITION_KEY,
    GSI_PARTITION_VALUE,
    GSI_SORT_KEY,
    PARTITION_KEY,
    TABLE_NAME,
    from_dynamodb_item,
    pack_embedding,
    to_dynamodb_item,
    unpack_embedding,
)
from common.models import ContributingFactor, Posting


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


def test_to_item_omits_score_key_entirely_when_unscored():
    # `score` is present with DynamoDB NULL type otherwise, which DynamoDB rejects on
    # a table where `score` is a GSI sort key (ValidationException) — verified against
    # a real table with the GSI defined, not assumed. Omission, not NULL, is required.
    item = to_dynamodb_item(_sample_posting(score=None))

    assert "score" not in item


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


def test_to_item_omits_embedding_similarity_key_entirely_when_missing():
    item = to_dynamodb_item(_sample_posting(embedding_similarity=None))

    assert "embedding_similarity" not in item


def test_round_trip_restores_none_for_an_unscored_posting_via_pydantic_default():
    original = _sample_posting(score=None, embedding_similarity=None)

    restored = from_dynamodb_item(to_dynamodb_item(original))

    assert restored.score is None
    assert restored.embedding_similarity is None
    assert restored == original


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


def test_to_item_converts_top_factors_shap_values_to_decimal():
    posting = _sample_posting(
        top_factors=[
            ContributingFactor(feature="skill_overlap_count", shap_value=0.42),
            ContributingFactor(feature="remote_match", shap_value=-0.1),
        ]
    )

    item = to_dynamodb_item(posting)

    assert item["top_factors"] == [
        {"feature": "skill_overlap_count", "shap_value": Decimal("0.42")},
        {"feature": "remote_match", "shap_value": Decimal("-0.1")},
    ]
    assert all(isinstance(entry["shap_value"], Decimal) for entry in item["top_factors"])


def test_to_item_omits_top_factors_key_entirely_when_none():
    item = to_dynamodb_item(_sample_posting(top_factors=None))

    assert "top_factors" not in item


def test_item_round_trip_preserves_top_factors():
    original = _sample_posting(
        top_factors=[ContributingFactor(feature="embedding_similarity", shap_value=1.23456)]
    )

    restored = from_dynamodb_item(to_dynamodb_item(original))

    assert restored == original
    assert isinstance(restored.top_factors[0].shap_value, float)


def test_unscored_posting_can_actually_be_put_into_a_table_with_the_real_gsi():
    """Regression test for a real bug: writing `score` as DynamoDB NULL (rather than
    omitting the key) makes DynamoDB reject PutItem on a table where `score` is a GSI
    sort key, with a ValidationException — a unit test of to_dynamodb_item()'s returned
    dict alone can't catch this, since it never touches a real (or moto-emulated)
    DynamoDB table. Earlier phases' tests never caught it because their mock tables
    never defined the GSI at all.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": PARTITION_KEY, "AttributeType": "S"},
                {"AttributeName": GSI_PARTITION_KEY, "AttributeType": "S"},
                {"AttributeName": GSI_SORT_KEY, "AttributeType": "N"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": GSI_NAME,
                    "KeySchema": [
                        {"AttributeName": GSI_PARTITION_KEY, "KeyType": "HASH"},
                        {"AttributeName": GSI_SORT_KEY, "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table = dynamodb.Table(TABLE_NAME)

        unscored = _sample_posting(score=None)
        table.put_item(Item=to_dynamodb_item(unscored))  # must not raise

        # And it's correctly absent from the score-ranked index until scored.
        results = table.query(
            IndexName=GSI_NAME,
            KeyConditionExpression=Key(GSI_PARTITION_KEY).eq(GSI_PARTITION_VALUE),
        )
        assert results["Items"] == []
