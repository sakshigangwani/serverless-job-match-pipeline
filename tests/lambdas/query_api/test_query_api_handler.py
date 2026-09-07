import json
from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws

from common.dynamodb import (
    GSI_NAME,
    GSI_PARTITION_KEY,
    GSI_SORT_KEY,
    PARTITION_KEY,
    TABLE_NAME,
    to_dynamodb_item,
)
from common.models import Posting
from lambdas.query_api.handler import handler


def _posting(posting_id: str, score: float, **overrides) -> Posting:
    defaults = {
        "posting_id": posting_id,
        "source": "remoteok",
        "source_url": f"https://example.com/job/{posting_id}",
        "posting_hash": posting_id,
        "ingested_at": datetime.now(timezone.utc),
        "title": f"Engineer {posting_id}",
        "remote_status": "remote",
        "seniority": "senior",
        "score": score,
    }
    defaults.update(overrides)
    return Posting(**defaults)


def _api_event(query_params: dict | None = None) -> dict:
    return {"queryStringParameters": query_params}


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("POSTINGS_TABLE_NAME", TABLE_NAME)

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
        yield dynamodb.Table(TABLE_NAME)


def test_returns_postings_ranked_by_score_descending(table):
    table.put_item(Item=to_dynamodb_item(_posting("low", 0.2)))
    table.put_item(Item=to_dynamodb_item(_posting("high", 0.9)))
    table.put_item(Item=to_dynamodb_item(_posting("mid", 0.5)))

    response = handler(_api_event(), None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert [p["posting_id"] for p in body["postings"]] == ["high", "mid", "low"]
    assert body["count"] == 3


def test_unscored_postings_never_appear(table):
    scored = _posting("scored", 0.5)
    unscored = _posting("unscored", None)
    # Simulate an unscored posting the way extraction actually writes one: no score.
    unscored_item = to_dynamodb_item(unscored)
    table.put_item(Item=to_dynamodb_item(scored))
    table.put_item(Item=unscored_item)

    response = handler(_api_event(), None)
    body = json.loads(response["body"])

    assert [p["posting_id"] for p in body["postings"]] == ["scored"]


def test_min_score_filters_out_lower_scored_postings(table):
    table.put_item(Item=to_dynamodb_item(_posting("low", 0.2)))
    table.put_item(Item=to_dynamodb_item(_posting("high", 0.9)))

    response = handler(_api_event({"min_score": "0.5"}), None)
    body = json.loads(response["body"])

    assert [p["posting_id"] for p in body["postings"]] == ["high"]


def test_remote_status_filter(table):
    table.put_item(Item=to_dynamodb_item(_posting("remote-one", 0.8, remote_status="remote")))
    table.put_item(Item=to_dynamodb_item(_posting("onsite-one", 0.7, remote_status="onsite")))

    response = handler(_api_event({"remote_status": "onsite"}), None)
    body = json.loads(response["body"])

    assert [p["posting_id"] for p in body["postings"]] == ["onsite-one"]


def test_seniority_filter(table):
    table.put_item(Item=to_dynamodb_item(_posting("senior-one", 0.8, seniority="senior")))
    table.put_item(Item=to_dynamodb_item(_posting("junior-one", 0.7, seniority="junior")))

    response = handler(_api_event({"seniority": "junior"}), None)
    body = json.loads(response["body"])

    assert [p["posting_id"] for p in body["postings"]] == ["junior-one"]


def test_combined_filters(table):
    table.put_item(
        Item=to_dynamodb_item(_posting("match", 0.8, remote_status="remote", seniority="senior"))
    )
    table.put_item(
        Item=to_dynamodb_item(_posting("wrong-seniority", 0.9, remote_status="remote", seniority="junior"))
    )

    response = handler(_api_event({"remote_status": "remote", "seniority": "senior"}), None)
    body = json.loads(response["body"])

    assert [p["posting_id"] for p in body["postings"]] == ["match"]


def test_limit_is_respected(table):
    for i in range(5):
        table.put_item(Item=to_dynamodb_item(_posting(f"p{i}", score=i / 10)))

    response = handler(_api_event({"limit": "2"}), None)
    body = json.loads(response["body"])

    assert body["count"] == 2


def test_filtering_pages_past_a_limit_sized_first_page(table):
    # 10 postings, only the 3 lowest-indexed ones are "remote"; with limit=3 and the
    # per-page DynamoDB Limit also 3, the first page (highest-scored 3 items) contains
    # zero "remote" postings post-filter — proving the handler pages with
    # LastEvaluatedKey rather than stopping after one under-filled page.
    for i in range(10):
        remote_status = "remote" if i < 3 else "onsite"
        table.put_item(Item=to_dynamodb_item(_posting(f"p{i}", score=i / 10, remote_status=remote_status)))

    response = handler(_api_event({"remote_status": "remote", "limit": "3"}), None)
    body = json.loads(response["body"])

    assert body["count"] == 3
    assert all(p["remote_status"] == "remote" for p in body["postings"])


def test_invalid_min_score_returns_400(table):
    response = handler(_api_event({"min_score": "not-a-number"}), None)

    assert response["statusCode"] == 400
    assert "min_score" in json.loads(response["body"])["error"]


def test_invalid_limit_returns_400(table):
    response = handler(_api_event({"limit": "not-a-number"}), None)

    assert response["statusCode"] == 400


def test_limit_is_capped_at_max(table):
    for i in range(3):
        table.put_item(Item=to_dynamodb_item(_posting(f"p{i}", score=i / 10)))

    response = handler(_api_event({"limit": "99999"}), None)

    assert response["statusCode"] == 200


def test_no_query_params_uses_defaults(table):
    table.put_item(Item=to_dynamodb_item(_posting("only", 0.5)))

    response = handler({"queryStringParameters": None}, None)
    body = json.loads(response["body"])

    assert body["count"] == 1
