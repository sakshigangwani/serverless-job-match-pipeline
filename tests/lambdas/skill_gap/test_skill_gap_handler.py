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
from common.models import CandidateProfile, Posting
from lambdas.skill_gap.handler import compute_skill_gaps, handler

CANDIDATE = CandidateProfile(
    candidate_id="c1",
    resume_text="",
    skills=["Python", "AWS"],
)


def _posting(posting_id: str, score: float, required_skills: list[str], **overrides) -> Posting:
    defaults = {
        "posting_id": posting_id,
        "source": "remoteok",
        "source_url": f"https://example.com/job/{posting_id}",
        "posting_hash": posting_id,
        "ingested_at": datetime.now(timezone.utc),
        "title": f"Engineer {posting_id}",
        "required_skills": required_skills,
        "score": score,
    }
    defaults.update(overrides)
    return Posting(**defaults)


def _api_event(query_params: dict | None = None) -> dict:
    return {"queryStringParameters": query_params}


def test_compute_skill_gaps_ranks_by_frequency_descending():
    postings = [
        _posting("a", 0.1, ["Python", "Rust", "Go"]),
        _posting("b", 0.2, ["Rust", "Go"]),
        _posting("c", 0.3, ["Rust"]),
    ]

    result = compute_skill_gaps(postings, CANDIDATE)

    assert result == [
        {"skill": "Rust", "missing_count": 3},
        {"skill": "Go", "missing_count": 2},
    ]


def test_compute_skill_gaps_excludes_candidate_skills_case_insensitively():
    postings = [_posting("a", 0.1, ["python", "AWS", "Kubernetes"])]

    result = compute_skill_gaps(postings, CANDIDATE)

    assert result == [{"skill": "Kubernetes", "missing_count": 1}]


def test_compute_skill_gaps_tie_breaks_alphabetically():
    postings = [_posting("a", 0.1, ["Zig", "Elixir"])]

    result = compute_skill_gaps(postings, CANDIDATE)

    assert [entry["skill"] for entry in result] == ["Elixir", "Zig"]


def test_compute_skill_gaps_respects_limit():
    postings = [_posting("a", 0.1, ["Rust", "Go", "Zig", "Elixir"])]

    result = compute_skill_gaps(postings, CANDIDATE, limit=2)

    assert len(result) == 2


def test_compute_skill_gaps_returns_empty_for_no_postings():
    assert compute_skill_gaps([], CANDIDATE) == []


@pytest.fixture
def table(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("POSTINGS_TABLE_NAME", TABLE_NAME)
    monkeypatch.setenv("FIT_THRESHOLD", "0.7")

    profile_path = tmp_path / "candidate.json"
    profile_path.write_text(json.dumps(CANDIDATE.model_dump(mode="json")))
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", str(profile_path))

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


def test_handler_only_analyzes_postings_below_fit_threshold(table):
    table.put_item(Item=to_dynamodb_item(_posting("low", 0.2, ["Rust"])))
    table.put_item(Item=to_dynamodb_item(_posting("high", 0.9, ["Kubernetes"])))

    response = handler(_api_event(), None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body["analyzed_postings"] == 1
    assert body["skill_gaps"] == [{"skill": "Rust", "missing_count": 1}]


def test_handler_unscored_postings_are_never_analyzed(table):
    # Unscored postings are absent from the ScoreIndex GSI entirely (PLAN.md Phase 5's
    # sparse-index behavior), so they can't distort the gap analysis either.
    table.put_item(Item=to_dynamodb_item(_posting("unscored", None, ["Rust"])))
    table.put_item(Item=to_dynamodb_item(_posting("low", 0.1, ["Go"])))

    response = handler(_api_event(), None)
    body = json.loads(response["body"])

    assert body["analyzed_postings"] == 1
    assert body["skill_gaps"] == [{"skill": "Go", "missing_count": 1}]


def test_handler_respects_limit_query_param(table):
    table.put_item(Item=to_dynamodb_item(_posting("a", 0.1, ["Rust", "Go", "Zig"])))

    response = handler(_api_event({"limit": "1"}), None)
    body = json.loads(response["body"])

    assert len(body["skill_gaps"]) == 1


def test_handler_invalid_limit_returns_400(table):
    response = handler(_api_event({"limit": "not-a-number"}), None)

    assert response["statusCode"] == 400
    assert "limit" in json.loads(response["body"])["error"]


def test_handler_no_query_params_uses_defaults(table):
    table.put_item(Item=to_dynamodb_item(_posting("a", 0.1, ["Rust"])))

    response = handler({"queryStringParameters": None}, None)

    assert response["statusCode"] == 200


def test_handler_returns_empty_gaps_when_nothing_is_below_threshold(table):
    table.put_item(Item=to_dynamodb_item(_posting("high", 0.9, ["Rust"])))

    response = handler(_api_event(), None)
    body = json.loads(response["body"])

    assert body == {"skill_gaps": [], "analyzed_postings": 0}
