import json

import boto3
import pytest
from boto3.dynamodb.types import TypeSerializer
from moto import mock_aws

from common.dynamodb import PARTITION_KEY, TABLE_NAME, to_dynamodb_item
from common.models import CandidateProfile, Posting
from lambdas.threshold.handler import handler

PROFILE_PATH = "common/candidate_profile.example.json"
CANDIDATE = CandidateProfile.from_file(PROFILE_PATH)

# Weights zero out every feature except embedding_similarity, so score is a simple,
# fully predictable function of it: sigmoid(-5 + 10 * embedding_similarity).
MODEL = {
    "weights": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
    "intercept": -5.0,
    "feature_names": [
        "skill_overlap_count",
        "seniority_match",
        "comp_range_fit",
        "remote_match",
        "visa_match",
        "embedding_similarity",
    ],
}

_serializer = TypeSerializer()


def _posting(posting_hash: str, embedding_similarity: float) -> Posting:
    return Posting(
        posting_id=posting_hash,
        source="remoteok",
        source_url=f"https://example.com/job/{posting_hash}",
        posting_hash=posting_hash,
        ingested_at="2026-09-06T12:00:00+00:00",
        title="Backend Engineer",
        company="Acme Corp",
        embedding_similarity=embedding_similarity,
    )


def _modify_stream_event(posting: Posting) -> dict:
    item = to_dynamodb_item(posting)
    new_image = {k: _serializer.serialize(v) for k, v in item.items()}
    return {"Records": [{"eventName": "MODIFY", "dynamodb": {"NewImage": new_image}}]}


@pytest.fixture
def aws(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", PROFILE_PATH)
    monkeypatch.setenv("FIT_THRESHOLD", "0.7")

    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(MODEL))
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(model_path))

    # Reset the module-level model cache between tests so each test's env var is honored.
    import lambdas.threshold.handler as handler_module

    handler_module._model_cache = None

    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": PARTITION_KEY, "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table = dynamodb.Table(TABLE_NAME)
        monkeypatch.setenv("POSTINGS_TABLE_NAME", TABLE_NAME)

        sns = boto3.client("sns", region_name="us-east-1")
        topic_arn = sns.create_topic(Name="jobpulse-alerts-test")["TopicArn"]
        monkeypatch.setenv("ALERTS_TOPIC_ARN", topic_arn)

        sqs = boto3.client("sqs", region_name="us-east-1")
        queue_url = sqs.create_queue(QueueName="alerts-inbox")["QueueUrl"]
        queue_arn = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])[
            "Attributes"
        ]["QueueArn"]
        sns.subscribe(TopicArn=topic_arn, Protocol="sqs", Endpoint=queue_arn)

        yield table, sqs, queue_url


def test_high_score_posting_gets_scored_and_alerted(aws):
    table, sqs, queue_url = aws
    posting = _posting("high", embedding_similarity=1.0)
    table.put_item(Item=to_dynamodb_item(posting))

    result = handler(_modify_stream_event(posting), None)

    assert result["processed"] == 1
    assert result["results"][0]["alert_sent"] is True
    assert result["results"][0]["score"] > 0.7

    item = table.get_item(Key={"posting_id": "high"})["Item"]
    assert float(item["score"]) > 0.7
    assert item["alert_sent"] is True

    messages = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=1).get("Messages", [])
    assert len(messages) == 1
    sns_envelope = json.loads(messages[0]["Body"])
    alert_body = json.loads(sns_envelope["Message"])
    assert alert_body["posting_id"] == "high"
    assert alert_body["company"] == "Acme Corp"


def test_low_score_posting_gets_scored_but_not_alerted(aws):
    table, sqs, queue_url = aws
    posting = _posting("low", embedding_similarity=0.0)
    table.put_item(Item=to_dynamodb_item(posting))

    result = handler(_modify_stream_event(posting), None)

    assert result["results"][0]["alert_sent"] is False
    assert result["results"][0]["score"] < 0.7

    item = table.get_item(Key={"posting_id": "low"})["Item"]
    assert item["alert_sent"] is False

    messages = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=1).get("Messages", [])
    assert messages == []


def test_non_modify_records_are_ignored(aws):
    _table, sqs, queue_url = aws
    posting = _posting("high", embedding_similarity=1.0)
    event = _modify_stream_event(posting)
    event["Records"][0]["eventName"] = "INSERT"

    result = handler(event, None)

    assert result == {"processed": 0, "results": []}
    messages = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=1).get("Messages", [])
    assert messages == []
