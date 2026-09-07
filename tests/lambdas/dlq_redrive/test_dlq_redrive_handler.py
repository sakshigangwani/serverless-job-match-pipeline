import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from boto3.dynamodb.types import Binary
from moto import mock_aws

from common.dynamodb import PARTITION_KEY, TABLE_NAME
from lambdas.dlq_redrive.handler import handler

EXTRACT_QUEUE_NAME = "extract-dlq-test"
EMBED_QUEUE_NAME = "embed-dlq-test"

ORIGINAL_S3_EVENT = {
    "Records": [{"s3": {"bucket": {"name": "raw-bucket"}, "object": {"key": "raw/remoteok/2026-09-06/abc.json"}}}]
}


def _lambda_destination_failure_message(original_event: dict) -> str:
    """Mimics the real envelope shape a Lambda Destinations on-failure target delivers."""
    return json.dumps(
        {
            "requestContext": {"condition": "RetriesExhausted"},
            "requestPayload": original_event,
            "responsePayload": {"errorType": "ExtractionError", "errorMessage": "boom"},
        }
    )


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("EXTRACT_FUNCTION_NAME", "ExtractLambda")
    monkeypatch.setenv("EMBED_FUNCTION_NAME", "EmbedLambda")
    monkeypatch.setenv("POSTINGS_TABLE_NAME", TABLE_NAME)

    lambda_mock = MagicMock()
    original_client = boto3.client

    def fake_client(service_name, *args, **kwargs):
        if service_name == "lambda":
            return lambda_mock
        return original_client(service_name, *args, **kwargs)

    with mock_aws():
        sqs = original_client("sqs", region_name="us-east-1")
        extract_queue_url = sqs.create_queue(QueueName=EXTRACT_QUEUE_NAME)["QueueUrl"]
        embed_queue_url = sqs.create_queue(QueueName=EMBED_QUEUE_NAME)["QueueUrl"]
        monkeypatch.setenv("EXTRACT_DLQ_URL", extract_queue_url)
        monkeypatch.setenv("EMBED_DLQ_URL", embed_queue_url)

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": PARTITION_KEY, "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table = dynamodb.Table(TABLE_NAME)

        with patch("boto3.client", side_effect=fake_client):
            yield sqs, extract_queue_url, embed_queue_url, table, lambda_mock


def test_redrive_extract_replays_original_s3_event_and_empties_queue(aws):
    sqs, extract_queue_url, _embed_url, _table, lambda_mock = aws
    sqs.send_message(QueueUrl=extract_queue_url, MessageBody=_lambda_destination_failure_message(ORIGINAL_S3_EVENT))

    result = handler({"target": "extract"}, None)

    assert result["extract_replayed"] == 1
    lambda_mock.invoke.assert_called_once_with(
        FunctionName="ExtractLambda",
        InvocationType="Event",
        Payload=json.dumps(ORIGINAL_S3_EVENT).encode("utf-8"),
    )
    remaining = sqs.receive_message(QueueUrl=extract_queue_url, WaitTimeSeconds=1)
    assert "Messages" not in remaining


def test_redrive_extract_respects_max_messages(aws):
    sqs, extract_queue_url, _embed_url, _table, lambda_mock = aws
    for i in range(3):
        event = {**ORIGINAL_S3_EVENT, "id": i}
        sqs.send_message(QueueUrl=extract_queue_url, MessageBody=_lambda_destination_failure_message(event))

    result = handler({"target": "extract", "max_messages": 2}, None)

    assert result["extract_replayed"] == 2
    assert lambda_mock.invoke.call_count == 2


def test_redrive_embed_drains_metadata_and_only_replays_missing_embeddings(aws):
    sqs, _extract_url, embed_queue_url, table, lambda_mock = aws
    sqs.send_message(QueueUrl=embed_queue_url, MessageBody=json.dumps({"DDBStreamBatchInfo": {"shardId": "x"}}))

    table.put_item(Item={"posting_id": "missing-embedding", "source": "remoteok"})
    table.put_item(Item={"posting_id": "has-embedding", "source": "remoteok", "embedding": Binary(b"\x00\x01")})

    result = handler({"target": "embed"}, None)

    assert result["embed"]["drained"] == 1
    assert result["embed"]["replayed"] == 1
    assert lambda_mock.invoke.call_count == 1

    call_kwargs = lambda_mock.invoke.call_args.kwargs
    assert call_kwargs["FunctionName"] == "EmbedLambda"
    payload = json.loads(call_kwargs["Payload"])
    record = payload["Records"][0]
    assert record["eventName"] == "INSERT"
    assert record["dynamodb"]["NewImage"]["posting_id"]["S"] == "missing-embedding"

    remaining = sqs.receive_message(QueueUrl=embed_queue_url, WaitTimeSeconds=1)
    assert "Messages" not in remaining


def test_handler_defaults_to_both_targets(aws):
    sqs, extract_queue_url, _embed_queue_url, table, lambda_mock = aws
    sqs.send_message(QueueUrl=extract_queue_url, MessageBody=_lambda_destination_failure_message(ORIGINAL_S3_EVENT))
    table.put_item(Item={"posting_id": "p1", "source": "remoteok"})

    result = handler({}, None)

    assert "extract_replayed" in result
    assert "embed" in result
    assert lambda_mock.invoke.call_count == 2


def test_redrive_embed_does_nothing_when_nothing_is_missing(aws):
    _sqs, _extract_url, _embed_url, table, lambda_mock = aws
    table.put_item(Item={"posting_id": "already-embedded", "embedding": Binary(b"\x00")})

    result = handler({"target": "embed"}, None)

    assert result["embed"] == {"drained": 0, "replayed": 0}
    lambda_mock.invoke.assert_not_called()
