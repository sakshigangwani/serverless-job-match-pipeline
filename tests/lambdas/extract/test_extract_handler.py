import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from common.dynamodb import PARTITION_KEY, TABLE_NAME
from lambdas.extract.handler import handler

BUCKET = "jobpulse-raw-postings-test"
RAW_KEY = "raw/remoteok/2026-09-06/deadbeef.json"

RAW_ENVELOPE = {
    "source": "remoteok",
    "source_url": "https://example.com/job/1",
    "posting_hash": "deadbeef",
    "ingested_at": "2026-09-06T12:00:00+00:00",
    "title": "Backend Engineer",
    "description_text": "We need a Python developer with AWS experience. $120k-$150k. Remote OK.",
    "raw_payload": {"id": "1"},
}

VALID_EXTRACTION = {
    "title": "Backend Engineer",
    "company": "Acme Corp",
    "comp_min": 120000,
    "comp_max": 150000,
    "seniority": "mid",
    "required_skills": ["Python", "AWS"],
    "remote_status": "remote",
    "visa_status": "unknown",
}


def _bedrock_text_response(text: str) -> dict:
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"content": [{"type": "text", "text": text}]}
    ).encode("utf-8")
    return {"body": body}


def _s3_event(bucket: str, key: str) -> dict:
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


@contextmanager
def _mocked_aws_with_bedrock(bedrock_runtime_mock, monkeypatch):
    """Real (moto-backed) S3 + DynamoDB, but a fully controlled bedrock-runtime mock —
    moto's AWS coverage doesn't need to include Bedrock for these tests.
    """
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    original_client = boto3.client

    def fake_client(service_name, *args, **kwargs):
        if service_name == "bedrock-runtime":
            return bedrock_runtime_mock
        return original_client(service_name, *args, **kwargs)

    with mock_aws():
        s3 = original_client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.put_object(Bucket=BUCKET, Key=RAW_KEY, Body=json.dumps(RAW_ENVELOPE))

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": PARTITION_KEY, "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        monkeypatch.setenv("POSTINGS_TABLE_NAME", TABLE_NAME)
        table = dynamodb.Table(TABLE_NAME)

        with patch("boto3.client", side_effect=fake_client):
            yield table


def test_handler_writes_structured_posting_on_first_valid_response(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet")
    bedrock = MagicMock()
    bedrock.invoke_model.return_value = _bedrock_text_response(json.dumps(VALID_EXTRACTION))

    with _mocked_aws_with_bedrock(bedrock, monkeypatch) as table:
        result = handler(_s3_event(BUCKET, RAW_KEY), None)

        assert result["processed"] == 1
        assert bedrock.invoke_model.call_count == 1
        assert result["posting_ids"] == ["deadbeef"]

        item = table.get_item(Key={"posting_id": "deadbeef"})["Item"]
        assert item["posting_id"] == "deadbeef"
        assert item["company"] == "Acme Corp"
        assert item["comp_min"] == 120000
        assert item["remote_status"] == "remote"
        assert item["required_skills"] == ["Python", "AWS"]
        assert item["gsi_pk"] == "POSTING"
        # Omitted, not NULL — a NULL-typed `score` on a table with `score` as a GSI
        # sort key would make DynamoDB reject the write (see common/dynamodb.py).
        assert "score" not in item


def test_handler_retries_once_on_malformed_json_then_succeeds(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet")
    bedrock = MagicMock()
    bedrock.invoke_model.side_effect = [
        _bedrock_text_response("not valid json"),
        _bedrock_text_response(json.dumps(VALID_EXTRACTION)),
    ]

    with _mocked_aws_with_bedrock(bedrock, monkeypatch) as table:
        result = handler(_s3_event(BUCKET, RAW_KEY), None)

        assert bedrock.invoke_model.call_count == 2
        assert result["processed"] == 1

        item = table.get_item(Key={"posting_id": "deadbeef"})["Item"]
        assert item["company"] == "Acme Corp"


def test_handler_retries_once_on_schema_violation_then_succeeds(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet")
    invalid_comp_range = {**VALID_EXTRACTION, "comp_min": 200000, "comp_max": 100000}
    bedrock = MagicMock()
    bedrock.invoke_model.side_effect = [
        _bedrock_text_response(json.dumps(invalid_comp_range)),
        _bedrock_text_response(json.dumps(VALID_EXTRACTION)),
    ]

    with _mocked_aws_with_bedrock(bedrock, monkeypatch):
        result = handler(_s3_event(BUCKET, RAW_KEY), None)

    assert bedrock.invoke_model.call_count == 2
    assert result["processed"] == 1


def test_handler_raises_after_exhausting_retries_on_persistent_bad_json(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet")
    bedrock = MagicMock()
    bedrock.invoke_model.return_value = _bedrock_text_response("still not json")

    with _mocked_aws_with_bedrock(bedrock, monkeypatch), pytest.raises(
        Exception, match="extraction failed after 2 attempts"
    ):
        handler(_s3_event(BUCKET, RAW_KEY), None)

    assert bedrock.invoke_model.call_count == 2
