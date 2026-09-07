import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from boto3.dynamodb.types import TypeSerializer
from moto import mock_aws

from common.dynamodb import (
    PARTITION_KEY,
    TABLE_NAME,
    to_dynamodb_item,
    unpack_embedding,
)
from common.models import CandidateProfile, Posting
from lambdas.embed.handler import handler

BUCKET = "jobpulse-raw-postings-test"
PROFILE_PATH = "common/candidate_profile.example.json"

PROFILE = CandidateProfile.from_file(PROFILE_PATH)
SECTION_TEXTS = PROFILE.section_texts()

# Deterministic 2-D vectors: resume matches posting-A exactly (similarity 1.0) and is
# orthogonal to posting-B (similarity 0.0), so exact expected values are known upfront.
RESUME_VECTOR = [1.0, 0.0]
SKILLS_VECTOR = [0.0, 1.0]
EXPERIENCE_VECTOR = [1.0, 1.0]
PROJECTS_VECTOR = [0.5, 0.5]
POSTING_A_VECTOR = [1.0, 0.0]  # identical to resume
POSTING_B_VECTOR = [0.0, 1.0]  # orthogonal to resume

TEXT_TO_VECTOR = {
    SECTION_TEXTS["resume"]: RESUME_VECTOR,
    SECTION_TEXTS["skills"]: SKILLS_VECTOR,
    SECTION_TEXTS["experience"]: EXPERIENCE_VECTOR,
    SECTION_TEXTS["projects"]: PROJECTS_VECTOR,
    "Posting A description": POSTING_A_VECTOR,
    "Posting B description": POSTING_B_VECTOR,
}

_serializer = TypeSerializer()


def _posting(posting_hash: str, description_text: str) -> Posting:
    return Posting(
        posting_id=posting_hash,
        source="remoteok",
        source_url=f"https://example.com/job/{posting_hash}",
        posting_hash=posting_hash,
        ingested_at="2026-09-06T12:00:00+00:00",
        description_text=description_text,
    )


def _insert_stream_event(posting: Posting) -> dict:
    """Build a DynamoDB Streams INSERT event carrying the same item shape extraction
    would have just put_item'd, serialized via the real low-level type serializer so
    this matches what an actual stream record looks like (not a hand-rolled shortcut).
    """
    item = to_dynamodb_item(posting)
    new_image = {k: _serializer.serialize(v) for k, v in item.items()}
    return {
        "Records": [
            {
                "eventName": "INSERT",
                "dynamodb": {"NewImage": new_image},
            }
        ]
    }


def _fake_bedrock_runtime() -> MagicMock:
    runtime = MagicMock()

    def _invoke_model(**kwargs):
        text = json.loads(kwargs["body"])["inputText"]
        vector = TEXT_TO_VECTOR[text]
        body = MagicMock()
        body.read.return_value = json.dumps({"embedding": vector}).encode("utf-8")
        return {"body": body}

    runtime.invoke_model.side_effect = _invoke_model
    return runtime


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", PROFILE_PATH)
    monkeypatch.setenv("RAW_BUCKET_NAME", BUCKET)
    monkeypatch.setenv("POSTINGS_TABLE_NAME", TABLE_NAME)

    bedrock = _fake_bedrock_runtime()
    original_client = boto3.client

    def fake_client(service_name, *args, **kwargs):
        if service_name == "bedrock-runtime":
            return bedrock
        return original_client(service_name, *args, **kwargs)

    with mock_aws():
        s3 = original_client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": PARTITION_KEY, "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table = dynamodb.Table(TABLE_NAME)

        with patch("boto3.client", side_effect=fake_client):
            yield s3, table, bedrock


def test_first_posting_computes_and_caches_candidate_embeddings(aws):
    s3, table, bedrock = aws
    posting = _posting("aaa111", "Posting A description")
    table.put_item(Item=to_dynamodb_item(posting))

    result = handler(_insert_stream_event(posting), None)

    # 4 candidate sections + 1 posting = 5 Bedrock calls on a cold cache.
    assert bedrock.invoke_model.call_count == 5
    assert result == {"processed": 1, "posting_ids": ["aaa111"]}

    cached = json.loads(s3.get_object(Bucket=BUCKET, Key="candidate/embeddings.json")["Body"].read())
    assert cached["candidate_id"] == PROFILE.candidate_id
    assert cached["sections"]["resume"] == RESUME_VECTOR

    item = table.get_item(Key={"posting_id": "aaa111"})["Item"]
    assert unpack_embedding(item["embedding"]) == pytest.approx(POSTING_A_VECTOR)
    assert float(item["embedding_similarity"]) == pytest.approx(1.0)


def test_second_posting_reuses_cached_candidate_embeddings(aws):
    _s3, table, bedrock = aws
    posting_a = _posting("aaa111", "Posting A description")
    table.put_item(Item=to_dynamodb_item(posting_a))
    handler(_insert_stream_event(posting_a), None)
    assert bedrock.invoke_model.call_count == 5

    posting_b = _posting("bbb222", "Posting B description")
    table.put_item(Item=to_dynamodb_item(posting_b))
    handler(_insert_stream_event(posting_b), None)

    # No re-embedding of the candidate profile: only 1 more call for posting B.
    assert bedrock.invoke_model.call_count == 6

    item = table.get_item(Key={"posting_id": "bbb222"})["Item"]
    assert unpack_embedding(item["embedding"]) == pytest.approx(POSTING_B_VECTOR)
    assert float(item["embedding_similarity"]) == pytest.approx(0.0)


def test_extraction_fields_are_untouched_by_the_embed_update(aws):
    _s3, table, _bedrock = aws
    posting = _posting("aaa111", "Posting A description")
    posting.company = "Acme Corp"
    table.put_item(Item=to_dynamodb_item(posting))

    handler(_insert_stream_event(posting), None)

    item = table.get_item(Key={"posting_id": "aaa111"})["Item"]
    assert item["company"] == "Acme Corp"
    assert item["source"] == "remoteok"


def test_non_insert_records_are_ignored(aws):
    _s3, _table, bedrock = aws
    posting = _posting("aaa111", "Posting A description")
    event = _insert_stream_event(posting)
    event["Records"][0]["eventName"] = "MODIFY"

    result = handler(event, None)

    assert result == {"processed": 0, "posting_ids": []}
    assert bedrock.invoke_model.call_count == 0
