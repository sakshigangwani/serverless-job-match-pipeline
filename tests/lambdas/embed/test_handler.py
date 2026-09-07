import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

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


def _posting(posting_hash: str, description_text: str) -> Posting:
    return Posting(
        posting_id=posting_hash,
        source="remoteok",
        source_url=f"https://example.com/job/{posting_hash}",
        posting_hash=posting_hash,
        ingested_at="2026-09-06T12:00:00+00:00",
        description_text=description_text,
    )


def _structured_key(posting: Posting) -> str:
    return f"structured/{posting.source}/2026-09-06/{posting.posting_hash}.json"


def _s3_event(bucket: str, key: str) -> dict:
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


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
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", PROFILE_PATH)

    bedrock = _fake_bedrock_runtime()
    original_client = boto3.client

    def fake_client(service_name, *args, **kwargs):
        if service_name == "bedrock-runtime":
            return bedrock
        return original_client(service_name, *args, **kwargs)

    with mock_aws():
        s3 = original_client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        with patch("boto3.client", side_effect=fake_client):
            yield s3, bedrock


def test_first_posting_computes_and_caches_candidate_embeddings(aws):
    s3, bedrock = aws
    posting = _posting("aaa111", "Posting A description")
    s3.put_object(Bucket=BUCKET, Key=_structured_key(posting), Body=posting.model_dump_json())

    result = handler(_s3_event(BUCKET, _structured_key(posting)), None)

    # 4 candidate sections + 1 posting = 5 Bedrock calls on a cold cache.
    assert bedrock.invoke_model.call_count == 5

    cached = json.loads(s3.get_object(Bucket=BUCKET, Key="candidate/embeddings.json")["Body"].read())
    assert cached["candidate_id"] == PROFILE.candidate_id
    assert cached["sections"]["resume"] == RESUME_VECTOR

    out_key = result["written_keys"][0]
    assert out_key == "embeddings/remoteok/2026-09-06/aaa111.json"
    embedding_doc = json.loads(s3.get_object(Bucket=BUCKET, Key=out_key)["Body"].read())
    assert embedding_doc["embedding"] == POSTING_A_VECTOR
    assert embedding_doc["embedding_similarity"] == pytest.approx(1.0)


def test_second_posting_reuses_cached_candidate_embeddings(aws):
    s3, bedrock = aws
    posting_a = _posting("aaa111", "Posting A description")
    s3.put_object(Bucket=BUCKET, Key=_structured_key(posting_a), Body=posting_a.model_dump_json())
    handler(_s3_event(BUCKET, _structured_key(posting_a)), None)
    assert bedrock.invoke_model.call_count == 5

    posting_b = _posting("bbb222", "Posting B description")
    s3.put_object(Bucket=BUCKET, Key=_structured_key(posting_b), Body=posting_b.model_dump_json())
    result = handler(_s3_event(BUCKET, _structured_key(posting_b)), None)

    # No re-embedding of the candidate profile: only 1 more call for posting B.
    assert bedrock.invoke_model.call_count == 6

    out_key = result["written_keys"][0]
    embedding_doc = json.loads(s3.get_object(Bucket=BUCKET, Key=out_key)["Body"].read())
    assert embedding_doc["embedding"] == POSTING_B_VECTOR
    assert embedding_doc["embedding_similarity"] == pytest.approx(0.0)


def test_embed_does_not_write_back_to_the_structured_key(aws):
    s3, _bedrock = aws
    posting = _posting("aaa111", "Posting A description")
    structured_key = _structured_key(posting)
    original_body = posting.model_dump_json()
    s3.put_object(Bucket=BUCKET, Key=structured_key, Body=original_body)

    handler(_s3_event(BUCKET, structured_key), None)

    unchanged = s3.get_object(Bucket=BUCKET, Key=structured_key)["Body"].read().decode()
    assert unchanged == original_body
