import json
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from lambdas.fetch.handler import handler
from lambdas.fetch.sources import RawPosting

BUCKET = "jobpulse-raw-postings-test"


@pytest.fixture
def raw_bucket():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield s3


def _fake_postings():
    return [
        RawPosting(
            source="remoteok",
            source_url="https://example.com/job/1",
            title="Backend Engineer",
            description_text="We need a Python developer.",
            raw_payload={"id": "1"},
        ),
        RawPosting(
            source="remoteok",
            source_url="https://example.com/job/2",
            title="Data Scientist",
            description_text="Build ML models.",
            raw_payload={"id": "2"},
        ),
    ]


def test_handler_writes_one_object_per_posting(raw_bucket, monkeypatch):
    monkeypatch.setenv("RAW_BUCKET_NAME", BUCKET)
    monkeypatch.setenv("JOB_SOURCE", "remoteok")

    with patch(
        "lambdas.fetch.handler.SOURCES",
        {"remoteok": type("S", (), {"fetch": staticmethod(_fake_postings)})()},
    ):
        result = handler({}, None)

    assert result == {"source": "remoteok", "fetched": 2, "written": 2}

    objects = raw_bucket.list_objects_v2(Bucket=BUCKET)["Contents"]
    assert len(objects) == 2
    assert all(key["Key"].startswith("raw/remoteok/") for key in objects)
    assert all(key["Key"].endswith(".json") for key in objects)


def test_handler_writes_untouched_raw_payload_as_audit_trail(raw_bucket, monkeypatch):
    monkeypatch.setenv("RAW_BUCKET_NAME", BUCKET)

    with patch(
        "lambdas.fetch.handler.SOURCES",
        {"remoteok": type("S", (), {"fetch": staticmethod(_fake_postings)})()},
    ):
        handler({}, None)

    key = raw_bucket.list_objects_v2(Bucket=BUCKET)["Contents"][0]["Key"]
    body = json.loads(raw_bucket.get_object(Bucket=BUCKET, Key=key)["Body"].read())

    assert body["raw_payload"] in ({"id": "1"}, {"id": "2"})
    assert body["source"] == "remoteok"
    assert "posting_hash" in body and "ingested_at" in body


def test_handler_defaults_to_remoteok_source(raw_bucket, monkeypatch):
    monkeypatch.setenv("RAW_BUCKET_NAME", BUCKET)
    monkeypatch.delenv("JOB_SOURCE", raising=False)

    with patch(
        "lambdas.fetch.handler.SOURCES",
        {"remoteok": type("S", (), {"fetch": staticmethod(_fake_postings)})()},
    ):
        result = handler({}, None)

    assert result["source"] == "remoteok"
