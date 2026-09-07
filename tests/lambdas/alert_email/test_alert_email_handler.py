import json

import boto3
import pytest
from moto import mock_aws

from lambdas.alert_email.handler import _format_email_body, handler

SENDER = "alerts@example.com"
RECIPIENT = "candidate@example.com"

ALERT = {
    "posting_id": "abc123",
    "title": "Backend Engineer",
    "company": "Acme Corp",
    "score": 0.93,
    "source_url": "https://example.com/job/abc123",
}


def _sns_event(alert: dict) -> dict:
    return {"Records": [{"Sns": {"Message": json.dumps(alert)}}]}


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("SENDER_EMAIL", SENDER)
    monkeypatch.setenv("RECIPIENT_EMAIL", RECIPIENT)

    with mock_aws():
        ses = boto3.client("ses", region_name="us-east-1")
        ses.verify_email_identity(EmailAddress=SENDER)
        yield ses


def test_sends_one_email_per_record(aws):
    result = handler(_sns_event(ALERT), None)

    assert result == {"emails_sent": 1, "posting_ids": ["abc123"]}


def test_email_body_includes_posting_details():
    body = _format_email_body(ALERT)

    assert "Backend Engineer" in body
    assert "Acme Corp" in body
    assert "0.930" in body
    assert "https://example.com/job/abc123" in body


def test_missing_title_and_company_render_without_crashing(aws):
    alert = {**ALERT, "title": None, "company": None}

    result = handler(_sns_event(alert), None)

    assert result["emails_sent"] == 1


def test_multiple_records_send_multiple_emails(aws):
    event = {
        "Records": [
            {"Sns": {"Message": json.dumps(ALERT)}},
            {"Sns": {"Message": json.dumps({**ALERT, "posting_id": "def456"})}},
        ]
    }

    result = handler(event, None)

    assert result["emails_sent"] == 2
    assert result["posting_ids"] == ["abc123", "def456"]
