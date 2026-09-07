"""Alert email Lambda (PLAN.md Phase 7.2): subscribed to the alerts SNS topic, sends the
actual email via SES. Not pre-scaffolded in Phase 0's lambda list — added here because
SNS has no native SES subscription protocol; a Lambda subscriber is the only way to
realize "SNS fans out to SES for email alerts" (spec section 2.1). Keeping this as its
own Lambda (rather than having the threshold Lambda call SES directly) is what makes the
"later, an additional subscriber" extensibility spec section 5.3 describes free: a
second subscriber (e.g. a future Slack/Discord webhook) attaches to the same SNS topic
without touching this Lambda or the threshold Lambda at all.
"""
from __future__ import annotations

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _format_email_body(alert: dict) -> str:
    return (
        f"A new posting cleared your fit threshold.\n\n"
        f"Title: {alert.get('title') or '(untitled)'}\n"
        f"Company: {alert.get('company') or 'unknown'}\n"
        f"Fit score: {alert['score']:.3f}\n"
        f"Link: {alert['source_url']}\n"
    )


def _process_record(ses, sender: str, recipient: str, record: dict) -> str:
    alert = json.loads(record["Sns"]["Message"])
    ses.send_email(
        Source=sender,
        Destination={"ToAddresses": [recipient]},
        Message={
            "Subject": {"Data": "JobPulse: high-fit posting found"},
            "Body": {"Text": {"Data": _format_email_body(alert)}},
        },
    )
    return alert["posting_id"]


def handler(event, context):
    sender = os.environ["SENDER_EMAIL"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    ses = boto3.client("ses")

    sent = [_process_record(ses, sender, recipient, record) for record in event.get("Records", [])]

    result = {"emails_sent": len(sent), "posting_ids": sent}
    logger.info(json.dumps(result))
    return result
