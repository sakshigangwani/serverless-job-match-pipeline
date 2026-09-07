"""DLQ redrive Lambda (PLAN.md Phase 12.4): manually invoked (not triggered
automatically) after investigating and fixing whatever caused extraction or embedding
failures, to replay them.

Extraction and embedding fail through two structurally different mechanisms, and their
DLQ messages carry different content as a result — this Lambda handles each accordingly
rather than treating "redrive" as one uniform operation:

- Extraction is triggered by an S3 event, an *asynchronous* Lambda invocation. Its DLQ
  (a Lambda Destinations on-failure target) carries a rich envelope including
  `requestPayload` — the exact original S3 event. Redriving it is a direct re-invoke of
  the extraction Lambda with that same event, which re-reads the same (still-present;
  S3 is our immutable audit trail) raw posting.
- Embedding is triggered by a DynamoDB Streams event source mapping, a fundamentally
  different, poll-based mechanism. Its DLQ (an event-source-mapping on-failure target)
  carries only *batch metadata* (shard id, sequence number range) — never the failed
  record's actual content, since a stream-sourced DLQ message can't replay stream data
  that may have already rolled off. The posting itself is untouched by embedding's
  failure (extraction already wrote it), so redriving means finding postings still
  missing an `embedding` attribute and synthesizing a fresh stream-event-shaped payload
  from their *current* DynamoDB item — not literally replaying the SQS message.
"""
from __future__ import annotations

import json
import os

import boto3
from boto3.dynamodb.conditions import Attr
from boto3.dynamodb.types import TypeSerializer

from common.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_serializer = TypeSerializer()

DEFAULT_MAX_MESSAGES = 10


def redrive_extract_dlq(sqs, lambda_client, queue_url: str, function_name: str, max_messages: int) -> int:
    replayed = 0
    while replayed < max_messages:
        response = sqs.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=min(10, max_messages - replayed), WaitTimeSeconds=1
        )
        messages = response.get("Messages", [])
        if not messages:
            break
        for message in messages:
            original_event = json.loads(message["Body"])["requestPayload"]
            lambda_client.invoke(
                FunctionName=function_name,
                InvocationType="Event",
                Payload=json.dumps(original_event).encode("utf-8"),
            )
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
            replayed += 1
            log_event(logger, "redrove extract DLQ message", message_id=message["MessageId"])
    return replayed


def _drain_embed_dlq_metadata(sqs, queue_url: str) -> int:
    """The embed DLQ's messages carry no item payload to replay (see module
    docstring) — draining them just clears the queue of metadata that's already been
    read and understood; the real reprocessing work happens via DynamoDB below.
    """
    drained = 0
    while True:
        response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=1)
        messages = response.get("Messages", [])
        if not messages:
            break
        for message in messages:
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
            drained += 1
    return drained


def _synthetic_insert_event(item: dict) -> dict:
    """Shapes a DynamoDB item exactly like the INSERT stream event embed_lambda's
    handler expects, so it can be re-invoked directly without any handler changes.
    """
    new_image = {k: _serializer.serialize(v) for k, v in item.items()}
    return {"Records": [{"eventName": "INSERT", "dynamodb": {"NewImage": new_image}}]}


def redrive_embed_dlq(sqs, lambda_client, table, queue_url: str, function_name: str, max_messages: int) -> dict:
    drained = _drain_embed_dlq_metadata(sqs, queue_url)
    if drained:
        log_event(logger, "drained embed DLQ batch-failure metadata", count=drained)

    response = table.scan(FilterExpression=Attr("embedding").not_exists(), Limit=max_messages)
    replayed = 0
    for item in response.get("Items", []):
        lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="Event",
            Payload=json.dumps(_synthetic_insert_event(item)).encode("utf-8"),
        )
        replayed += 1
        log_event(logger, "redrove posting missing embedding", posting_id=item.get("posting_id"))

    return {"drained": drained, "replayed": replayed}


def handler(event, context):
    target = event.get("target", "both")
    max_messages = int(event.get("max_messages", DEFAULT_MAX_MESSAGES))

    sqs = boto3.client("sqs")
    lambda_client = boto3.client("lambda")

    result: dict = {}
    if target in ("extract", "both"):
        result["extract_replayed"] = redrive_extract_dlq(
            sqs,
            lambda_client,
            queue_url=os.environ["EXTRACT_DLQ_URL"],
            function_name=os.environ["EXTRACT_FUNCTION_NAME"],
            max_messages=max_messages,
        )
    if target in ("embed", "both"):
        table = boto3.resource("dynamodb").Table(os.environ["POSTINGS_TABLE_NAME"])
        result["embed"] = redrive_embed_dlq(
            sqs,
            lambda_client,
            table,
            queue_url=os.environ["EMBED_DLQ_URL"],
            function_name=os.environ["EMBED_FUNCTION_NAME"],
            max_messages=max_messages,
        )

    log_event(logger, "redrive complete", **result)
    return result
