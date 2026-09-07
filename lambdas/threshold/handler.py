"""Scoring + threshold + alerting Lambda (PLAN.md Phase 7): triggered by a DynamoDB
Streams MODIFY event once the embed Lambda has set a posting's embedding (the CDK event
source filters to exactly that transition — see infra/jobpulse_infra/jobpulse_stack.py).
Scores the posting with the trained re-ranker, writes the final score + alert_sent back
to DynamoDB, and publishes to SNS (fanning out to SES for email) when the score clears
the fit threshold.

update_item here produces a further MODIFY record with a non-null `score`, which the
same event source filter (score still NULL) excludes — so, like the embed Lambda before
it, this can't re-trigger itself.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import boto3
from boto3.dynamodb.types import TypeDeserializer

from common.dynamodb import from_dynamodb_item
from common.logging_utils import get_logger, log_event
from common.metrics import emit_metric
from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile
from ml.features import build_feature_vector

try:
    # Local/package-style import, used by pytest (lambdas.threshold.handler).
    from lambdas.threshold.reranker import load_model, predict_proba
except ImportError:
    # AWS Lambda flattens lambdas/threshold/* into /var/task, so this is a top-level
    # sibling module there, not part of a "lambdas.threshold" package.
    from reranker import load_model, predict_proba  # type: ignore[no-redef]

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.json"
DEFAULT_FIT_THRESHOLD = 0.7

_deserializer = TypeDeserializer()
_model_cache: dict | None = None


def _get_model() -> dict:
    """Loaded once per warm Lambda execution environment, not on every invocation."""
    global _model_cache
    if _model_cache is None:
        model_path = os.environ.get("RERANKER_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        _model_cache = load_model(model_path)
    return _model_cache


def _process_record(table, sns, topic_arn: str, candidate: CandidateProfile, fit_threshold: float, stream_record: dict) -> dict:
    new_image = {
        k: _deserializer.deserialize(v) for k, v in stream_record["dynamodb"]["NewImage"].items()
    }
    posting = from_dynamodb_item(new_image)

    feature_vector = build_feature_vector(posting, candidate)
    score = predict_proba(_get_model(), feature_vector)
    alert_sent = score >= fit_threshold

    if alert_sent:
        sns.publish(
            TopicArn=topic_arn,
            Subject="JobPulse: high-fit posting found",
            Message=json.dumps(
                {
                    "posting_id": posting.posting_id,
                    "title": posting.title,
                    "company": posting.company,
                    "score": score,
                    "source_url": posting.source_url,
                }
            ),
        )

    table.update_item(
        Key={"posting_id": posting.posting_id},
        UpdateExpression="SET score = :score, alert_sent = :alert_sent",
        ExpressionAttributeValues={
            ":score": Decimal(str(score)),
            ":alert_sent": alert_sent,
        },
    )
    log_event(logger, "scored posting", posting_id=posting.posting_id, score=score, alert_sent=alert_sent)
    emit_metric("PostingScore", score)
    emit_metric("AlertsSent", 1 if alert_sent else 0)
    return {"posting_id": posting.posting_id, "score": score, "alert_sent": alert_sent}


def handler(event, context):
    table_name = os.environ["POSTINGS_TABLE_NAME"]
    topic_arn = os.environ["ALERTS_TOPIC_ARN"]
    profile_path = os.environ.get("CANDIDATE_PROFILE_PATH", str(DEFAULT_CANDIDATE_PROFILE_PATH))
    fit_threshold = float(os.environ.get("FIT_THRESHOLD", DEFAULT_FIT_THRESHOLD))

    candidate = CandidateProfile.from_file(profile_path)
    table = boto3.resource("dynamodb").Table(table_name)
    sns = boto3.client("sns")

    processed = []
    for record in event.get("Records", []):
        if record.get("eventName") != "MODIFY":
            # Defense in depth: the CDK event source is already filtered to postings
            # whose score is still NULL and whose embedding just appeared (see
            # jobpulse_stack.py), so this Lambda's own writes never reach here in
            # production.
            continue
        processed.append(_process_record(table, sns, topic_arn, candidate, fit_threshold, record))

    result = {"processed": len(processed), "results": processed}
    log_event(logger, "threshold batch complete", processed=len(processed))
    return result
