"""Extraction Lambda (PLAN.md Phase 3, persistence updated in Phase 5): triggered by S3
ObjectCreated events under raw/, calls Bedrock (Claude) to extract structured fields from
a posting's raw text, validates the result against common.models.Posting, and writes the
structured record to DynamoDB.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime

import boto3
from pydantic import ValidationError

from common.dynamodb import to_dynamodb_item
from common.logging_utils import get_logger, log_event
from common.metrics import emit_metric
from common.models import Posting

try:
    # Local/package-style import, used by pytest (lambdas.extract.handler).
    from lambdas.extract.bedrock_client import (
        ExtractionError,
        invoke_claude,
        parse_llm_json,
    )
    from lambdas.extract.prompts import build_extraction_prompt
except ImportError:
    # AWS Lambda flattens lambdas/extract/* into /var/task, so these are top-level
    # sibling modules there, not part of a "lambdas.extract" package.
    from bedrock_client import (  # type: ignore[no-redef]
        ExtractionError,
        invoke_claude,
        parse_llm_json,
    )
    from prompts import build_extraction_prompt  # type: ignore[no-redef]

logger = get_logger(__name__)

MAX_ATTEMPTS = 2


def _extract_posting(bedrock_runtime, model_id: str, raw: dict) -> Posting:
    """Call the LLM and validate its output, retrying once on bad JSON or a schema
    violation (e.g. an out-of-range comp value) before giving up.
    """
    posting_id = raw["posting_hash"]
    prompt = build_extraction_prompt(raw["title"], raw["description_text"])
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            text = invoke_claude(bedrock_runtime, model_id, prompt)
            fields = parse_llm_json(text)
            return Posting(
                posting_id=posting_id,
                source=raw["source"],
                source_url=raw["source_url"],
                posting_hash=posting_id,
                ingested_at=datetime.fromisoformat(raw["ingested_at"]),
                description_text=raw["description_text"],
                title=fields.get("title") or raw.get("title"),
                company=fields.get("company"),
                comp_min=fields.get("comp_min"),
                comp_max=fields.get("comp_max"),
                seniority=fields.get("seniority"),
                required_skills=fields.get("required_skills") or [],
                remote_status=fields.get("remote_status", "unknown"),
                visa_status=fields.get("visa_status", "unknown"),
            )
        except (ExtractionError, ValidationError) as exc:
            last_error = exc
            log_event(
                logger, "extraction attempt failed", posting_id=posting_id, attempt=attempt, error=str(exc)
            )
            prompt = (
                build_extraction_prompt(raw["title"], raw["description_text"])
                + f"\n\nYour previous response was invalid: {exc}\n"
                + "Return ONLY a corrected JSON object, with no extra text or markdown."
            )

    emit_metric("ExtractionFailures", 1)
    raise ExtractionError(f"extraction failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _process_record(s3, bedrock_runtime, table, model_id: str, bucket: str, key: str) -> str:
    raw = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    posting = _extract_posting(bedrock_runtime, model_id, raw)

    table.put_item(Item=to_dynamodb_item(posting))
    log_event(logger, "extracted and stored posting", posting_id=posting.posting_id)
    return posting.posting_id


def handler(event, context):
    model_id = os.environ["BEDROCK_MODEL_ID"]
    table_name = os.environ["POSTINGS_TABLE_NAME"]

    s3 = boto3.client("s3")
    bedrock_runtime = boto3.client("bedrock-runtime")
    table = boto3.resource("dynamodb").Table(table_name)

    written = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        written.append(_process_record(s3, bedrock_runtime, table, model_id, bucket, key))

    result = {"processed": len(written), "posting_ids": written}
    log_event(logger, "extraction batch complete", **result)
    return result
