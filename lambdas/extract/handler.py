"""Extraction Lambda (PLAN.md Phase 3): triggered by S3 ObjectCreated events under
raw/, calls Bedrock (Claude) to extract structured fields from a posting's raw text,
validates the result against common.models.Posting, and writes the structured record to
S3 under structured/ (an interim store — Phase 5 replaces this with DynamoDB).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
from datetime import datetime

import boto3
from pydantic import ValidationError

from common.models import Posting
from common.storage_keys import structured_posting_key

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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_ATTEMPTS = 2


def _extract_posting(bedrock_runtime, model_id: str, raw: dict) -> Posting:
    """Call the LLM and validate its output, retrying once on bad JSON or a schema
    violation (e.g. an out-of-range comp value) before giving up.
    """
    prompt = build_extraction_prompt(raw["title"], raw["description_text"])
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            text = invoke_claude(bedrock_runtime, model_id, prompt)
            fields = parse_llm_json(text)
            return Posting(
                posting_id=raw["posting_hash"],
                source=raw["source"],
                source_url=raw["source_url"],
                posting_hash=raw["posting_hash"],
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
            logger.warning("extraction attempt %s failed: %s", attempt, exc)
            prompt = (
                build_extraction_prompt(raw["title"], raw["description_text"])
                + f"\n\nYour previous response was invalid: {exc}\n"
                + "Return ONLY a corrected JSON object, with no extra text or markdown."
            )

    raise ExtractionError(f"extraction failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _process_record(s3, bedrock_runtime, model_id: str, bucket: str, key: str) -> str:
    raw = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    posting = _extract_posting(bedrock_runtime, model_id, raw)

    out_key = structured_posting_key(
        posting.source, posting.ingested_at.date(), posting.posting_hash
    )
    s3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=posting.model_dump_json().encode("utf-8"),
        ContentType="application/json",
    )
    return out_key


def handler(event, context):
    model_id = os.environ["BEDROCK_MODEL_ID"]

    s3 = boto3.client("s3")
    bedrock_runtime = boto3.client("bedrock-runtime")

    written = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        written.append(_process_record(s3, bedrock_runtime, model_id, bucket, key))

    result = {"processed": len(written), "written_keys": written}
    logger.info(json.dumps(result))
    return result
