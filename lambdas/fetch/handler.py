"""Ingestion Lambda (PLAN.md Phase 2): pulls postings from a JobSource and writes each
one, untouched, to the S3 raw-landing-zone as an audit trail. Triggered on a schedule by
an EventBridge rule (see infra/jobpulse_infra/jobpulse_stack.py).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3

from common.logging_utils import get_logger, log_event
from common.metrics import emit_metric
from common.storage_keys import compute_posting_hash, raw_posting_key

try:
    # Local/package-style import, used by pytest (lambdas.fetch.handler).
    from lambdas.fetch.sources import JobSource, RemoteOKSource
except ImportError:
    # AWS Lambda flattens lambdas/fetch/* into /var/task, so sources.py is a top-level
    # sibling module there, not part of a "lambdas.fetch" package.
    from sources import JobSource, RemoteOKSource  # type: ignore[no-redef]

logger = get_logger(__name__)

SOURCES: dict[str, JobSource] = {
    "remoteok": RemoteOKSource(),
}


def handler(event, context):
    bucket = os.environ["RAW_BUCKET_NAME"]
    source_name = os.environ.get("JOB_SOURCE", "remoteok")
    source = SOURCES[source_name]

    s3 = boto3.client("s3")
    postings = source.fetch()
    today = datetime.now(timezone.utc).date()

    written = 0
    for posting in postings:
        posting_hash = compute_posting_hash(
            posting.source, posting.source_url, posting.description_text
        )
        key = raw_posting_key(posting.source, today, posting_hash)
        body = {
            "source": posting.source,
            "source_url": posting.source_url,
            "posting_hash": posting_hash,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "title": posting.title,
            "description_text": posting.description_text,
            "raw_payload": posting.raw_payload,
        }
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
        )
        written += 1
        log_event(logger, "wrote raw posting", posting_id=posting_hash, source=posting.source, key=key)

    result = {"source": source_name, "fetched": len(postings), "written": written}
    log_event(logger, "fetch complete", **result)
    emit_metric("PostingsIngested", written, dimensions={"Source": source_name})
    return result
