"""Structured JSON logging with a posting-level correlation id (PLAN.md Phase 9.1).

Every Lambda in the pipeline handles postings one at a time (or a small batch), but
each posting passes through several *separate* Lambda invocations (fetch -> extract ->
embed -> threshold -> alert_email) with no shared request ID linking them the way a
single synchronous call chain would have. Logging `posting_id` on every structured log
line is what makes it possible to reconstruct one posting's whole journey after the
fact — a CloudWatch Logs Insights query filtered to one `posting_id` across every
Lambda's log group returns every step that posting went through, in order.
"""
from __future__ import annotations

import json
import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, message: str, *, posting_id: str | None = None, **fields) -> None:
    """Emit one structured JSON log line. `posting_id`, when known, is always the first
    field after `message` so it's easy to spot when scanning raw CloudWatch Logs output.
    """
    payload: dict = {"message": message}
    if posting_id is not None:
        payload["posting_id"] = posting_id
    payload.update(fields)
    logger.info(json.dumps(payload))
