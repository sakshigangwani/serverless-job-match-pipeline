"""Skill-gap analysis Lambda (PLAN.md Phase 14): a learning recommendation, not a match
score — it surfaces which skills most often separate the candidate from a "good fit"
posting.

Queries the same ScoreIndex GSI query_api (Phase 8) reads, filtered to postings scored
below the fit threshold (the same threshold the threshold Lambda, Phase 7, uses to decide
whether to alert), then counts how often each of those postings' required_skills is a
skill the candidate doesn't have — ranked most-frequent first. Served behind the same API
Gateway REST API as query_api (a `/skill-gap` sibling resource — see
infra/jobpulse_infra/jobpulse_stack.py), gated by the same API key.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

from common.dynamodb import (
    GSI_NAME,
    GSI_PARTITION_KEY,
    GSI_PARTITION_VALUE,
    GSI_SORT_KEY,
    from_dynamodb_item,
)
from common.logging_utils import get_logger, log_event
from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile, Posting

logger = get_logger(__name__)

DEFAULT_FIT_THRESHOLD = 0.7
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class InvalidQueryParams(ValueError):
    pass


def _query_low_scoring_postings(table, fit_threshold: float) -> list[Posting]:
    """All postings scored below fit_threshold, paged to completion. Unlike query_api's
    page-until-`limit` loop (Phase 8), skill-gap analysis needs the full low-scoring
    population to count accurately, not a capped sample of it.
    """
    key_condition = Key(GSI_PARTITION_KEY).eq(GSI_PARTITION_VALUE) & Key(GSI_SORT_KEY).lt(
        Decimal(str(fit_threshold))
    )
    items: list[dict] = []
    exclusive_start_key = None
    while True:
        query_kwargs = {"IndexName": GSI_NAME, "KeyConditionExpression": key_condition}
        if exclusive_start_key:
            query_kwargs["ExclusiveStartKey"] = exclusive_start_key
        response = table.query(**query_kwargs)
        items.extend(response.get("Items", []))
        exclusive_start_key = response.get("LastEvaluatedKey")
        if not exclusive_start_key:
            break
    return [from_dynamodb_item(item) for item in items]


def compute_skill_gaps(
    postings: list[Posting], candidate: CandidateProfile, limit: int = DEFAULT_LIMIT
) -> list[dict]:
    """Counts, across `postings`, how often each required skill the candidate lacks
    appears — ranked most-frequent first, alphabetically tie-broken for determinism.
    Case-insensitive comparison, same as ml/features.py's skill_overlap_count.
    """
    candidate_skills = {s.lower() for s in candidate.skills}
    counts: Counter[str] = Counter()
    original_casing: dict[str, str] = {}
    for posting in postings:
        for skill in posting.required_skills:
            key = skill.lower()
            if key in candidate_skills:
                continue
            counts[key] += 1
            original_casing.setdefault(key, skill)

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"skill": original_casing[key], "missing_count": count} for key, count in ranked[:limit]]


def _parse_limit(params: dict) -> int:
    try:
        limit = int(params.get("limit", DEFAULT_LIMIT))
    except ValueError as exc:
        raise InvalidQueryParams(f"limit must be an integer, got {params['limit']!r}") from exc
    return max(1, min(limit, MAX_LIMIT))


def _error_response(status_code: int, message: str) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }


def handler(event, context):
    table_name = os.environ["POSTINGS_TABLE_NAME"]
    profile_path = os.environ.get("CANDIDATE_PROFILE_PATH", str(DEFAULT_CANDIDATE_PROFILE_PATH))
    fit_threshold = float(os.environ.get("FIT_THRESHOLD", DEFAULT_FIT_THRESHOLD))

    params = (event or {}).get("queryStringParameters") or {}
    try:
        limit = _parse_limit(params)
    except InvalidQueryParams as exc:
        return _error_response(400, str(exc))

    candidate = CandidateProfile.from_file(profile_path)
    table = boto3.resource("dynamodb").Table(table_name)

    postings = _query_low_scoring_postings(table, fit_threshold)
    skill_gaps = compute_skill_gaps(postings, candidate, limit=limit)

    result = {"skill_gaps": skill_gaps, "analyzed_postings": len(postings)}
    log_event(logger, "skill gap analysis served", analyzed_postings=len(postings), gap_count=len(skill_gaps))
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result),
    }
