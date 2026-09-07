"""Query API Lambda (PLAN.md Phase 8): serves ranked, filterable postings on demand
behind API Gateway. Reads the ScoreIndex GSI (score descending) rather than scanning
the whole table — postings without a score yet (Phase 5's sparse-index behavior) are
never in this GSI at all, so every result here has already cleared extraction+embedding
+scoring.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal, InvalidOperation

import boto3
from boto3.dynamodb.conditions import Attr, Key

from common.dynamodb import (
    GSI_NAME,
    GSI_PARTITION_KEY,
    GSI_PARTITION_VALUE,
    GSI_SORT_KEY,
    from_dynamodb_item,
)
from common.logging_utils import get_logger, log_event

logger = get_logger(__name__)

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class InvalidQueryParams(ValueError):
    pass


def _parse_query_params(params: dict) -> dict:
    filters: dict = {}

    if params.get("min_score") is not None:
        try:
            filters["min_score"] = Decimal(params["min_score"])
        except InvalidOperation as exc:
            raise InvalidQueryParams(f"min_score must be a number, got {params['min_score']!r}") from exc

    if params.get("remote_status"):
        filters["remote_status"] = params["remote_status"]

    if params.get("seniority"):
        filters["seniority"] = params["seniority"]

    try:
        limit = int(params.get("limit", DEFAULT_LIMIT))
    except ValueError as exc:
        raise InvalidQueryParams(f"limit must be an integer, got {params['limit']!r}") from exc
    filters["limit"] = max(1, min(limit, MAX_LIMIT))

    return filters


def _query_ranked_postings(table, filters: dict) -> list[dict]:
    key_condition = Key(GSI_PARTITION_KEY).eq(GSI_PARTITION_VALUE)
    if "min_score" in filters:
        key_condition &= Key(GSI_SORT_KEY).gte(filters["min_score"])

    filter_expression = None
    if "remote_status" in filters:
        filter_expression = Attr("remote_status").eq(filters["remote_status"])
    if "seniority" in filters:
        condition = Attr("seniority").eq(filters["seniority"])
        filter_expression = condition if filter_expression is None else filter_expression & condition

    limit = filters["limit"]
    results: list[dict] = []
    exclusive_start_key = None
    # FilterExpression is applied after DynamoDB reads a page, not before Limit is
    # counted — a single Query call could return fewer than `limit` filtered results
    # even though more matches exist further in the index. Paging with
    # LastEvaluatedKey until `limit` is reached (or the index is exhausted) is what
    # makes filtered results behave like an actual limit, not a per-page cap.
    while len(results) < limit:
        query_kwargs = {
            "IndexName": GSI_NAME,
            "KeyConditionExpression": key_condition,
            "ScanIndexForward": False,
            "Limit": limit,
        }
        if filter_expression is not None:
            query_kwargs["FilterExpression"] = filter_expression
        if exclusive_start_key:
            query_kwargs["ExclusiveStartKey"] = exclusive_start_key

        response = table.query(**query_kwargs)
        results.extend(response.get("Items", []))
        exclusive_start_key = response.get("LastEvaluatedKey")
        if not exclusive_start_key:
            break

    return results[:limit]


def _error_response(status_code: int, message: str) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }


def handler(event, context):
    table_name = os.environ["POSTINGS_TABLE_NAME"]
    table = boto3.resource("dynamodb").Table(table_name)

    params = event.get("queryStringParameters") or {}
    try:
        filters = _parse_query_params(params)
    except InvalidQueryParams as exc:
        return _error_response(400, str(exc))

    items = _query_ranked_postings(table, filters)
    postings = [json.loads(from_dynamodb_item(item).model_dump_json()) for item in items]

    result = {"postings": postings, "count": len(postings)}
    log_event(logger, "query served", count=len(postings), filters=params)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result),
    }
