"""Embedding Lambda (PLAN.md Phase 4, trigger + persistence updated in Phase 5):
triggered by a DynamoDB Streams INSERT event on the postings table (the CDK event source
is filtered to INSERT only — see infra/jobpulse_infra/jobpulse_stack.py), embeds each
new posting's description text and the candidate profile (once, cached in S3), and
merges the posting's packed embedding vector plus a cosine-similarity baseline into the
same DynamoDB item via update_item.

update_item (not put_item) produces a MODIFY stream record, not an INSERT one, so this
can never re-trigger itself even without the INSERT-only filter — belt and suspenders,
the same one-directional-flow principle used for the raw/ vs. structured/ split when
this Lambda was S3-triggered.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal

import boto3
from boto3.dynamodb.types import TypeDeserializer

from common.dynamodb import from_dynamodb_item, pack_embedding
from common.logging_utils import get_logger, log_event
from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile
from common.similarity import cosine_similarity
from common.storage_keys import CANDIDATE_EMBEDDINGS_KEY

try:
    # Local/package-style import, used by pytest (lambdas.embed.handler).
    from lambdas.embed.bedrock_embeddings import embed_text
except ImportError:
    # AWS Lambda flattens lambdas/embed/* into /var/task, so this is a top-level
    # sibling module there, not part of a "lambdas.embed" package.
    from bedrock_embeddings import embed_text  # type: ignore[no-redef]

logger = get_logger(__name__)

_deserializer = TypeDeserializer()


def _get_or_load_candidate_embeddings(
    s3, bedrock_runtime, model_id: str, bucket: str, profile_path: str
) -> dict:
    """Cache-aside: read the cached candidate embeddings if present, otherwise compute
    and cache them. Not race-free under concurrent cold-start invocations — a rare
    double-embed of the candidate profile is harmless (same fixed key, same content) and
    far cheaper than a distributed lock for a profile that changes on the order of
    "whenever a resume is updated," not per-posting.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=CANDIDATE_EMBEDDINGS_KEY)
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        profile = CandidateProfile.from_file(profile_path)
        sections = {
            name: embed_text(bedrock_runtime, model_id, text)
            for name, text in profile.section_texts().items()
            if text
        }
        payload = {
            "candidate_id": profile.candidate_id,
            "model_id": model_id,
            "sections": sections,
        }
        s3.put_object(
            Bucket=bucket,
            Key=CANDIDATE_EMBEDDINGS_KEY,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
        return payload


def _process_record(s3, bedrock_runtime, table, model_id: str, bucket: str, stream_record: dict, profile_path: str) -> str:
    new_image = {
        k: _deserializer.deserialize(v) for k, v in stream_record["dynamodb"]["NewImage"].items()
    }
    posting = from_dynamodb_item(new_image)

    candidate_embeddings = _get_or_load_candidate_embeddings(
        s3, bedrock_runtime, model_id, bucket, profile_path
    )
    posting_embedding = embed_text(bedrock_runtime, model_id, posting.description_text)

    resume_vector = candidate_embeddings["sections"].get("resume")
    similarity = cosine_similarity(posting_embedding, resume_vector) if resume_vector else None

    table.update_item(
        Key={"posting_id": posting.posting_id},
        UpdateExpression="SET embedding = :embedding, embedding_similarity = :similarity",
        ExpressionAttributeValues={
            ":embedding": pack_embedding(posting_embedding),
            # boto3's Table resource requires Decimal, not float, for DynamoDB Numbers
            # (common.dynamodb.to_dynamodb_item does the same conversion for `score`).
            ":similarity": Decimal(str(similarity)) if similarity is not None else None,
        },
    )
    log_event(logger, "embedded posting", posting_id=posting.posting_id, embedding_similarity=similarity)
    return posting.posting_id


def handler(event, context):
    model_id = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]
    table_name = os.environ["POSTINGS_TABLE_NAME"]
    bucket = os.environ["RAW_BUCKET_NAME"]
    profile_path = os.environ.get("CANDIDATE_PROFILE_PATH", str(DEFAULT_CANDIDATE_PROFILE_PATH))

    s3 = boto3.client("s3")
    bedrock_runtime = boto3.client("bedrock-runtime")
    table = boto3.resource("dynamodb").Table(table_name)

    updated = []
    for record in event.get("Records", []):
        if record.get("eventName") != "INSERT":
            # Defense in depth: the CDK event source is already filtered to INSERT only
            # (see jobpulse_stack.py), so this Lambda's own update_item calls (which
            # produce MODIFY records) never reach here in production.
            continue
        updated.append(_process_record(s3, bedrock_runtime, table, model_id, bucket, record, profile_path))

    result = {"processed": len(updated), "posting_ids": updated}
    log_event(logger, "embedding batch complete", **result)
    return result
