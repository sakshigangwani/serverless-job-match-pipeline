"""Embedding Lambda (PLAN.md Phase 4): triggered by S3 ObjectCreated events under
structured/, embeds each posting's description text and the candidate profile (once,
cached) with Bedrock Titan Embeddings, and writes the posting's embedding vector plus a
cosine-similarity baseline to S3 under embeddings/.

Writes only to embeddings/ — never back to the structured/ key that triggered this
Lambda — so this notification can never re-trigger itself.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse

import boto3

from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile, Posting
from common.similarity import cosine_similarity
from common.storage_keys import CANDIDATE_EMBEDDINGS_KEY, posting_embedding_key

try:
    # Local/package-style import, used by pytest (lambdas.embed.handler).
    from lambdas.embed.bedrock_embeddings import embed_text
except ImportError:
    # AWS Lambda flattens lambdas/embed/* into /var/task, so this is a top-level
    # sibling module there, not part of a "lambdas.embed" package.
    from bedrock_embeddings import embed_text  # type: ignore[no-redef]

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def _process_record(
    s3, bedrock_runtime, model_id: str, bucket: str, key: str, profile_path: str
) -> str:
    posting = Posting.model_validate_json(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    candidate_embeddings = _get_or_load_candidate_embeddings(
        s3, bedrock_runtime, model_id, bucket, profile_path
    )

    posting_embedding = embed_text(bedrock_runtime, model_id, posting.description_text)

    resume_vector = candidate_embeddings["sections"].get("resume")
    similarity = cosine_similarity(posting_embedding, resume_vector) if resume_vector else None

    out_key = posting_embedding_key(posting.source, posting.ingested_at.date(), posting.posting_hash)
    s3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=json.dumps(
            {
                "posting_id": posting.posting_id,
                "candidate_id": candidate_embeddings["candidate_id"],
                "model_id": model_id,
                "embedding": posting_embedding,
                "embedding_similarity": similarity,
            }
        ).encode("utf-8"),
        ContentType="application/json",
    )
    return out_key


def handler(event, context):
    model_id = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]
    profile_path = os.environ.get("CANDIDATE_PROFILE_PATH", str(DEFAULT_CANDIDATE_PROFILE_PATH))

    s3 = boto3.client("s3")
    bedrock_runtime = boto3.client("bedrock-runtime")

    written = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        written.append(_process_record(s3, bedrock_runtime, model_id, bucket, key, profile_path))

    result = {"processed": len(written), "written_keys": written}
    logger.info(json.dumps(result))
    return result
