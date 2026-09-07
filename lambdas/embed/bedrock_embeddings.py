"""Thin wrapper around Bedrock's Titan Embeddings API (PLAN.md Phase 4.2).

Deliberately sends only `inputText` — no Titan-V2-specific params like `dimensions` or
`normalize` — so BEDROCK_EMBEDDING_MODEL_ID can be swapped between Titan Embeddings
versions (or a future one) without any code change here.
"""
from __future__ import annotations

import json


def embed_text(bedrock_runtime, model_id: str, text: str) -> list[float]:
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    payload = json.loads(response["body"].read())
    return payload["embedding"]
