"""Thin wrapper around Bedrock's Claude Messages API, plus LLM-JSON repair (PLAN.md
Phase 3.1-3.2).
"""
from __future__ import annotations

import json
import re


class ExtractionError(Exception):
    """Raised when the LLM's response can't be turned into valid JSON."""


def invoke_claude(bedrock_runtime, model_id: str, prompt: str, max_tokens: int = 1024) -> str:
    """Call Bedrock's Claude Messages API and return the model's text response."""
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0,  # deterministic structured extraction, not creative text
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )
    payload = json.loads(response["body"].read())
    return payload["content"][0]["text"]


def parse_llm_json(text: str) -> dict:
    """Parse the model's response as JSON, tolerating a markdown code fence around it."""
    try:
        return json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"LLM did not return valid JSON: {exc}") from exc


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^\s*json\s*\n", "", text, flags=re.IGNORECASE)
    return text.strip()
