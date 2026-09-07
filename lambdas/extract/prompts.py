"""The structured-extraction prompt (PLAN.md Phase 3.1).

Field set and allowed enum values must stay in sync with common.models.Posting — the
handler validates the LLM's output against that schema, so a prompt/schema mismatch
shows up as extraction failures, not silently wrong data.
"""
from __future__ import annotations

EXTRACTION_INSTRUCTIONS = """You are extracting structured fields from a job posting for a job-matching pipeline.

Read the posting title and description below and return ONLY a single JSON object (no \
markdown fences, no commentary) with exactly these keys:

- "title": string, the job title (clean it up if the source title is noisy)
- "company": string, or null if not mentioned
- "comp_min": integer (annual USD), or null if no compensation is mentioned
- "comp_max": integer (annual USD), or null if no compensation is mentioned
- "seniority": string (e.g. "junior", "mid", "senior", "staff", "lead"), or null if unclear
- "required_skills": array of strings, the key required skills/technologies
- "remote_status": one of exactly "remote", "hybrid", "onsite", "unknown"
- "visa_status": one of exactly "sponsors", "no_sponsorship", "unknown"

Only use information stated in the posting. Do not guess a value that isn't stated —
use null (or "unknown" for the two status fields) instead."""


def build_extraction_prompt(title: str, description_text: str) -> str:
    return (
        f"{EXTRACTION_INSTRUCTIONS}\n\n"
        f"Title: {title}\n\n"
        f"Description:\n{description_text}\n"
    )
