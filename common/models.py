"""Shared data contracts for JobPulse (spec PLAN.md Phase 1).

These models are the single source of truth for the candidate-profile shape and the
structured-posting shape. Every Lambda that reads or writes a posting record (extract,
embed, threshold, query_api, skill_gap, dlq_redrive) and every offline ml/ script must
import from here rather than redefining fields, so the schema only changes in one place.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RemoteStatus = Literal["remote", "hybrid", "onsite", "unknown"]
VisaStatus = Literal["sponsors", "no_sponsorship", "unknown"]

# Bundled into CommonLayer alongside common/*.py (PLAN.md Phase 0's layer build copies
# *.json too) — the default pluggable profile the embedding Lambda embeds against unless
# CANDIDATE_PROFILE_PATH overrides it (PLAN.md Phase 4.1).
DEFAULT_CANDIDATE_PROFILE_PATH = Path(__file__).resolve().parent / "candidate_profile.example.json"


class ExperienceEntry(BaseModel):
    title: str
    company: str
    start_date: str  # "YYYY-MM"
    end_date: str | None = None  # None means "present"
    description: str = ""

    def as_text(self) -> str:
        span = f"{self.start_date} - {self.end_date or 'present'}"
        return f"{self.title} at {self.company} ({span}): {self.description}".strip()


class ProjectEntry(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        tech = ", ".join(self.technologies)
        return f"{self.name}: {self.description} (technologies: {tech})".strip()


class CandidateProfile(BaseModel):
    """A pluggable candidate profile.

    JobPulse is a general-purpose job-market intelligence tool, not a single-user script
    (spec section 1) — a profile is loaded from a JSON file rather than hardcoded, so any
    candidate can plug in their own resume data without touching pipeline code.
    """

    candidate_id: str
    resume_text: str
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> CandidateProfile:
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)

    def section_texts(self) -> dict[str, str]:
        """Per-section text blocks for the per-section embeddings in PLAN.md Phase 4."""
        return {
            "resume": self.resume_text,
            "skills": ", ".join(self.skills),
            "experience": "\n".join(e.as_text() for e in self.experience),
            "projects": "\n".join(p.as_text() for p in self.projects),
        }


class Posting(BaseModel):
    """The structured record produced by extraction (PLAN.md Phase 3) and enriched by
    embedding (Phase 4) and scoring (Phase 7). Field set matches spec section 2.1's
    "Structured store" row and section 1's extraction field list, plus the housekeeping
    fields (`posting_id`, `posting_hash`, `ingested_at`, `score`, `alert_sent`) needed to
    operate the pipeline.
    """

    posting_id: str
    source: str
    source_url: str
    posting_hash: str
    ingested_at: datetime

    title: str | None = None
    company: str | None = None
    comp_min: int | None = None
    comp_max: int | None = None
    seniority: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    remote_status: RemoteStatus = "unknown"
    visa_status: VisaStatus = "unknown"
    description_text: str = ""

    # Raw cosine similarity against the candidate's resume embedding (PLAN.md Phase 4) —
    # the embedding-only baseline. Distinct from `score`, which is the final re-ranked
    # decision score produced once the Phase 6 model exists.
    embedding_similarity: float | None = None

    score: float | None = None
    alert_sent: bool = False

    @model_validator(mode="after")
    def _check_comp_range(self) -> Posting:
        if (
            self.comp_min is not None
            and self.comp_max is not None
            and self.comp_min > self.comp_max
        ):
            raise ValueError("comp_min must be <= comp_max")
        return self
