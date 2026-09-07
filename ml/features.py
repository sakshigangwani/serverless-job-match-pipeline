"""Engineered features for the re-ranker (PLAN.md Phase 6.4).

This module is shared by training (ml/train.py) and, in Phase 7, the threshold Lambda's
inference path — training and serving must compute features identically, or the model's
learned weights are meaningless at inference time ("training/serving skew"). Importing
from here rather than reimplementing feature logic in the Lambda is what guarantees that.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from common.models import CandidateProfile, Posting

# Order matters: this is the exact column order used for both training and inference.
FEATURE_NAMES = [
    "skill_overlap_count",
    "seniority_match",
    "comp_range_fit",
    "remote_match",
    "visa_match",
    "embedding_similarity",
]

# A heuristic mapping from free-text seniority labels (including whatever the Phase 3
# LLM extraction happened to produce) to an ordinal level. Unmapped/unknown text can't
# be compared, so it's treated as "no information" rather than guessed at.
_SENIORITY_LEVELS = {
    "junior": 0,
    "entry": 0,
    "mid": 1,
    "mid-level": 1,
    "intermediate": 1,
    "senior": 2,
    "staff": 3,
    "lead": 4,
    "principal": 4,
}
_MAX_SENIORITY_DISTANCE = max(_SENIORITY_LEVELS.values()) - min(_SENIORITY_LEVELS.values())


def skill_overlap_count(posting: Posting, candidate: CandidateProfile) -> float:
    """Case-insensitive count of skills the posting asks for that the candidate has."""
    posting_skills = {s.lower() for s in posting.required_skills}
    candidate_skills = {s.lower() for s in candidate.skills}
    return float(len(posting_skills & candidate_skills))


def estimate_candidate_seniority(candidate: CandidateProfile, as_of: date | None = None) -> str | None:
    """Heuristic: total months of experience -> a seniority label.

    A simplification — a real system might let the candidate state their level
    directly rather than inferring it from experience-entry dates, and the thresholds
    below are one reasonable choice among many. Returns None (no info) for a candidate
    with no experience entries, rather than guessing "junior".
    """
    if not candidate.experience:
        return None

    as_of = as_of or datetime.now(timezone.utc).date()
    total_months = 0
    for entry in candidate.experience:
        start = _parse_year_month(entry.start_date)
        end = _parse_year_month(entry.end_date) if entry.end_date else as_of
        if start is None or end is None:
            continue
        total_months += max(0, (end.year - start.year) * 12 + (end.month - start.month))

    years = total_months / 12
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    if years < 8:
        return "senior"
    if years < 12:
        return "staff"
    return "lead"


def _parse_year_month(value: str) -> date | None:
    try:
        year, month = value.split("-")
        return date(int(year), int(month), 1)
    except (ValueError, AttributeError):
        return None


def seniority_match(posting: Posting, candidate: CandidateProfile, as_of: date | None = None) -> float:
    """1.0 for an exact seniority match, decaying toward 0.0 the further apart the
    posting's and candidate's levels are; 0.5 ("no information") when either side can't
    be mapped to a known level.
    """
    posting_level = _SENIORITY_LEVELS.get((posting.seniority or "").lower())
    candidate_seniority = estimate_candidate_seniority(candidate, as_of=as_of)
    candidate_level = _SENIORITY_LEVELS.get(candidate_seniority or "")

    if posting_level is None or candidate_level is None:
        return 0.5

    distance = abs(posting_level - candidate_level)
    return max(0.0, 1 - distance / _MAX_SENIORITY_DISTANCE)


def comp_range_fit(posting: Posting, candidate: CandidateProfile) -> float:
    """1.0 if the posting's (best-known) comp meets the candidate's stated minimum,
    a partial-credit ratio if it falls short, or 0.5 ("no information") if either side
    hasn't stated a number.
    """
    if candidate.desired_comp_min is None:
        return 0.5

    effective_comp = posting.comp_max if posting.comp_max is not None else posting.comp_min
    if effective_comp is None:
        return 0.5

    if effective_comp >= candidate.desired_comp_min:
        return 1.0
    return max(0.0, effective_comp / candidate.desired_comp_min)


def remote_match(posting: Posting, candidate: CandidateProfile) -> float:
    """1.0/0.0 for a stated match/mismatch, 0.5 ("no information") if either the
    candidate has no preference or the posting didn't disclose its remote status.
    """
    if candidate.remote_preference == "unknown" or posting.remote_status == "unknown":
        return 0.5
    return 1.0 if posting.remote_status == candidate.remote_preference else 0.0


def visa_match(posting: Posting, candidate: CandidateProfile) -> float:
    """1.0 if the candidate doesn't need sponsorship (any posting is fine), otherwise
    reflects whether the posting says it sponsors, doesn't, or didn't disclose.
    """
    if not candidate.needs_sponsorship:
        return 1.0
    if posting.visa_status == "sponsors":
        return 1.0
    if posting.visa_status == "unknown":
        return 0.5
    return 0.0


def build_feature_vector(
    posting: Posting, candidate: CandidateProfile, as_of: date | None = None
) -> dict[str, float]:
    """The full engineered-feature vector for one posting, keyed by FEATURE_NAMES."""
    return {
        "skill_overlap_count": skill_overlap_count(posting, candidate),
        "seniority_match": seniority_match(posting, candidate, as_of=as_of),
        "comp_range_fit": comp_range_fit(posting, candidate),
        "remote_match": remote_match(posting, candidate),
        "visa_match": visa_match(posting, candidate),
        "embedding_similarity": posting.embedding_similarity or 0.0,
    }


def feature_vector_to_array(vector: dict[str, float]) -> list[float]:
    """Order a feature dict per FEATURE_NAMES for a model's expected input shape."""
    return [vector[name] for name in FEATURE_NAMES]
