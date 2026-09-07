"""Synthetic labeled dataset generator (PLAN.md Phase 6.3, standing in for real labels).

The spec calls for manually labeling 150-300 *real* postings against the candidate's
*real* resume — that's inherently something only a human with a real resume and real
scraped postings can do, and can't be fabricated here. This module generates a
synthetic-but-structurally-realistic dataset instead, so the rest of the ML pipeline
(feature engineering, baselines, training, evaluation) can be built, tested, and proven
correct end-to-end before real labels exist. See ml/README.md for how to replace this
with a real labeled dataset later — every downstream script (ml/train.py,
ml/evaluate.py) reads the same ml/dataset.py file format regardless of where it came
from, so nothing else changes when you do.

Ground-truth construction: each posting's "true fit" is a weighted combination of the
*real* engineered features from ml/features.py (not a separately-invented rule) — this
makes the synthetic dataset a genuine sanity check that the feature+model pipeline is
wired correctly. A posting's embedding_similarity is a *noisy* proxy for that true fit
(simulating an imperfect semantic signal), and the label itself has separate noise
(simulating that even manual human labeling isn't perfectly consistent). The intent is
that the re-ranker, which sees the clean structured features *and* the noisy embedding
similarity, should out-perform either noisy signal alone — the same hypothesis the real
evaluation is meant to test.
"""
from __future__ import annotations

import random
from datetime import date, datetime, time, timezone

from common.models import CandidateProfile, Posting
from ml.features import (
    comp_range_fit,
    remote_match,
    seniority_match,
    skill_overlap_count,
    visa_match,
)

SKILL_POOL = [
    "Python", "AWS", "SQL", "Machine Learning", "Docker", "Kubernetes", "React",
    "Java", "Go", "Terraform", "DynamoDB", "Spark", "TensorFlow", "PyTorch",
    "CI/CD", "Linux", "GraphQL", "Node.js",
]
SENIORITY_POOL = ["junior", "mid", "senior", "staff", "lead"]
REMOTE_POOL = ["remote", "hybrid", "onsite", "unknown"]
VISA_POOL = ["sponsors", "no_sponsorship", "unknown"]

# Fixed so estimate_candidate_seniority — and every other timestamp in the dataset — is
# exactly reproducible regardless of what day this script happens to be run on.
REFERENCE_DATE = date(2026, 1, 1)
REFERENCE_DATETIME = datetime.combine(REFERENCE_DATE, time.min, tzinfo=timezone.utc)

_FEATURE_WEIGHTS = {
    "skill_overlap": 0.15,
    "seniority_match": 0.25,
    "comp_range_fit": 0.20,
    "remote_match": 0.20,
    "visa_match": 0.20,
}
# Set near the empirical median of _clean_fit_score across the skill/seniority/comp/
# remote/visa distributions above, so the synthetic dataset comes out roughly balanced
# rather than mostly-one-class (which would make precision/recall uninformative).
_LABEL_THRESHOLD = 0.69


def _clean_fit_score(posting: Posting, candidate: CandidateProfile) -> float:
    """0..1 weighted combination of the real engineered features (minus embedding
    similarity, which is generated as a noisy proxy of this score, not an input to it).
    """
    normalized_skill_overlap = min(1.0, skill_overlap_count(posting, candidate) / 3)
    return (
        _FEATURE_WEIGHTS["skill_overlap"] * normalized_skill_overlap
        + _FEATURE_WEIGHTS["seniority_match"] * seniority_match(posting, candidate, as_of=REFERENCE_DATE)
        + _FEATURE_WEIGHTS["comp_range_fit"] * comp_range_fit(posting, candidate)
        + _FEATURE_WEIGHTS["remote_match"] * remote_match(posting, candidate)
        + _FEATURE_WEIGHTS["visa_match"] * visa_match(posting, candidate)
    )


def _make_posting(index: int, rng: random.Random) -> Posting:
    required_skills = rng.sample(SKILL_POOL, k=rng.randint(2, 6))
    seniority = rng.choice(SENIORITY_POOL)
    remote_status = rng.choice(REMOTE_POOL)
    visa_status = rng.choice(VISA_POOL)
    comp_min = rng.randint(60, 220) * 1000
    comp_max = comp_min + rng.randint(0, 40) * 1000
    posting_id = f"synthetic-{index:04d}"

    return Posting(
        posting_id=posting_id,
        source="synthetic",
        source_url=f"https://example.com/synthetic/{index}",
        posting_hash=posting_id,
        ingested_at=REFERENCE_DATETIME,
        title=f"{seniority.title()} Engineer",
        required_skills=required_skills,
        seniority=seniority,
        remote_status=remote_status,
        visa_status=visa_status,
        comp_min=comp_min,
        comp_max=comp_max,
        description_text=(
            f"We are looking for a {seniority} engineer skilled in "
            f"{', '.join(required_skills)}. This is a {remote_status} position."
        ),
    )


def generate_dataset(
    candidate: CandidateProfile, n: int = 200, seed: int = 42
) -> tuple[list[Posting], dict[str, int]]:
    """Deterministic given (candidate, n, seed) — same inputs always produce the exact
    same postings and labels, so the synthetic demo dataset is fully reproducible.
    """
    rng = random.Random(seed)
    postings: list[Posting] = []
    labels: dict[str, int] = {}

    for i in range(n):
        posting = _make_posting(i, rng)
        clean_score = _clean_fit_score(posting, candidate)

        # embedding_similarity: a noisy proxy of the clean score, rescaled from [0, 1]
        # to roughly cosine similarity's [-1, 1] range, then clipped to stay valid.
        embedding_noise = rng.gauss(0, 0.2)
        posting.embedding_similarity = max(-1.0, min(1.0, clean_score * 2 - 1 + embedding_noise))

        # The label itself gets separate, smaller noise: even a real human labeler
        # wouldn't apply a "good fit" cutoff with perfect consistency.
        label_noise = rng.gauss(0, 0.1)
        labels[posting.posting_id] = 1 if (clean_score + label_noise) >= _LABEL_THRESHOLD else 0

        postings.append(posting)

    return postings, labels
