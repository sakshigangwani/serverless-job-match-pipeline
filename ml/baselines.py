"""The two non-learned baselines to compare the re-ranker against (PLAN.md Phase 6.1-2).

Both return one score per posting, in the same order as the input list, so
ml/evaluate.py can rank and score any of the three approaches (keyword, embedding,
embedding+re-ranker) identically.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from common.models import CandidateProfile, Posting


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_scores(postings: list[Posting], candidate: CandidateProfile) -> list[float]:
    """Keyword/BM25 baseline: rank postings by BM25 relevance of their description text
    against a query built from the candidate's resume and skills.
    """
    corpus = [_tokenize(p.description_text) for p in postings]
    bm25 = BM25Okapi(corpus)

    query = _tokenize(f"{candidate.resume_text} {' '.join(candidate.skills)}")
    return list(bm25.get_scores(query))


def embedding_baseline_scores(postings: list[Posting]) -> list[float]:
    """Embedding-only baseline: the raw cosine similarity already computed in Phase 4,
    with an unembedded posting (embedding_similarity=None) scored as 0.0 rather than
    excluded, so every posting still gets ranked.
    """
    return [p.embedding_similarity or 0.0 for p in postings]
