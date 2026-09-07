from datetime import datetime, timezone

from common.models import CandidateProfile, Posting
from ml.baselines import bm25_scores, embedding_baseline_scores


def _posting(**overrides) -> Posting:
    defaults = {
        "posting_id": "p1",
        "source": "remoteok",
        "source_url": "https://example.com/job/1",
        "posting_hash": "p1",
        "ingested_at": datetime.now(timezone.utc),
        "description_text": "",
    }
    defaults.update(overrides)
    return Posting(**defaults)


def test_bm25_ranks_matching_posting_higher_than_unrelated_one():
    # BM25's IDF term (log((N - freq + 0.5) / (freq + 0.5))) collapses to exactly zero
    # for a term appearing in exactly half of a 2-document corpus, so a couple of filler
    # postings are included to keep the IDF values non-degenerate at this tiny scale.
    candidate = CandidateProfile(
        candidate_id="c1",
        resume_text="Experienced backend engineer",
        skills=["python", "aws", "dynamodb"],
    )
    matching = _posting(
        posting_id="matching",
        description_text="We need a python backend engineer with aws and dynamodb experience",
    )
    unrelated = _posting(
        posting_id="unrelated",
        description_text="Looking for a graphic designer with adobe photoshop skills",
    )
    filler_a = _posting(posting_id="filler_a", description_text="Sales representative role")
    filler_b = _posting(posting_id="filler_b", description_text="Warehouse operations associate")

    scores = bm25_scores([matching, unrelated, filler_a, filler_b], candidate)

    assert scores[0] > scores[1]


def test_bm25_returns_one_score_per_posting_in_order():
    candidate = CandidateProfile(candidate_id="c1", resume_text="engineer", skills=["python"])
    postings = [_posting(posting_id=str(i), description_text=f"posting {i}") for i in range(5)]

    scores = bm25_scores(postings, candidate)

    assert len(scores) == 5


def test_embedding_baseline_returns_the_stored_similarity():
    postings = [_posting(embedding_similarity=0.42)]

    assert embedding_baseline_scores(postings) == [0.42]


def test_embedding_baseline_defaults_missing_similarity_to_zero():
    postings = [_posting(embedding_similarity=None)]

    assert embedding_baseline_scores(postings) == [0.0]


def test_embedding_baseline_preserves_order():
    postings = [
        _posting(posting_id="a", embedding_similarity=0.1),
        _posting(posting_id="b", embedding_similarity=0.9),
    ]

    assert embedding_baseline_scores(postings) == [0.1, 0.9]
