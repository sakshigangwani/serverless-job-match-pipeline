"""Regression test (PLAN.md Phase 10.1): the re-ranker's evaluated performance on the
committed synthetic dataset must not silently regress below a floor. Trains and
evaluates fresh here (rather than reading a possibly-stale checked-in report), so this
is part of the normal `pytest` run every PR already executes in CI — no separate
CI-only script needed to satisfy "run the ML evaluation script... assert metrics don't
regress below a floor."
"""
from common.models import CandidateProfile
from ml.dataset import load_dataset, split_dataset
from ml.evaluate import run_evaluation
from ml.train import run_training

CANDIDATE = CandidateProfile.from_file("common/candidate_profile.example.json")

# Set comfortably below the currently-measured values (0.540 / 0.689 / 0.749 ROC-AUC for
# keyword / embedding-only / embedding+re-ranker on the committed dataset) so this only
# fires on an actual regression in ml/features.py, ml/train.py, or the committed
# dataset — not on noise (there is none; generation is seeded and deterministic).
MIN_RERANKER_ROC_AUC = 0.65
MIN_RERANKER_PRECISION_AT_10 = 0.6


def test_reranker_meets_the_minimum_performance_floor():
    postings, labels = load_dataset()
    _train_postings, test_postings, _train_labels, test_labels = split_dataset(postings, labels)
    training = run_training(postings, labels, CANDIDATE)

    results = run_evaluation(test_postings, test_labels, CANDIDATE, training["model"], k=10)
    reranker = results["embedding_plus_reranker"]

    assert reranker["roc_auc"] >= MIN_RERANKER_ROC_AUC
    assert reranker["precision_at_k"] >= MIN_RERANKER_PRECISION_AT_10


def test_reranker_beats_the_embedding_only_baseline():
    """The whole point of the re-ranker (spec section 3): combining engineered features
    with the embedding signal should outperform the embedding signal alone.
    """
    postings, labels = load_dataset()
    _train_postings, test_postings, _train_labels, test_labels = split_dataset(postings, labels)
    training = run_training(postings, labels, CANDIDATE)

    results = run_evaluation(test_postings, test_labels, CANDIDATE, training["model"], k=10)

    assert results["embedding_plus_reranker"]["roc_auc"] >= results["embedding_only"]["roc_auc"]
