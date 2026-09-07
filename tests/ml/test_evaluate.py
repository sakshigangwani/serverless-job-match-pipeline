from common.models import CandidateProfile
from ml.dataset import split_dataset
from ml.evaluate import (
    _format_report,
    evaluate_ranking,
    precision_at_k,
    recall_at_k,
    run_evaluation,
)
from ml.synthetic_data import generate_dataset
from ml.train import run_training

CANDIDATE = CandidateProfile.from_file("common/candidate_profile.example.json")


def test_precision_at_k_counts_positives_in_top_k():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    labels = [1, 1, 0, 0, 0]

    assert precision_at_k(scores, labels, k=2) == 1.0
    assert precision_at_k(scores, labels, k=4) == 0.5


def test_precision_at_k_ignores_score_order_in_the_input_lists():
    # Deliberately unsorted input: precision_at_k must sort internally.
    scores = [0.5, 0.9, 0.6, 0.8, 0.7]
    labels = [0, 1, 0, 1, 0]

    assert precision_at_k(scores, labels, k=2) == 1.0


def test_recall_at_k_is_fraction_of_all_positives_found():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    labels = [1, 0, 1, 0, 1]  # 3 total positives

    assert recall_at_k(scores, labels, k=1) == 1 / 3
    assert recall_at_k(scores, labels, k=5) == 1.0


def test_recall_at_k_is_zero_with_no_positives_at_all():
    scores = [0.9, 0.8]
    labels = [0, 0]

    assert recall_at_k(scores, labels, k=2) == 0.0


def test_evaluate_ranking_returns_all_three_metrics():
    scores = [0.9, 0.1, 0.5, 0.4, 0.8]
    labels = [1, 0, 1, 0, 1]

    result = evaluate_ranking(scores, labels, k=3)

    assert set(result) == {"precision_at_k", "recall_at_k", "roc_auc"}
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_run_evaluation_covers_all_three_approaches():
    postings, labels = generate_dataset(CANDIDATE, n=150, seed=42)
    training = run_training(postings, labels, CANDIDATE)
    _train, test_postings, _train_labels, test_labels = split_dataset(postings, labels)

    results = run_evaluation(test_postings, test_labels, CANDIDATE, training["model"], k=10)

    assert set(results) == {"keyword_bm25", "embedding_only", "embedding_plus_reranker"}
    for metrics in results.values():
        assert 0.0 <= metrics["precision_at_k"] <= 1.0
        assert 0.0 <= metrics["recall_at_k"] <= 1.0


def test_format_report_includes_synthetic_data_disclaimer():
    results = {
        "keyword_bm25": {"precision_at_k": 0.5, "recall_at_k": 0.3, "roc_auc": 0.6},
        "embedding_only": {"precision_at_k": 0.6, "recall_at_k": 0.4, "roc_auc": 0.65},
        "embedding_plus_reranker": {"precision_at_k": 0.8, "recall_at_k": 0.5, "roc_auc": 0.75},
    }

    report = _format_report(results, "xgboost", k=10, test_size=45)

    assert "synthetic" in report.lower()
    assert "not real" in report.lower() or "not be quoted" in report.lower()
    assert "xgboost" in report
    assert "0.750" in report  # embedding_plus_reranker roc_auc


def test_format_report_includes_all_three_approaches_as_rows():
    results = {
        "keyword_bm25": {"precision_at_k": 0.5, "recall_at_k": 0.3, "roc_auc": 0.6},
        "embedding_only": {"precision_at_k": 0.6, "recall_at_k": 0.4, "roc_auc": 0.65},
        "embedding_plus_reranker": {"precision_at_k": 0.8, "recall_at_k": 0.5, "roc_auc": 0.75},
    }

    report = _format_report(results, "logistic_regression", k=10, test_size=45)

    assert "Keyword / BM25" in report
    assert "Embedding similarity only" in report
    assert "Embedding + re-ranker" in report
