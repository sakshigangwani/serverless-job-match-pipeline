"""Evaluate the three approaches — keyword/BM25, embedding-only, and embedding +
re-ranker — on the same held-out test split, and write ml/evaluation_report.md
(PLAN.md Phase 6.6).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import roc_auc_score

from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile, Posting
from ml.baselines import bm25_scores, embedding_baseline_scores
from ml.dataset import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    load_dataset,
    split_dataset,
)
from ml.features import build_feature_vector, feature_vector_to_array
from ml.train import ARTIFACT_PATH

DEFAULT_K = 10
REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.md"


def precision_at_k(scores: list[float], labels: list[int], k: int) -> float:
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    top_k = ranked[:k]
    return sum(label for _, label in top_k) / k


def recall_at_k(scores: list[float], labels: list[int], k: int) -> float:
    total_positive = sum(labels)
    if total_positive == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    top_k = ranked[:k]
    return sum(label for _, label in top_k) / total_positive


def evaluate_ranking(scores: list[float], labels: list[int], k: int = DEFAULT_K) -> dict[str, float]:
    roc_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    return {
        "precision_at_k": precision_at_k(scores, labels, k),
        "recall_at_k": recall_at_k(scores, labels, k),
        "roc_auc": roc_auc,
    }


def run_evaluation(
    test_postings: list[Posting],
    test_labels: dict[str, int],
    candidate: CandidateProfile,
    model: Any,
    k: int = DEFAULT_K,
) -> dict[str, dict[str, float]]:
    y = [test_labels[p.posting_id] for p in test_postings]

    keyword = evaluate_ranking(bm25_scores(test_postings, candidate), y, k)
    embedding = evaluate_ranking(embedding_baseline_scores(test_postings), y, k)

    X = [feature_vector_to_array(build_feature_vector(p, candidate)) for p in test_postings]
    reranker_scores = list(model.predict_proba(X)[:, 1])
    reranker = evaluate_ranking(reranker_scores, y, k)

    return {"keyword_bm25": keyword, "embedding_only": embedding, "embedding_plus_reranker": reranker}


def _format_report(results: dict[str, dict[str, float]], model_name: str, k: int, test_size: int) -> str:
    def row(label: str, key: str) -> str:
        m = results[key]
        return f"| {label} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | {m['roc_auc']:.3f} |"

    delta_auc = results["embedding_plus_reranker"]["roc_auc"] - results["embedding_only"]["roc_auc"]

    return f"""# Match-Scoring Model Evaluation

**This report is generated from a synthetic demo dataset (`ml/data/`), not real manually
labeled postings.** Spec section 3.2 calls for manually labeling 150-300 *real* postings
against the candidate's *real* resume — that step requires a human and real scraped
data, and can't be fabricated. `ml/synthetic_data.py` generates a structurally realistic
stand-in so the training/evaluation pipeline itself is provably correct end-to-end. See
`ml/README.md` for how to regenerate this report from a real labeled dataset — the
numbers below should not be quoted as real model performance (e.g. in a resume bullet).

Re-ranker model: **{model_name}** · Test set size: {test_size} (held out, unseen during training) · K = {k}

| Approach | Precision@{k} | Recall@{k} | ROC-AUC |
|---|---|---|---|
{row("Keyword / BM25", "keyword_bm25")}
{row("Embedding similarity only", "embedding_only")}
{row("Embedding + re-ranker", "embedding_plus_reranker")}

Embedding + re-ranker vs. embedding-only baseline: **{delta_auc:+.3f} ROC-AUC**.
"""


def write_report(
    results: dict[str, dict[str, float]],
    model_name: str,
    k: int,
    test_size: int,
    report_path: Path = REPORT_PATH,
) -> None:
    report_path.write_text(_format_report(results, model_name, k, test_size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", default=str(DEFAULT_CANDIDATE_PROFILE_PATH))
    parser.add_argument("--artifact-path", default=str(ARTIFACT_PATH))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    candidate = CandidateProfile.from_file(args.profile_path)
    postings, labels = load_dataset()
    _train_postings, test_postings, _train_labels, test_labels = split_dataset(
        postings, labels, test_size=args.test_size, random_state=args.random_state
    )

    artifact = joblib.load(args.artifact_path)
    results = run_evaluation(test_postings, test_labels, candidate, artifact["model"], k=args.k)

    write_report(results, artifact["model_name"], args.k, len(test_postings))
    print(f"Wrote {REPORT_PATH}")
    for approach, metrics in results.items():
        print(approach, metrics)


if __name__ == "__main__":
    main()
