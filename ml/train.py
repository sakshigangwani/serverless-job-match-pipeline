"""Train the re-ranker (PLAN.md Phase 6.4-5): engineered features in, a trained
classifier out, packaged as a self-describing joblib artifact for Phase 7's threshold
Lambda to load without retraining at inference time.

Trains both a logistic regression and a gradient-boosted tree (XGBoost) — spec section
3.1 asks for both, since a linear model is easy to explain in an interview while a
boosted tree usually captures more of the feature interactions — and keeps whichever
scores higher on the held-out test set, rather than assuming one is always better.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile, Posting
from ml.dataset import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    load_dataset,
    split_dataset,
)
from ml.features import FEATURE_NAMES, build_feature_vector, feature_vector_to_array

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "reranker.joblib"


def build_training_arrays(
    postings: list[Posting], labels: dict[str, int], candidate: CandidateProfile
) -> tuple[list[list[float]], list[int]]:
    X = [feature_vector_to_array(build_feature_vector(p, candidate)) for p in postings]
    y = [labels[p.posting_id] for p in postings]
    return X, y


def train_candidate_models(
    X_train: list[list[float]], y_train: list[int], random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Any]:
    logistic_regression = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    xgboost_model = XGBClassifier(
        n_estimators=100, max_depth=3, eval_metric="logloss", random_state=random_state
    ).fit(X_train, y_train)
    return {"logistic_regression": logistic_regression, "xgboost": xgboost_model}


def select_best_model(
    models: dict[str, Any], X_test: list[list[float]], y_test: list[int]
) -> tuple[str, Any, float]:
    """Picks by ROC-AUC on the held-out test set — independent of any single
    classification threshold, so it reflects overall ranking quality.
    """
    best_name, best_model, best_auc = "", None, -1.0
    for name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
        if auc > best_auc:
            best_name, best_model, best_auc = name, model, auc
    return best_name, best_model, best_auc


def run_training(
    postings: list[Posting],
    labels: dict[str, int],
    candidate: CandidateProfile,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Pure orchestration (no file I/O) so it's directly unit-testable: split, build
    feature arrays, train both candidate models, and pick the better one.
    """
    train_postings, test_postings, train_labels, test_labels = split_dataset(
        postings, labels, test_size=test_size, random_state=random_state
    )
    X_train, y_train = build_training_arrays(train_postings, train_labels, candidate)
    X_test, y_test = build_training_arrays(test_postings, test_labels, candidate)

    models = train_candidate_models(X_train, y_train, random_state=random_state)
    best_name, best_model, best_auc = select_best_model(models, X_test, y_test)

    return {
        "model_name": best_name,
        "model": best_model,
        "test_auc": best_auc,
        "train_size": len(train_postings),
        "test_size": len(test_postings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", default=str(DEFAULT_CANDIDATE_PROFILE_PATH))
    parser.add_argument("--artifact-path", default=str(ARTIFACT_PATH))
    args = parser.parse_args()

    candidate = CandidateProfile.from_file(args.profile_path)
    postings, labels = load_dataset()

    result = run_training(postings, labels, candidate)

    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result["model"],
            "model_name": result["model_name"],
            "feature_names": FEATURE_NAMES,
        },
        artifact_path,
    )

    print(f"Selected model: {result['model_name']} (test ROC-AUC: {result['test_auc']:.3f})")
    print(f"Train/test sizes: {result['train_size']}/{result['test_size']}")
    print(f"Saved to: {artifact_path}")


if __name__ == "__main__":
    main()
