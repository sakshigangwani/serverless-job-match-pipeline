"""Train the re-ranker (PLAN.md Phase 6.4-5): engineered features in, a trained
classifier out, packaged as a self-describing joblib artifact for offline evaluation
(ml/evaluate.py), plus a pure-Python inference JSON for Phase 7's threshold Lambda.

Trains both a logistic regression and a gradient-boosted tree (XGBoost) — spec section
3.1 asks for both, since a linear model is easy to explain in an interview while a
boosted tree usually captures more of the feature interactions — and keeps whichever
scores higher on the held-out test set, rather than assuming one is always better, for
*offline comparison and reporting*.

That "best" model isn't necessarily what gets deployed, though: scikit-learn + numpy +
scipy + xgboost together are ~345MB unzipped (verified directly — pip-downloaded the
real manylinux wheels and measured), well past Lambda's 250MB function+layers limit.
Rather than pay that packaging cost, only the logistic regression's fitted parameters
(a weight per feature, plus an intercept) are exported as JSON — a Lambda can compute
`sigmoid(dot(weights, features) + intercept)` in a few lines of stdlib Python with no ML
library installed at all. See export_inference_json() below and
lambdas/threshold/reranker.py for the matching runtime side.

The same reasoning applies to Phase 13's SHAP explainability: the real `shap` library
depends on numpy (and more, depending on the explainer) — instead of bundling it into
the Lambda, export_inference_json also carries each feature's training-set mean
(compute_feature_means), which is exactly the baseline `shap.LinearExplainer` uses for a
linear model, letting the Lambda compute exact (not sampled/approximated) SHAP values
itself with `weight_i * (x_i - feature_means[i])`. `shap` itself is only ever imported
offline, to prove that formula matches (tests/ml/test_train.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
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
INFERENCE_JSON_PATH = Path(__file__).resolve().parent / "artifacts" / "reranker_inference.json"


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


def compute_feature_means(X_train: list[list[float]]) -> list[float]:
    """Per-feature mean over the training set — the SHAP baseline (PLAN.md Phase 13):
    for a linear model with independent features, SHAP's "expected value" reference is
    exactly the background dataset's mean, so `shap.LinearExplainer`'s default output
    (with a mean-summarized background) matches `weight_i * (x_i - feature_means[i])`
    (see lambdas/threshold/reranker.compute_shap_values). Plain stdlib `statistics.mean`
    rather than numpy — X_train is already a plain list of lists here, and this is a
    training-time-only computation, so there's no runtime-packaging reason to pull numpy
    in just for this.
    """
    return [statistics.mean(column) for column in zip(*X_train)]


def export_inference_json(
    model: LogisticRegression, feature_names: list[str], feature_means: list[float]
) -> dict[str, Any]:
    """A LogisticRegression's decision function is exactly `dot(coef_, x) + intercept_`
    — extracting those fitted numbers is all a pure-Python runtime needs; it doesn't
    need scikit-learn itself. Only implemented for logistic regression: there's no
    equally small, equally faithful pure-Python re-implementation of an XGBoost
    ensemble's prediction logic.

    `feature_means` (see compute_feature_means) rides along in the same self-describing
    artifact so the threshold Lambda can compute SHAP values with no separate file and
    no risk of the baseline drifting out of sync with these weights.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"export_inference_json only supports LogisticRegression, got {type(model).__name__}"
        )
    return {
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "feature_names": feature_names,
        "feature_means": feature_means,
    }


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
        "models": models,
        "train_size": len(train_postings),
        "test_size": len(test_postings),
        "X_train": X_train,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", default=str(DEFAULT_CANDIDATE_PROFILE_PATH))
    parser.add_argument("--artifact-path", default=str(ARTIFACT_PATH))
    parser.add_argument("--inference-json-path", default=str(INFERENCE_JSON_PATH))
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
    print(f"Selected model (offline comparison): {result['model_name']} (test ROC-AUC: {result['test_auc']:.3f})")
    print(f"Train/test sizes: {result['train_size']}/{result['test_size']}")
    print(f"Saved joblib artifact to: {artifact_path}")

    # Deployment always uses logistic regression, regardless of which model won the
    # offline comparison above — see the module docstring for the package-size reason.
    logistic_regression = result["models"]["logistic_regression"]
    feature_means = compute_feature_means(result["X_train"])
    inference_json_path = Path(args.inference_json_path)
    inference_json_path.parent.mkdir(parents=True, exist_ok=True)
    inference_json_path.write_text(
        json.dumps(export_inference_json(logistic_regression, FEATURE_NAMES, feature_means), indent=2)
    )

    if result["model_name"] != "logistic_regression":
        print(
            f"NOTE: xgboost won the offline comparison (AUC={result['test_auc']:.3f}) but is "
            "not deployed — bundling scikit-learn+xgboost+numpy+scipy into a Lambda is "
            "~345MB unzipped, over the 250MB function+layers limit. Deploying "
            "logistic_regression instead. A container-image Lambda (10GB limit) would lift "
            "this constraint if xgboost's improvement is ever worth the added complexity."
        )
    print(f"Saved deployable inference JSON (logistic_regression) to: {inference_json_path}")


if __name__ == "__main__":
    main()
