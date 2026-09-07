import math

import pytest

from common.models import CandidateProfile
from ml.features import FEATURE_NAMES, build_feature_vector, feature_vector_to_array
from ml.synthetic_data import generate_dataset
from ml.train import (
    build_training_arrays,
    export_inference_json,
    run_training,
    select_best_model,
    train_candidate_models,
)

CANDIDATE = CandidateProfile.from_file("common/candidate_profile.example.json")


def test_build_training_arrays_produces_one_row_per_posting():
    postings, labels = generate_dataset(CANDIDATE, n=20, seed=42)

    X, y = build_training_arrays(postings, labels, CANDIDATE)

    assert len(X) == len(postings) == len(y)
    assert all(len(row) == 6 for row in X)  # 6 engineered features


def test_train_candidate_models_returns_both_model_types():
    postings, labels = generate_dataset(CANDIDATE, n=100, seed=42)
    X, y = build_training_arrays(postings, labels, CANDIDATE)

    models = train_candidate_models(X, y)

    assert set(models) == {"logistic_regression", "xgboost"}
    # Both must be able to predict fit-probabilities on unseen rows.
    for model in models.values():
        probabilities = model.predict_proba(X[:5])
        assert probabilities.shape == (5, 2)


def test_select_best_model_returns_a_valid_auc():
    postings, labels = generate_dataset(CANDIDATE, n=100, seed=42)
    X, y = build_training_arrays(postings, labels, CANDIDATE)
    models = train_candidate_models(X, y)

    name, model, auc = select_best_model(models, X, y)

    assert name in models
    assert model is models[name]
    assert 0.0 <= auc <= 1.0


def test_run_training_end_to_end_on_synthetic_data():
    postings, labels = generate_dataset(CANDIDATE, n=150, seed=42)

    result = run_training(postings, labels, CANDIDATE)

    assert result["model_name"] in {"logistic_regression", "xgboost"}
    assert 0.0 <= result["test_auc"] <= 1.0
    assert result["train_size"] + result["test_size"] == 150
    assert result["train_size"] > result["test_size"]  # 70/30 split
    assert hasattr(result["model"], "predict_proba")


def test_run_training_is_deterministic_given_the_same_inputs():
    postings, labels = generate_dataset(CANDIDATE, n=100, seed=42)

    result_a = run_training(postings, labels, CANDIDATE)
    result_b = run_training(postings, labels, CANDIDATE)

    assert result_a["model_name"] == result_b["model_name"]
    assert result_a["test_auc"] == result_b["test_auc"]


def test_run_training_exposes_both_trained_models():
    postings, labels = generate_dataset(CANDIDATE, n=100, seed=42)

    result = run_training(postings, labels, CANDIDATE)

    assert set(result["models"]) == {"logistic_regression", "xgboost"}


def test_export_inference_json_matches_the_models_own_predict_proba():
    postings, labels = generate_dataset(CANDIDATE, n=100, seed=42)
    X, y = build_training_arrays(postings, labels, CANDIDATE)
    models = train_candidate_models(X, y)
    logistic_regression = models["logistic_regression"]

    exported = export_inference_json(logistic_regression, FEATURE_NAMES)

    for posting in postings[:10]:
        features = feature_vector_to_array(build_feature_vector(posting, CANDIDATE))
        expected = logistic_regression.predict_proba([features])[0][1]

        z = exported["intercept"] + sum(w * f for w, f in zip(exported["weights"], features))
        actual = 1.0 / (1.0 + math.exp(-z))

        assert actual == pytest.approx(expected, abs=1e-9)


def test_export_inference_json_rejects_non_logistic_models():
    postings, labels = generate_dataset(CANDIDATE, n=100, seed=42)
    X, y = build_training_arrays(postings, labels, CANDIDATE)
    models = train_candidate_models(X, y)

    with pytest.raises(TypeError):
        export_inference_json(models["xgboost"], FEATURE_NAMES)
