import json
import math

import pytest

from lambdas.threshold.reranker import (
    compute_shap_values,
    load_model,
    predict_proba,
    top_contributing_factors,
)


def test_load_model_reads_json_file(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"weights": [1.0], "intercept": 0.0, "feature_names": ["a"]}))

    model = load_model(model_path)

    assert model == {"weights": [1.0], "intercept": 0.0, "feature_names": ["a"]}


def test_predict_proba_matches_manual_sigmoid_computation():
    model = {"weights": [2.0, -1.0], "intercept": 0.5, "feature_names": ["a", "b"]}
    feature_vector = {"a": 1.0, "b": 3.0}

    result = predict_proba(model, feature_vector)

    z = 0.5 + 2.0 * 1.0 + -1.0 * 3.0
    expected = 1.0 / (1.0 + math.exp(-z))
    assert result == expected


def test_predict_proba_uses_the_models_own_feature_order_not_dict_order():
    model = {"weights": [10.0, 0.0], "intercept": 0.0, "feature_names": ["b", "a"]}
    # dict insertion order is a, b — deliberately the opposite of feature_names
    feature_vector = {"a": 1.0, "b": 100.0}

    result = predict_proba(model, feature_vector)

    # If feature order were taken from dict order instead of model["feature_names"],
    # this would weight "a" (1.0) by 10.0 instead of "b" (100.0) by 10.0.
    z = 10.0 * 100.0 + 0.0 * 1.0
    expected = 1.0 / (1.0 + math.exp(-z))
    assert result == expected


def test_predict_proba_stays_between_zero_and_one():
    model = {"weights": [100.0], "intercept": 100.0, "feature_names": ["a"]}

    assert 0.0 <= predict_proba(model, {"a": 1.0}) <= 1.0
    assert 0.0 <= predict_proba(model, {"a": -1.0}) <= 1.0


def test_compute_shap_values_is_weight_times_deviation_from_baseline():
    model = {
        "weights": [2.0, -1.0],
        "intercept": 0.5,
        "feature_names": ["a", "b"],
        "feature_means": [0.5, 3.0],
    }
    feature_vector = {"a": 1.0, "b": 3.0}

    result = compute_shap_values(model, feature_vector)

    assert result == {"a": 2.0 * (1.0 - 0.5), "b": -1.0 * (3.0 - 3.0)}


def test_compute_shap_values_sum_to_margin_minus_base_value():
    # shap_i sums to (z - base_value), the additive decomposition SHAP guarantees:
    # z = intercept + dot(weights, x); base_value = intercept + dot(weights, means).
    model = {
        "weights": [2.0, -1.0, 0.3],
        "intercept": 0.5,
        "feature_names": ["a", "b", "c"],
        "feature_means": [0.5, 3.0, -1.0],
    }
    feature_vector = {"a": 1.0, "b": 5.0, "c": 2.0}

    shap_values = compute_shap_values(model, feature_vector)

    z = model["intercept"] + sum(w * feature_vector[n] for n, w in zip(model["feature_names"], model["weights"]))
    base_value = model["intercept"] + sum(w * m for w, m in zip(model["weights"], model["feature_means"]))
    assert sum(shap_values.values()) == pytest.approx(z - base_value)


def test_top_contributing_factors_ranks_by_absolute_value_descending():
    shap_values = {"a": 0.1, "b": -0.9, "c": 0.5}

    result = top_contributing_factors(shap_values, n=2)

    assert result == [
        {"feature": "b", "shap_value": -0.9},
        {"feature": "c", "shap_value": 0.5},
    ]


def test_top_contributing_factors_respects_n():
    shap_values = {"a": 0.1, "b": -0.9, "c": 0.5, "d": 0.05}

    assert len(top_contributing_factors(shap_values, n=1)) == 1
    assert len(top_contributing_factors(shap_values, n=10)) == 4
