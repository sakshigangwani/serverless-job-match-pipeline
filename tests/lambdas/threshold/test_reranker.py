import json
import math

from lambdas.threshold.reranker import load_model, predict_proba


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
