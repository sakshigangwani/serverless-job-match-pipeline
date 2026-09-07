"""Pure-Python logistic-regression inference — no scikit-learn at runtime.

See ml/train.py's module docstring for why: scikit-learn + numpy + scipy + xgboost
together are ~345MB unzipped, well over Lambda's 250MB function+layers limit. A fitted
logistic regression's decision function is just `sigmoid(dot(weights, x) + intercept)`
— ml/train.py's export_inference_json extracts those numbers at training time, and this
module re-implements just that arithmetic, in stdlib Python, at inference time.
tests/ml/test_train.py proves this reproduces scikit-learn's own predict_proba exactly.

compute_shap_values (PLAN.md Phase 13) applies the same pattern to explainability: for a
linear model with independent features, SHAP values in log-odds space are exactly
`weight_i * (x_i - baseline_i)`, with `baseline_i` the feature's training-set mean
(exported as `feature_means`, alongside weights/intercept, by the same training step).
That's the closed-form result `shap.LinearExplainer` itself computes for a linear model
— tests/ml/test_train.py proves this function's output matches shap's directly, so no
`shap` import (and none of its own dependencies) is needed at Lambda runtime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_model(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def predict_proba(model: dict, feature_vector: dict[str, float]) -> float:
    """Score one posting's feature vector against the exported model. Uses
    `model["feature_names"]` for column order, not FEATURE_NAMES directly — the model
    is self-describing (PLAN.md Phase 6.5) precisely so this can never silently
    misalign weights to the wrong feature.
    """
    features = [feature_vector[name] for name in model["feature_names"]]
    z = model["intercept"] + sum(w * f for w, f in zip(model["weights"], features))
    return 1.0 / (1.0 + math.exp(-z))


def compute_shap_values(model: dict, feature_vector: dict[str, float]) -> dict[str, float]:
    """Exact (not sampled) per-feature SHAP values, in log-odds space, for the linear
    re-ranker: `shap_i = weight_i * (x_i - feature_means[i])`. These sum to
    `z - base_value` (the margin minus the training-mean prediction), the same additive
    decomposition `shap.LinearExplainer` guarantees. See module docstring.
    """
    return {
        name: weight * (feature_vector[name] - mean)
        for name, weight, mean in zip(model["feature_names"], model["weights"], model["feature_means"])
    }


def top_contributing_factors(shap_values: dict[str, float], n: int = 3) -> list[dict[str, float | str]]:
    """The `n` features whose SHAP value magnitude most influenced the score, ranked
    most-influential first (sign preserved, so a caller can tell "helped" from "hurt").
    """
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": name, "shap_value": value} for name, value in ranked[:n]]
