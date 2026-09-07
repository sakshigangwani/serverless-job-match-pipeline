"""Pure-Python logistic-regression inference — no scikit-learn at runtime.

See ml/train.py's module docstring for why: scikit-learn + numpy + scipy + xgboost
together are ~345MB unzipped, well over Lambda's 250MB function+layers limit. A fitted
logistic regression's decision function is just `sigmoid(dot(weights, x) + intercept)`
— ml/train.py's export_inference_json extracts those numbers at training time, and this
module re-implements just that arithmetic, in stdlib Python, at inference time.
tests/ml/test_train.py proves this reproduces scikit-learn's own predict_proba exactly.
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
