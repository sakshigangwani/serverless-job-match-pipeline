"""Cosine similarity between embedding vectors (PLAN.md Phase 4).

Deliberately pure stdlib (no numpy): this keeps it shippable in the dependency-free
CommonLayer, the same way common.storage_keys is, rather than needing a compiled-
dependency layer like PydanticLayer.
"""
from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
