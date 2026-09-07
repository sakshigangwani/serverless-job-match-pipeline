import math

import pytest

from common.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_scaled_vectors_have_the_same_similarity_as_unscaled():
    assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(
        cosine_similarity([1.0, 1.0], [1.0, 1.0])
    )


def test_zero_vector_returns_zero_instead_of_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_dimensions_raises_value_error():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_similarity_is_bounded_between_negative_one_and_one():
    result = cosine_similarity([3.0, -1.0, 4.0], [-2.0, 5.0, 0.5])
    assert -1.0 <= result <= 1.0
    assert not math.isnan(result)
