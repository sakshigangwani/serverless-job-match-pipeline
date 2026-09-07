from common.models import CandidateProfile
from ml.synthetic_data import generate_dataset

CANDIDATE = CandidateProfile.from_file("common/candidate_profile.example.json")


def test_generation_is_deterministic_given_the_same_seed():
    postings_a, labels_a = generate_dataset(CANDIDATE, n=50, seed=42)
    postings_b, labels_b = generate_dataset(CANDIDATE, n=50, seed=42)

    assert [p.model_dump() for p in postings_a] == [p.model_dump() for p in postings_b]
    assert labels_a == labels_b


def test_different_seeds_produce_different_datasets():
    postings_a, _ = generate_dataset(CANDIDATE, n=50, seed=1)
    postings_b, _ = generate_dataset(CANDIDATE, n=50, seed=2)

    assert postings_a != postings_b


def test_generates_the_requested_number_of_postings():
    postings, labels = generate_dataset(CANDIDATE, n=37, seed=42)

    assert len(postings) == 37
    assert len(labels) == 37


def test_every_posting_has_a_label():
    postings, labels = generate_dataset(CANDIDATE, n=50, seed=42)

    assert {p.posting_id for p in postings} == set(labels)


def test_labels_are_binary():
    _postings, labels = generate_dataset(CANDIDATE, n=50, seed=42)

    assert set(labels.values()) <= {0, 1}


def test_dataset_is_not_degenerately_one_sided():
    _postings, labels = generate_dataset(CANDIDATE, n=200, seed=42)

    positive_rate = sum(labels.values()) / len(labels)
    assert 0.25 <= positive_rate <= 0.75


def test_postings_have_valid_embedding_similarity_range():
    postings, _labels = generate_dataset(CANDIDATE, n=50, seed=42)

    for posting in postings:
        assert -1.0 <= posting.embedding_similarity <= 1.0


def test_postings_are_valid_synthetic_source():
    postings, _labels = generate_dataset(CANDIDATE, n=10, seed=42)

    assert all(p.source == "synthetic" for p in postings)


def test_description_text_contains_the_required_skills_for_bm25_signal():
    postings, _labels = generate_dataset(CANDIDATE, n=10, seed=42)

    for posting in postings:
        for skill in posting.required_skills:
            assert skill in posting.description_text
