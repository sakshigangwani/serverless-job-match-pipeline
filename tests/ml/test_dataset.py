from common.models import CandidateProfile
from ml.dataset import load_dataset, save_dataset
from ml.synthetic_data import generate_dataset

CANDIDATE = CandidateProfile.from_file("common/candidate_profile.example.json")


def test_save_and_load_round_trips_postings_and_labels(tmp_path):
    postings, labels = generate_dataset(CANDIDATE, n=20, seed=42)
    postings_path = tmp_path / "postings.json"
    labels_path = tmp_path / "labels.csv"

    save_dataset(postings, labels, postings_path=postings_path, labels_path=labels_path)
    restored_postings, restored_labels = load_dataset(
        postings_path=postings_path, labels_path=labels_path
    )

    assert [p.posting_id for p in restored_postings] == [p.posting_id for p in postings]
    assert restored_labels == labels


def test_save_creates_parent_directories(tmp_path):
    postings, labels = generate_dataset(CANDIDATE, n=5, seed=42)
    nested_postings_path = tmp_path / "nested" / "dir" / "postings.json"
    nested_labels_path = tmp_path / "nested" / "dir" / "labels.csv"

    save_dataset(
        postings, labels, postings_path=nested_postings_path, labels_path=nested_labels_path
    )

    assert nested_postings_path.exists()
    assert nested_labels_path.exists()


def test_labels_csv_has_expected_header(tmp_path):
    postings, labels = generate_dataset(CANDIDATE, n=5, seed=42)
    labels_path = tmp_path / "labels.csv"

    save_dataset(postings, labels, postings_path=tmp_path / "postings.json", labels_path=labels_path)

    header = labels_path.read_text().splitlines()[0]
    assert header == "posting_id,label"


def test_loaded_postings_preserve_embedding_similarity(tmp_path):
    postings, labels = generate_dataset(CANDIDATE, n=5, seed=42)
    postings_path = tmp_path / "postings.json"
    labels_path = tmp_path / "labels.csv"

    save_dataset(postings, labels, postings_path=postings_path, labels_path=labels_path)
    restored_postings, _ = load_dataset(postings_path=postings_path, labels_path=labels_path)

    for original, restored in zip(postings, restored_postings):
        assert restored.embedding_similarity == original.embedding_similarity
