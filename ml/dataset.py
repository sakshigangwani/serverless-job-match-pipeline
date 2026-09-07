"""Generic labeled-dataset I/O (PLAN.md Phase 6.3): save/load a list of Postings plus
their posting_id -> label map. Not synthetic-specific — this is the same format a real,
manually-labeled dataset would use (see ml/README.md for that workflow); only
ml/synthetic_data.py's *generator* is synthetic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from common.models import Posting

DEFAULT_POSTINGS_PATH = Path(__file__).resolve().parent / "data" / "postings.json"
DEFAULT_LABELS_PATH = Path(__file__).resolve().parent / "data" / "labels.csv"

# Shared by ml/train.py and ml/evaluate.py so both scripts reproduce the exact same
# train/test split from the same dataset files — the re-ranker must be evaluated on the
# same held-out postings the baselines are compared against for the comparison to mean
# anything (spec section 3.3's "improvement delta").
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42


def save_dataset(
    postings: list[Posting],
    labels: dict[str, int],
    postings_path: Path = DEFAULT_POSTINGS_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
) -> None:
    postings_path.parent.mkdir(parents=True, exist_ok=True)
    postings_path.write_text(
        json.dumps([json.loads(p.model_dump_json()) for p in postings], indent=2)
    )

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["posting_id", "label"])
        for posting_id, label in labels.items():
            writer.writerow([posting_id, label])


def load_dataset(
    postings_path: Path = DEFAULT_POSTINGS_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
) -> tuple[list[Posting], dict[str, int]]:
    postings = [Posting.model_validate(item) for item in json.loads(postings_path.read_text())]

    labels: dict[str, int] = {}
    with labels_path.open(newline="") as f:
        for row in csv.DictReader(f):
            labels[row["posting_id"]] = int(row["label"])

    return postings, labels


def split_dataset(
    postings: list[Posting],
    labels: dict[str, int],
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[list[Posting], list[Posting], dict[str, int], dict[str, int]]:
    """Stratified train/test split (preserves the good-fit/not-a-fit ratio in both
    halves) so a small or imbalanced dataset doesn't accidentally starve one split of
    positive examples.
    """
    y = [labels[p.posting_id] for p in postings]
    train_postings, test_postings = train_test_split(
        postings, test_size=test_size, random_state=random_state, stratify=y
    )
    train_labels = {p.posting_id: labels[p.posting_id] for p in train_postings}
    test_labels = {p.posting_id: labels[p.posting_id] for p in test_postings}
    return train_postings, test_postings, train_labels, test_labels
