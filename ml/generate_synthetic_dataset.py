"""CLI: generate the synthetic demo dataset and write it to ml/data/ (PLAN.md Phase 6.3
stand-in — see ml/README.md for the real labeling workflow this replaces).

    python -m ml.generate_synthetic_dataset
"""
from __future__ import annotations

import argparse

from common.models import DEFAULT_CANDIDATE_PROFILE_PATH, CandidateProfile
from ml.dataset import DEFAULT_LABELS_PATH, DEFAULT_POSTINGS_PATH, save_dataset
from ml.synthetic_data import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", default=str(DEFAULT_CANDIDATE_PROFILE_PATH))
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate = CandidateProfile.from_file(args.profile_path)
    postings, labels = generate_dataset(candidate, n=args.n, seed=args.seed)
    save_dataset(postings, labels)

    positive_rate = sum(labels.values()) / len(labels)
    print(f"Wrote {len(postings)} postings to {DEFAULT_POSTINGS_PATH}")
    print(f"Wrote {len(labels)} labels to {DEFAULT_LABELS_PATH} ({positive_rate:.0%} positive)")


if __name__ == "__main__":
    main()
