from datetime import date

from common.storage_keys import (
    compute_posting_hash,
    raw_posting_key,
    structured_posting_key,
)


def test_posting_hash_is_deterministic():
    h1 = compute_posting_hash("remoteok", "https://example.com/job/1", "We need a Python dev")
    h2 = compute_posting_hash("remoteok", "https://example.com/job/1", "We need a Python dev")

    assert h1 == h2


def test_posting_hash_changes_when_description_changes():
    h1 = compute_posting_hash("remoteok", "https://example.com/job/1", "Original text")
    h2 = compute_posting_hash("remoteok", "https://example.com/job/1", "Edited text")

    assert h1 != h2


def test_posting_hash_changes_when_source_url_changes():
    h1 = compute_posting_hash("remoteok", "https://example.com/job/1", "Same text")
    h2 = compute_posting_hash("remoteok", "https://example.com/job/2", "Same text")

    assert h1 != h2


def test_raw_posting_key_convention_with_date_object():
    key = raw_posting_key("remoteok", date(2026, 9, 6), "deadbeef")

    assert key == "raw/remoteok/2026-09-06/deadbeef.json"


def test_raw_posting_key_convention_with_string_date():
    key = raw_posting_key("adzuna", "2026-09-06", "deadbeef")

    assert key == "raw/adzuna/2026-09-06/deadbeef.json"


def test_structured_posting_key_convention_with_date_object():
    key = structured_posting_key("remoteok", date(2026, 9, 6), "deadbeef")

    assert key == "structured/remoteok/2026-09-06/deadbeef.json"


def test_structured_posting_key_uses_a_different_prefix_than_raw():
    raw_key = raw_posting_key("remoteok", date(2026, 9, 6), "deadbeef")
    structured_key = structured_posting_key("remoteok", date(2026, 9, 6), "deadbeef")

    assert raw_key != structured_key
    assert raw_key.startswith("raw/")
    assert structured_key.startswith("structured/")
