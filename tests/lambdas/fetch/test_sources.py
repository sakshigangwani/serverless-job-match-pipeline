import json
from unittest.mock import MagicMock, patch

from lambdas.fetch.sources import RemoteOKSource

FAKE_REMOTEOK_RESPONSE = [
    {"legal": "Please link back if you use this API, thanks!"},
    {
        "id": "12345",
        "position": "Backend Engineer",
        "description": "We need a Python developer.",
        "url": "https://remoteok.com/remote-jobs/12345-backend-engineer",
    },
    {
        "id": "67890",
        "position": "Data Scientist",
        "description": "Build ML models.",
        "url": "https://remoteok.com/remote-jobs/67890-data-scientist",
    },
]


def _mock_urlopen(payload):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    return mock_response


def test_remoteok_source_skips_the_legal_notice_entry():
    with patch(
        "lambdas.fetch.sources.urllib.request.urlopen",
        return_value=_mock_urlopen(FAKE_REMOTEOK_RESPONSE),
    ):
        postings = RemoteOKSource().fetch()

    assert len(postings) == 2
    assert all(p.source == "remoteok" for p in postings)


def test_remoteok_source_maps_fields_correctly():
    with patch(
        "lambdas.fetch.sources.urllib.request.urlopen",
        return_value=_mock_urlopen(FAKE_REMOTEOK_RESPONSE),
    ):
        postings = RemoteOKSource().fetch()

    backend = next(p for p in postings if "Backend" in p.title)
    assert backend.source_url == "https://remoteok.com/remote-jobs/12345-backend-engineer"
    assert backend.description_text == "We need a Python developer."
    assert backend.raw_payload["id"] == "12345"


def test_remoteok_source_falls_back_to_generated_url_when_missing():
    payload = [{"id": "999", "position": "Engineer", "description": "desc"}]
    with patch(
        "lambdas.fetch.sources.urllib.request.urlopen",
        return_value=_mock_urlopen(payload),
    ):
        postings = RemoteOKSource().fetch()

    assert postings[0].source_url == "https://remoteok.com/remote-jobs/999"


def test_remoteok_source_returns_empty_list_for_no_postings():
    with patch(
        "lambdas.fetch.sources.urllib.request.urlopen",
        return_value=_mock_urlopen([{"legal": "only a notice"}]),
    ):
        postings = RemoteOKSource().fetch()

    assert postings == []
