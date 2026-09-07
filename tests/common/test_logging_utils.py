import json
import logging

from common.logging_utils import get_logger, log_event


def test_get_logger_sets_info_level():
    logger = get_logger("test.logger.info")

    assert logger.level == logging.INFO


def test_log_event_emits_valid_json(caplog):
    logger = get_logger("test.logger.json")

    with caplog.at_level(logging.INFO, logger="test.logger.json"):
        log_event(logger, "did a thing")

    payload = json.loads(caplog.records[0].message)
    assert payload["message"] == "did a thing"


def test_log_event_includes_posting_id_when_given(caplog):
    logger = get_logger("test.logger.posting_id")

    with caplog.at_level(logging.INFO, logger="test.logger.posting_id"):
        log_event(logger, "processed", posting_id="abc123")

    payload = json.loads(caplog.records[0].message)
    assert payload["posting_id"] == "abc123"


def test_log_event_omits_posting_id_when_not_given(caplog):
    logger = get_logger("test.logger.no_posting_id")

    with caplog.at_level(logging.INFO, logger="test.logger.no_posting_id"):
        log_event(logger, "batch summary")

    payload = json.loads(caplog.records[0].message)
    assert "posting_id" not in payload


def test_log_event_includes_extra_fields(caplog):
    logger = get_logger("test.logger.extra")

    with caplog.at_level(logging.INFO, logger="test.logger.extra"):
        log_event(logger, "fetched", source="remoteok", count=5)

    payload = json.loads(caplog.records[0].message)
    assert payload["source"] == "remoteok"
    assert payload["count"] == 5
