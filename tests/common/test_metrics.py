import json

from common.metrics import NAMESPACE, emit_metric


def test_emit_metric_prints_valid_json(capsys):
    emit_metric("PostingsIngested", 5)

    document = json.loads(capsys.readouterr().out)
    assert document["PostingsIngested"] == 5


def test_emit_metric_uses_the_jobpulse_namespace(capsys):
    emit_metric("PostingsIngested", 5)

    document = json.loads(capsys.readouterr().out)
    assert document["_aws"]["CloudWatchMetrics"][0]["Namespace"] == NAMESPACE


def test_emit_metric_declares_the_metric_name_and_unit(capsys):
    emit_metric("AlertsSent", 1, unit="Count")

    document = json.loads(capsys.readouterr().out)
    metric = document["_aws"]["CloudWatchMetrics"][0]["Metrics"][0]
    assert metric == {"Name": "AlertsSent", "Unit": "Count"}


def test_emit_metric_includes_dimensions_as_top_level_fields(capsys):
    emit_metric("PostingsIngested", 5, dimensions={"Source": "remoteok"})

    document = json.loads(capsys.readouterr().out)
    assert document["Source"] == "remoteok"
    assert document["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Source"]]


def test_emit_metric_with_no_dimensions_declares_an_empty_dimension_set(capsys):
    emit_metric("ExtractionFailures", 1)

    document = json.loads(capsys.readouterr().out)
    assert document["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [[]]


def test_emit_metric_includes_a_millisecond_timestamp(capsys):
    emit_metric("PostingScore", 0.87)

    document = json.loads(capsys.readouterr().out)
    timestamp = document["_aws"]["Timestamp"]
    assert isinstance(timestamp, int)
    assert timestamp > 1_700_000_000_000  # sanity: after 2023-11-14
