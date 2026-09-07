"""Custom CloudWatch metrics via Embedded Metric Format, or "EMF" (PLAN.md Phase 9.2).

Printing a correctly-shaped JSON line to stdout is enough for CloudWatch Logs to
automatically extract it into a real CloudWatch metric — no `cloudwatch:PutMetricData`
IAM permission, no extra API call, no Lambda extension or agent needed. The official
`aws-embedded-metrics` PyPI package wraps this exact JSON shape; it's hand-rolled here
instead, to avoid another dependency/layer for something this small (the same reasoning
that kept the fetch Lambda dependency-free in Phase 2, and the threshold Lambda's
inference pure-Python in Phase 7).

emit_metric() only builds and prints the EMF document — CloudWatch's own aggregation
(the Average/Max/Sum/etc. statistic a dashboard or alarm picks) is what turns many
individual emitted values (e.g. one PostingScore per posting) into "the average score,"
not any averaging done here.
"""
from __future__ import annotations

import json
import time

NAMESPACE = "JobPulse"


def emit_metric(
    metric_name: str,
    value: float,
    unit: str = "Count",
    dimensions: dict[str, str] | None = None,
) -> None:
    dimensions = dimensions or {}
    document = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [list(dimensions.keys())],
                    "Metrics": [{"Name": metric_name, "Unit": unit}],
                }
            ],
        },
        metric_name: value,
        **dimensions,
    }
    print(json.dumps(document))
