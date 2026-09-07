"""Job-board source adapters (PLAN.md Phase 2.1).

`JobSource` is the adapter interface: a new job board becomes a new subclass with a
`fetch()` method, without the Lambda handler needing to change. `RemoteOKSource` is the
one real source wired up for the core pipeline.
"""
from __future__ import annotations

import abc
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawPosting:
    """A posting as pulled from a source, before any LLM extraction happens."""

    source: str
    source_url: str
    title: str
    description_text: str
    raw_payload: dict[str, Any] = field(default_factory=dict)


class JobSource(abc.ABC):
    name: str

    @abc.abstractmethod
    def fetch(self) -> list[RawPosting]:
        """Return the current set of postings from this source."""


class RemoteOKSource(JobSource):
    """https://remoteok.com/api — free, no API key required.

    The API's first array element is a legal/attribution notice, not a job posting
    (it has no "id" field) — it's filtered out below.
    """

    name = "remoteok"
    API_URL = "https://remoteok.com/api"
    TIMEOUT_SECONDS = 10

    def fetch(self) -> list[RawPosting]:
        request = urllib.request.Request(
            self.API_URL, headers={"User-Agent": "JobPulse/1.0"}
        )
        with urllib.request.urlopen(request, timeout=self.TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))

        postings = []
        for item in payload:
            if not isinstance(item, dict) or "id" not in item:
                continue
            postings.append(
                RawPosting(
                    source=self.name,
                    source_url=item.get(
                        "url", f"https://remoteok.com/remote-jobs/{item['id']}"
                    ),
                    title=item.get("position", ""),
                    description_text=item.get("description", ""),
                    raw_payload=item,
                )
            )
        return postings
