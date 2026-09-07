import json
from unittest.mock import MagicMock

import pytest

from lambdas.extract.bedrock_client import (
    ExtractionError,
    invoke_claude,
    parse_llm_json,
)


def _mock_bedrock_runtime(response_text: str) -> MagicMock:
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"content": [{"type": "text", "text": response_text}]}
    ).encode("utf-8")
    runtime = MagicMock()
    runtime.invoke_model.return_value = {"body": body}
    return runtime


def test_invoke_claude_sends_messages_api_request_and_extracts_text():
    runtime = _mock_bedrock_runtime('{"title": "Engineer"}')

    result = invoke_claude(runtime, "anthropic.claude-3-5-sonnet", "extract this")

    assert result == '{"title": "Engineer"}'
    call_kwargs = runtime.invoke_model.call_args.kwargs
    assert call_kwargs["modelId"] == "anthropic.claude-3-5-sonnet"
    body = json.loads(call_kwargs["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["messages"] == [{"role": "user", "content": "extract this"}]
    assert body["temperature"] == 0


def test_parse_llm_json_handles_plain_json():
    assert parse_llm_json('{"title": "Engineer"}') == {"title": "Engineer"}


def test_parse_llm_json_strips_markdown_code_fence():
    text = '```json\n{"title": "Engineer"}\n```'

    assert parse_llm_json(text) == {"title": "Engineer"}


def test_parse_llm_json_strips_bare_code_fence_without_language_tag():
    text = '```\n{"title": "Engineer"}\n```'

    assert parse_llm_json(text) == {"title": "Engineer"}


def test_parse_llm_json_raises_extraction_error_on_garbage():
    with pytest.raises(ExtractionError):
        parse_llm_json("this is not json at all")
