import json
from unittest.mock import MagicMock

from lambdas.embed.bedrock_embeddings import embed_text


def _mock_bedrock_runtime(embedding: list[float]) -> MagicMock:
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"embedding": embedding, "inputTextTokenCount": 7}
    ).encode("utf-8")
    runtime = MagicMock()
    runtime.invoke_model.return_value = {"body": body}
    return runtime


def test_embed_text_returns_the_embedding_vector():
    runtime = _mock_bedrock_runtime([0.1, 0.2, 0.3])

    result = embed_text(runtime, "amazon.titan-embed-text-v2:0", "some text")

    assert result == [0.1, 0.2, 0.3]


def test_embed_text_sends_only_input_text_no_model_specific_params():
    runtime = _mock_bedrock_runtime([0.1])

    embed_text(runtime, "amazon.titan-embed-text-v2:0", "some text")

    call_kwargs = runtime.invoke_model.call_args.kwargs
    assert call_kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(call_kwargs["body"])
    assert body == {"inputText": "some text"}
