"""Existing classes ignore `OPENROUTER_API_KEY` and keep their request shape."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_perplexity import (
    ChatPerplexity,
    PerplexityEmbeddings,
    PerplexitySearchResults,
    PerplexitySearchRetriever,
)

KEY_SOURCES = {
    "explicit": ({}, {"api_key": "explicit-key"}),
    "pplx_env": ({"PPLX_API_KEY": "pplx-env-key"}, {}),
    "both_env": (
        {"PPLX_API_KEY": "pplx-env-key", "PERPLEXITY_API_KEY": "perplexity-env-key"},
        {},
    ),
    "perplexity_env_only": ({"PERPLEXITY_API_KEY": "perplexity-env-key"}, {}),
}


class _Captured(Exception):
    """Raised by mocked SDK calls after the request payload is recorded."""


def _capture(target: Any, name: str) -> MagicMock:
    mock = MagicMock(side_effect=_Captured)
    setattr(target, name, mock)
    return mock


def _client_fingerprint(client: Any) -> dict[str, Any]:
    return {
        "base_url": str(client.base_url),
        "api_key": client.api_key,
        "auth_headers": dict(client.auth_headers),
    }


def _fingerprint(kwargs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    chat = ChatPerplexity(**kwargs)
    create = _capture(chat.client.chat.completions, "create")
    with pytest.raises(_Captured):
        chat.invoke("hello")
    result["chat"] = {
        **_client_fingerprint(chat.client),
        "model": chat.model,
        "payload": create.call_args.kwargs,
    }

    embeddings = PerplexityEmbeddings(**kwargs)
    create = _capture(embeddings.client.embeddings, "create")
    with pytest.raises(_Captured):
        embeddings.embed_query("hello")
    result["embeddings"] = {
        **_client_fingerprint(embeddings.client),
        "model": embeddings.model,
        "payload": create.call_args.kwargs,
    }

    retriever_kwargs = {"pplx_api_key": v for v in kwargs.values()}
    retriever = PerplexitySearchRetriever(**retriever_kwargs)
    create = _capture(retriever.client.search, "create")
    with pytest.raises(_Captured):
        retriever.invoke("hello")
    result["retriever"] = {
        **_client_fingerprint(retriever.client),
        "payload": create.call_args.kwargs,
    }

    tool = PerplexitySearchResults(**retriever_kwargs)
    create = _capture(tool.client.search, "create")
    tool.invoke("hello")
    result["tool"] = {
        **_client_fingerprint(tool.client),
        "payload": create.call_args.kwargs,
    }
    return result


@pytest.mark.parametrize("key_source", list(KEY_SOURCES))
def test_openrouter_key_does_not_change_existing_classes(
    key_source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, kwargs = KEY_SOURCES[key_source]
    for name in ["OPENROUTER_API_KEY", "PPLX_API_KEY", "PERPLEXITY_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    without_openrouter = _fingerprint(kwargs)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy")
    with_openrouter = _fingerprint(kwargs)

    assert with_openrouter == without_openrouter
    for fingerprint in without_openrouter.values():
        assert fingerprint["base_url"].rstrip("/") == "https://api.perplexity.ai"
        assert "sk-or-dummy" not in str(fingerprint)
    assert without_openrouter["chat"]["model"] == "sonar"
    assert without_openrouter["embeddings"]["model"] == "pplx-embed-v1-4b"
