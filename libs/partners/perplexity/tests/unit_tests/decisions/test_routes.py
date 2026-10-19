"""Route defaults match the build plan's route table."""

from __future__ import annotations

import pytest

from langchain_perplexity.decisions._routes import (
    OPENROUTER_REQUIRES_INSTRUCTIONS,
    ROUTES,
    join_url,
)


def test_openrouter_defaults() -> None:
    route = ROUTES["openrouter"]
    assert route.base_url == "https://openrouter.ai/api"
    assert route.path == "/alpha/decisions"
    assert route.url == "https://openrouter.ai/api/alpha/decisions"
    assert route.api_key_env_vars == ("OPENROUTER_API_KEY",)
    assert route.default_model == "perplexity/pplx-decider-v1.1-27b"
    assert route.ls_provider == "openrouter"
    assert route.extra_allowlist == {"provider", "session_id", "trace", "user"}
    assert route.request_id_header is None
    assert route.request_id_in_body is True
    assert route.attribution_headers is True


def test_native_defaults() -> None:
    route = ROUTES["native"]
    assert route.base_url == "https://api.perplexity.ai"
    assert route.path == "/v1/decisions"
    assert route.url == "https://api.perplexity.ai/v1/decisions"
    assert route.api_key_env_vars == ("PPLX_API_KEY", "PERPLEXITY_API_KEY")
    assert route.default_model == "pplx-decider-v1.1-27b"
    assert route.ls_provider == "perplexity"
    assert route.extra_allowlist == frozenset()
    assert route.request_id_header == "x-request-id"
    assert route.request_id_in_body is False
    assert route.attribution_headers is False


def test_only_two_routes_and_default_flag() -> None:
    assert set(ROUTES) == {"openrouter", "native"}
    assert OPENROUTER_REQUIRES_INSTRUCTIONS is False


def test_join_url() -> None:
    assert join_url("https://h/api/", "/x") == "https://h/api/x"
    assert join_url("https://h", "/x") == "https://h/x"
    with pytest.raises(ValueError, match="start with '/'"):
        join_url("https://h", "x")
