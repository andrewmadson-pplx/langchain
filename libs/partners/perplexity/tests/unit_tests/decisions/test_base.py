"""`PerplexityDecisions` configuration and transport behavior."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any, Literal

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from langchain_perplexity import PerplexityDecisions, __version__
from langchain_perplexity.decisions import (
    DecisionRequest,
    DecisionsEndpoint,
    Noul,
    PerplexityDecisionsServerError,
    PerplexityDecisionsTimeoutError,
)

KEY_VARS = [
    "OPENROUTER_API_KEY",
    "PPLX_API_KEY",
    "PERPLEXITY_API_KEY",
    "OPENROUTER_APP_URL",
    "OPENROUTER_APP_TITLE",
]
OK_BODY = {"answers": {"q": {"type": "noul", "noul": 0.5}}}
REQUEST: DecisionRequest = {
    "state": "hello",
    "questions": {"q": Noul(instructions="Is it a greeting?")},
}


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)


class Recorder:
    """Mock transport that records requests and returns a canned response."""

    def __init__(
        self, respond: Callable[[httpx.Request], httpx.Response] | None = None
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond or (lambda _: httpx.Response(200, json=OK_BODY))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


@pytest.fixture
def recorder() -> Iterator[Recorder]:
    yield Recorder()


def _payload(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_openrouter_defaults(recorder: Recorder) -> None:
    decisions = PerplexityDecisions(api_key="or-key", client=recorder.client())
    response = decisions.invoke(REQUEST)

    assert response.nouls["q"].noul == 0.5
    [request] = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer or-key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == f"langchain-perplexity/{__version__}"
    assert request.headers["HTTP-Referer"] == "https://docs.langchain.com"
    assert request.headers["X-Title"] == "LangChain"
    assert _payload(request) == {
        "model": "perplexity/pplx-decider-v1.1-27b",
        "state": "hello",
        "questions": {"q": {"type": "noul", "instructions": "Is it a greeting?"}},
    }
    assert decisions.ls_provider == "openrouter"
    assert decisions.lc_secrets == {"api_key": "OPENROUTER_API_KEY"}


def test_native_defaults(recorder: Recorder) -> None:
    decisions = PerplexityDecisions(
        route="native", api_key="pplx-key", client=recorder.client()
    )
    decisions.invoke(REQUEST)

    [request] = recorder.requests
    assert str(request.url) == "https://api.perplexity.ai/v1/decisions"
    assert request.headers["Authorization"] == "Bearer pplx-key"
    assert request.headers["User-Agent"] == f"langchain-perplexity/{__version__}"
    assert "HTTP-Referer" not in request.headers
    assert "X-Title" not in request.headers
    assert _payload(request)["model"] == "pplx-decider-v1.1-27b"
    assert decisions.ls_provider == "perplexity"
    assert decisions.lc_secrets == {"api_key": "PPLX_API_KEY"}


def test_explicit_fields_win_over_route_defaults(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    decisions = PerplexityDecisions(
        model=" custom/model ",
        api_key="explicit",
        app_url="https://example.com/app",
        app_title="My App",
        client=recorder.client(),
    )
    decisions.invoke(REQUEST)

    [request] = recorder.requests
    assert request.headers["Authorization"] == "Bearer explicit"
    assert request.headers["HTTP-Referer"] == "https://example.com/app"
    assert request.headers["X-Title"] == "My App"
    assert _payload(request)["model"] == "custom/model"


def test_attribution_env_overrides(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_APP_URL", "https://env.example.com")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Env Title")
    PerplexityDecisions(api_key="k", client=recorder.client()).invoke(REQUEST)

    [request] = recorder.requests
    assert request.headers["HTTP-Referer"] == "https://env.example.com"
    assert request.headers["X-Title"] == "Env Title"


@pytest.mark.parametrize(
    ("route", "env_var", "other_vars"),
    [
        ("openrouter", "OPENROUTER_API_KEY", ["PPLX_API_KEY", "PERPLEXITY_API_KEY"]),
        ("native", "PPLX_API_KEY", ["OPENROUTER_API_KEY"]),
        ("native", "PERPLEXITY_API_KEY", ["OPENROUTER_API_KEY"]),
    ],
)
def test_key_from_route_env_only(
    route: Literal["openrouter", "native"],
    env_var: str,
    other_vars: list[str],
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in other_vars:
        monkeypatch.setenv(name, "wrong-route-key")
    with pytest.raises(ValidationError, match=env_var) as exc_info:
        PerplexityDecisions(route=route)
    for name in other_vars:
        assert name not in str(exc_info.value)

    monkeypatch.setenv(env_var, "right-key")
    decisions = PerplexityDecisions(route=route, client=recorder.client())
    decisions.invoke(REQUEST)
    assert recorder.requests[0].headers["Authorization"] == "Bearer right-key"


def test_native_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_API_KEY", "first")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "second")
    decisions = PerplexityDecisions(route="native")
    assert isinstance(decisions.api_key, SecretStr)
    assert decisions.api_key.get_secret_value() == "first"


def test_extra_allowlist_and_collisions(recorder: Recorder) -> None:
    decisions = PerplexityDecisions(
        api_key="k",
        client=recorder.client(),
        extra={"user": "u-1", "provider": {"order": ["perplexity"]}},
    )
    decisions.invoke(REQUEST)
    payload = _payload(recorder.requests[0])
    assert payload["user"] == "u-1"
    assert payload["provider"] == {"order": ["perplexity"]}
    assert set(payload) == {"model", "state", "questions", "user", "provider"}

    with pytest.raises(ValidationError, match="reserved"):
        PerplexityDecisions(api_key="k", extra={"model": "other"})
    with pytest.raises(ValidationError, match="not allowed on the openrouter route"):
        PerplexityDecisions(api_key="k", extra={"temperature": 0})
    with pytest.raises(ValidationError, match="native route does not accept"):
        PerplexityDecisions(route="native", api_key="k", extra={"user": "u-1"})
    assert len(recorder.requests) == 1


def test_exactly_one_request_on_error() -> None:
    recorder = Recorder(lambda _: httpx.Response(503, json={"error": "down"}))
    decisions = PerplexityDecisions(api_key="k", client=recorder.client())
    with pytest.raises(PerplexityDecisionsServerError):
        decisions.invoke(REQUEST)
    assert len(recorder.requests) == 1


def test_timeout_maps_to_timeout_error_without_leaking() -> None:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    recorder = Recorder(raise_timeout)
    decisions = PerplexityDecisions(api_key="top-secret", client=recorder.client())
    with pytest.raises(PerplexityDecisionsTimeoutError) as exc_info:
        decisions.invoke(REQUEST)
    assert "top-secret" not in str(exc_info.value)
    assert "openrouter.ai/api/alpha/decisions" in str(exc_info.value)
    assert len(recorder.requests) == 1


def test_endpoint_requires_explicit_key_and_is_custom(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint = DecisionsEndpoint(base_url="https://gw.example.com/api/", path="/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    with pytest.raises(ValidationError, match="explicit `api_key`"):
        PerplexityDecisions(endpoint=endpoint)

    decisions = PerplexityDecisions(
        endpoint=endpoint, api_key="gw-key", client=recorder.client()
    )
    decisions.invoke(REQUEST)
    assert str(recorder.requests[0].url) == "https://gw.example.com/api/d"
    assert recorder.requests[0].headers["Authorization"] == "Bearer gw-key"
    assert decisions.ls_provider == "custom"
    assert decisions.url == "https://gw.example.com/api/d"


def test_redirect_is_an_error_and_not_followed() -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        if request.url.host == "openrouter.ai":
            return httpx.Response(
                307, headers={"Location": "https://evil.example/alpha/decisions"}
            )
        return httpx.Response(200, json=OK_BODY)

    recorder = Recorder(redirect)
    injected = httpx.Client(
        transport=httpx.MockTransport(recorder), follow_redirects=True
    )
    decisions = PerplexityDecisions(api_key="k", client=injected)
    with pytest.raises(PerplexityDecisionsServerError, match="Redirects are disabled"):
        decisions.invoke(REQUEST)
    assert [r.url.host for r in recorder.requests] == ["openrouter.ai"]


def test_created_clients_are_closed_and_injected_clients_are_not(
    recorder: Recorder,
) -> None:
    injected = recorder.client()
    with_injected = PerplexityDecisions(api_key="k", client=injected)
    with_injected.invoke(REQUEST)
    with_injected.close()
    assert not injected.is_closed

    owned = PerplexityDecisions(api_key="k", timeout=7.5)
    created = owned._sync_client()
    assert created.timeout == httpx.Timeout(7.5)
    assert created.follow_redirects is False
    owned.close()
    assert created.is_closed
    assert owned._owned_client is None


async def test_async_clients_follow_the_same_ownership_rules(
    recorder: Recorder,
) -> None:
    injected = recorder.async_client()
    with_injected = PerplexityDecisions(api_key="k", async_client=injected)
    response = await with_injected.ainvoke(REQUEST)
    assert response.nouls["q"].noul == 0.5
    await with_injected.aclose()
    assert not injected.is_closed

    owned = PerplexityDecisions(api_key="k")
    created = owned._async_client()
    assert created.follow_redirects is False
    await owned.aclose()
    assert created.is_closed


def test_repr_and_str_hide_key_and_clients(recorder: Recorder) -> None:
    decisions = PerplexityDecisions(api_key="hidden-key", client=recorder.client())
    for text in (repr(decisions), str(decisions)):
        assert "hidden-key" not in text
        assert "client" not in text


def test_model_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        PerplexityDecisions(api_key="k", model="  ")
