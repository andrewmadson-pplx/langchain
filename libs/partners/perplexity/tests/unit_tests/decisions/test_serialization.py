"""`dumpd` and `load` round trips for both routes."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
import pytest
from langchain_core.load import dumpd, load

from langchain_perplexity import PerplexityDecisions
from langchain_perplexity.decisions import DecisionRequest, Noul

OK_BODY = {"answers": {"q": {"type": "noul", "noul": 0.5}}}
REQUEST: DecisionRequest = {
    "state": "hello",
    "questions": {"q": Noul(instructions="Greeting?")},
}


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["OPENROUTER_API_KEY", "PPLX_API_KEY", "PERPLEXITY_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


def _load(serialized: dict[str, Any], **kwargs: Any) -> PerplexityDecisions:
    restored = load(serialized, allowed_objects=[PerplexityDecisions], **kwargs)
    assert isinstance(restored, PerplexityDecisions)
    return restored


def _invoke_with_mock(decisions: PerplexityDecisions) -> httpx.Request:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK_BODY)

    decisions.client = httpx.Client(transport=httpx.MockTransport(respond))
    decisions.invoke(REQUEST)
    return seen[0]


def test_dumpd_excludes_key_value_and_clients() -> None:
    decisions = PerplexityDecisions(
        route="native",
        api_key="very-secret",
        model="pplx-decider-v1.2",
        client=httpx.Client(),
        async_client=httpx.AsyncClient(),
    )
    serialized = dumpd(decisions)
    text = json.dumps(serialized)
    assert "very-secret" not in text
    assert "client" not in serialized["kwargs"]
    assert "async_client" not in serialized["kwargs"]
    assert serialized["kwargs"]["api_key"] == {
        "lc": 1,
        "type": "secret",
        "id": ["PPLX_API_KEY"],
    }
    assert serialized["kwargs"]["route"] == "native"
    assert serialized["id"][-1] == "PerplexityDecisions"


@pytest.mark.parametrize(
    ("route", "secret_id", "host"),
    [
        ("openrouter", "OPENROUTER_API_KEY", "openrouter.ai"),
        ("native", "PPLX_API_KEY", "api.perplexity.ai"),
    ],
)
def test_load_with_secrets_map_restores_working_instance(
    route: Literal["openrouter", "native"], secret_id: str, host: str
) -> None:
    original = PerplexityDecisions(route=route, api_key="original")
    restored = _load(dumpd(original), secrets_map={secret_id: "restored-key"})

    assert restored.route == route
    assert restored.client is None
    assert restored.async_client is None
    request = _invoke_with_mock(restored)
    assert request.url.host == host
    assert request.headers["Authorization"] == "Bearer restored-key"


def test_native_restore_with_only_perplexity_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PerplexityDecisions(route="native", api_key="original")
    serialized = dumpd(original)

    monkeypatch.setenv("PERPLEXITY_API_KEY", "alias-key")
    restored = _load(serialized, secrets_from_env=True)
    request = _invoke_with_mock(restored)
    assert request.headers["Authorization"] == "Bearer alias-key"

    with pytest.raises(ValueError, match="PPLX_API_KEY or PERPLEXITY_API_KEY"):
        monkeypatch.delenv("PERPLEXITY_API_KEY")
        _load(serialized)
