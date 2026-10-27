"""Live structural checks for the openrouter route."""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx
import pytest

from langchain_perplexity import PerplexityDecisions
from langchain_perplexity.decisions import DecisionRequest, DecisionResponse

KEY = "OPENROUTER_API_KEY"

ROUTE = "openrouter"

pytestmark = [
    pytest.mark.live_required(KEY),
    pytest.mark.skipif(not os.environ.get(KEY), reason=f"{KEY} is not set"),
]


@pytest.fixture
def decisions() -> PerplexityDecisions:
    return PerplexityDecisions(route="openrouter")


def test_quickstart(
    decisions: PerplexityDecisions,
    quickstart_request: DecisionRequest,
    check_quickstart_structure: Callable[[DecisionResponse], None],
    recorded_response: Callable[[str, str], dict[str, object]],
    check_same_top_level_shape: Callable[[DecisionResponse, dict[str, object]], None],
    record_property: Callable[[str, object], None],
) -> None:
    response = decisions.invoke(quickstart_request)
    check_quickstart_structure(response)
    check_same_top_level_shape(response, recorded_response("row12_baseline", ROUTE))
    record_property("request_id", response.request_id)
    record_property("provider", response.provider)
    record_property("usage", response.usage.model_dump() if response.usage else None)


def test_criteria_only_noul_is_rejected_locally(
    criteria_only_request: DecisionRequest,
) -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answers": {}})

    decisions = PerplexityDecisions(
        route="openrouter",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    with pytest.raises(ValueError, match="openrouter route requires `instructions`"):
        decisions.invoke(criteria_only_request)
    assert seen == []


def test_single_level_score(
    decisions: PerplexityDecisions,
    single_level_request: DecisionRequest,
) -> None:
    response = decisions.invoke(single_level_request)
    severity = response.scores["severity"]
    assert severity.score == 0.0
    if severity.probabilities is not None:
        assert severity.probabilities == {0: 1.0}


def test_image_in_top_level_state(
    decisions: PerplexityDecisions,
    image_request: DecisionRequest,
    image_data_url: str,
    record_property: Callable[[str, object], None],
) -> None:
    response = decisions.invoke(
        {
            "state": [
                "Which color is the square?",
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
            "questions": image_request["questions"],
        }
    )
    color = response.choices["color"]
    assert color.choice in {"red", "blue", "green"}
    record_property("image_choice", color.choice)


def test_nested_image_is_rejected_locally(image_request: DecisionRequest) -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answers": {}})

    decisions = PerplexityDecisions(
        route="openrouter",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    with pytest.raises(ValueError, match="top-level `state` items"):
        decisions.invoke(image_request)
    assert seen == []


async def test_async_quickstart(
    decisions: PerplexityDecisions,
    quickstart_request: DecisionRequest,
    check_quickstart_structure: Callable[[DecisionResponse], None],
) -> None:
    response = await decisions.ainvoke(quickstart_request)
    check_quickstart_structure(response)
    await decisions.aclose()
