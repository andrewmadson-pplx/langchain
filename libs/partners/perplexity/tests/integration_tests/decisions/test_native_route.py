"""Live structural checks for the native route."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from langchain_perplexity import PerplexityDecisions
from langchain_perplexity.decisions import DecisionRequest, DecisionResponse

KEYS = ("PPLX_API_KEY", "PERPLEXITY_API_KEY")

ROUTE = "native"

pytestmark = [
    pytest.mark.live_required(*KEYS),
    pytest.mark.skipif(
        not any(os.environ.get(name) for name in KEYS),
        reason=f"none of {KEYS} is set",
    ),
]


@pytest.fixture
def decisions() -> PerplexityDecisions:
    return PerplexityDecisions(route="native")


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
    assert response.request_id is not None
    record_property("request_id", response.request_id)
    record_property("usage", response.usage.model_dump() if response.usage else None)


def test_criteria_only_noul(
    decisions: PerplexityDecisions,
    criteria_only_request: DecisionRequest,
) -> None:
    response = decisions.invoke(criteria_only_request)
    assert set(response.answers) == {"defect"}
    assert 0.0 <= response.nouls["defect"].noul <= 1.0
    assert response.request_id is not None


def test_single_level_score(
    decisions: PerplexityDecisions,
    single_level_request: DecisionRequest,
) -> None:
    response = decisions.invoke(single_level_request)
    severity = response.scores["severity"]
    assert severity.score == 0.0
    if severity.probabilities is not None:
        assert severity.probabilities == {0: 1.0}


def test_image_in_human_message(
    decisions: PerplexityDecisions,
    image_request: DecisionRequest,
    record_property: Callable[[str, object], None],
) -> None:
    response = decisions.invoke(image_request)
    color = response.choices["color"]
    assert color.choice in {"red", "blue", "green"}
    record_property("image_choice", color.choice)


async def test_async_quickstart(
    decisions: PerplexityDecisions,
    quickstart_request: DecisionRequest,
    check_quickstart_structure: Callable[[DecisionResponse], None],
) -> None:
    response = await decisions.ainvoke(quickstart_request)
    check_quickstart_structure(response)
    await decisions.aclose()
