"""Live structural checks for the native route."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from langchain_perplexity import PerplexityDecisions
from langchain_perplexity.decisions import (
    DecisionRequest,
    DecisionResponse,
    PerplexityDecisionsInvalidRequestError,
)

KEYS = ("PPLX_API_KEY", "PERPLEXITY_API_KEY")

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
    record_property: Callable[[str, object], None],
) -> None:
    response = decisions.invoke(quickstart_request)
    check_quickstart_structure(response)
    assert response.request_id is not None
    record_property("request_id", response.request_id)
    record_property("usage", response.usage.model_dump() if response.usage else None)


def test_criteria_only_noul_is_recorded(
    decisions: PerplexityDecisions,
    criteria_only_request: DecisionRequest,
    record_property: Callable[[str, object], None],
) -> None:
    try:
        response = decisions.invoke(criteria_only_request)
    except PerplexityDecisionsInvalidRequestError as error:
        record_property("criteria_only_status", error.status)
        return
    record_property("criteria_only_status", 200)
    assert 0.0 <= response.nouls["defect"].noul <= 1.0


def test_single_level_score_is_recorded(
    decisions: PerplexityDecisions,
    single_level_request: DecisionRequest,
    record_property: Callable[[str, object], None],
) -> None:
    try:
        response = decisions.invoke(single_level_request)
    except PerplexityDecisionsInvalidRequestError as error:
        record_property("single_level_status", error.status)
        return
    record_property("single_level_status", 200)
    assert response.scores["severity"].score == 0.0


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
