"""Validation rules and wire serialization of question and answer types."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from langchain_perplexity.decisions import (
    Choice,
    DecisionRequest,
    DecisionResponse,
    DecisionsEndpoint,
    Noul,
    NoulCriteria,
    Score,
    _routes,
)
from langchain_perplexity.decisions._client import build_payload
from langchain_perplexity.decisions._routes import ROUTES

QUICKSTART_PAYLOAD = (
    '{"model":"pplx-decider-v1.1-27b",'
    '"state":{"title":"Battery died after two weeks",'
    '"review":"The headphones sound great, but the battery stopped charging '
    'after two weeks."},'
    '"questions":{'
    '"defect":{"type":"noul","instructions":"Does the review report a product '
    'defect?"},'
    '"sentiment":{"type":"choice","instructions":"What is the overall sentiment '
    'of the review?","criteria":{"positive":"Mostly satisfied","mixed":"Praise '
    'and complaints in one review","negative":"Mostly dissatisfied"}},'
    '"severity":{"type":"score","instructions":"How severe is the reported '
    'problem?","criteria":["Cosmetic","Inconvenient","Product unusable"]}}}'
)


def test_noul_accepts_instructions_criteria_or_both() -> None:
    Noul(instructions="Is it broken?")
    Noul(criteria=NoulCriteria(true="Broken"))
    Noul(instructions="Is it broken?", criteria=NoulCriteria(false="Fine"))


def test_noul_rejects_neither() -> None:
    with pytest.raises(ValidationError, match="instructions"):
        Noul()
    with pytest.raises(ValidationError, match="instructions"):
        Noul(criteria=NoulCriteria())


def test_choice_bounds() -> None:
    Choice(instructions="Pick", criteria={"only": None})
    Choice(instructions="Pick", criteria={str(i): None for i in range(255)})
    with pytest.raises(ValidationError, match="instructions"):
        Choice.model_validate({"criteria": {"a": None}})
    with pytest.raises(ValidationError, match="at least 1"):
        Choice(instructions="Pick", criteria={})
    with pytest.raises(ValidationError, match="at most 255"):
        Choice(instructions="Pick", criteria={str(i): None for i in range(256)})
    with pytest.raises(ValidationError, match="non-empty"):
        Choice(instructions="Pick", criteria={"": "blank"})


def test_score_bounds() -> None:
    Score(instructions="Rate", criteria=["only"])
    Score(instructions="Rate", criteria=[str(i) for i in range(10)])
    with pytest.raises(ValidationError, match="instructions"):
        Score.model_validate({"criteria": ["a", "b"]})
    with pytest.raises(ValidationError, match="at least 1"):
        Score(instructions="Rate", criteria=[])
    with pytest.raises(ValidationError, match="at most 10"):
        Score(instructions="Rate", criteria=[str(i) for i in range(11)])


def test_question_count_and_names() -> None:
    route = ROUTES["native"]
    question = Noul(instructions="q")
    build_payload(
        {"state": "s", "questions": {str(i): question for i in range(128)}},
        model="m",
        route=route,
    )
    with pytest.raises(ValueError, match="1 to 128"):
        build_payload({"state": "s", "questions": {}}, model="m", route=route)
    with pytest.raises(ValueError, match="1 to 128"):
        build_payload(
            {"state": "s", "questions": {str(i): question for i in range(129)}},
            model="m",
            route=route,
        )
    with pytest.raises(ValueError, match="non-empty"):
        build_payload(
            {"state": "s", "questions": {"": question}}, model="m", route=route
        )


def test_openrouter_instructions_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    request: DecisionRequest = {
        "state": "s",
        "questions": {"n": Noul(criteria=NoulCriteria(true="yes"))},
    }
    build_payload(request, model="m", route=ROUTES["openrouter"])
    monkeypatch.setattr(_routes, "OPENROUTER_REQUIRES_INSTRUCTIONS", True)
    build_payload(request, model="m", route=ROUTES["native"])
    with pytest.raises(ValueError, match="OpenRouter requires instructions"):
        build_payload(request, model="m", route=ROUTES["openrouter"])


def test_quickstart_payload_bytes() -> None:
    payload = build_payload(
        {
            "state": {
                "title": "Battery died after two weeks",
                "review": (
                    "The headphones sound great, but the battery stopped charging "
                    "after two weeks."
                ),
            },
            "questions": {
                "defect": Noul(instructions="Does the review report a product defect?"),
                "sentiment": Choice(
                    instructions="What is the overall sentiment of the review?",
                    criteria={
                        "positive": "Mostly satisfied",
                        "mixed": "Praise and complaints in one review",
                        "negative": "Mostly dissatisfied",
                    },
                ),
                "severity": Score(
                    instructions="How severe is the reported problem?",
                    criteria=["Cosmetic", "Inconvenient", "Product unusable"],
                ),
            },
        },
        model="pplx-decider-v1.1-27b",
        route=ROUTES["native"],
    )
    assert json.dumps(payload, separators=(",", ":")) == QUICKSTART_PAYLOAD


def test_payload_omits_none_fields_and_keeps_null_descriptions() -> None:
    payload = build_payload(
        {
            "state": "s",
            "questions": {
                "n": Noul(criteria=NoulCriteria(true="yes")),
                "c": Choice(instructions="i", criteria={"other": None}),
            },
        },
        model="m",
        route=ROUTES["native"],
    )
    assert payload["questions"] == {
        "n": {"type": "noul", "criteria": {"true": "yes"}},
        "c": {"type": "choice", "instructions": "i", "criteria": {"other": None}},
    }


def test_response_requires_only_answers_and_keeps_unknown_fields() -> None:
    response = DecisionResponse.model_validate(
        {
            "answers": {
                "n": {"type": "noul", "noul": 0.5},
                "c": {"type": "choice", "choice": "a"},
                "s": {
                    "type": "score",
                    "score": 0.0,
                    "legend": {"0": "only"},
                    "probabilities": {"0": 1.0},
                },
            },
            "object": "decision",
        }
    )
    assert response.model is None
    assert response.usage is None
    assert response.provider is None
    assert response.choices["c"].probabilities is None
    assert response.choices["c"].confidence is None
    assert response.scores["s"].legend == {0: "only"}
    assert response.scores["s"].probabilities == {0: 1.0}
    assert response.raw == {"object": "decision"}
    assert set(response.nouls) == {"n"}


def test_response_rejects_out_of_range_noul() -> None:
    with pytest.raises(ValidationError):
        DecisionResponse.model_validate({"answers": {"n": {"type": "noul", "noul": 2}}})


def test_endpoint_validation() -> None:
    DecisionsEndpoint(base_url="https://gw.example.com/api/", path="/decisions")
    with pytest.raises(ValidationError, match="start with '/'"):
        DecisionsEndpoint(base_url="https://gw.example.com", path="decisions")
    with pytest.raises(ValidationError, match="http"):
        DecisionsEndpoint(base_url="gw.example.com", path="/decisions")
