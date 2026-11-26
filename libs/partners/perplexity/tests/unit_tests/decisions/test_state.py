"""State serialization for strings, JSON, and LangChain messages."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_perplexity.decisions._state import serialize_state

IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="


@pytest.mark.parametrize(
    "state",
    ["plain text", {"title": "t", "tags": ["a", 1, 2.5, True, None]}, ["a", {"b": 1}]],
)
def test_json_state_passes_through(state: Any) -> None:
    assert serialize_state(state) == state


def test_human_message_with_text_and_image_parts() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Which color is the square?"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ]
    )
    assert serialize_state(message) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Which color is the square?"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    }


def test_standard_image_block_becomes_image_url_part() -> None:
    message = HumanMessage(
        content=[{"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"}]
    )
    assert serialize_state(message) == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}}],
    }


def test_message_sequence_and_nested_messages() -> None:
    messages = [
        SystemMessage("You are a support assistant."),
        HumanMessage("My integration is broken."),
        AIMessage("I can help."),
    ]
    assert serialize_state(messages) == [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "My integration is broken."},
        {"role": "assistant", "content": "I can help."},
    ]
    assert serialize_state({"ticket": 7, "thread": messages[1:]}) == {
        "ticket": 7,
        "thread": [
            {"role": "user", "content": "My integration is broken."},
            {"role": "assistant", "content": "I can help."},
        ],
    }


@pytest.mark.parametrize("state", [None, 1, 2.5, True])
def test_scalar_roots_are_rejected(state: Any) -> None:
    with pytest.raises(TypeError, match="must be a string, object, array"):
        serialize_state(state)


def test_non_string_keys_and_non_json_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        serialize_state({1: "a"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="Unsupported"):
        serialize_state({"when": object()})  # type: ignore[dict-item]
