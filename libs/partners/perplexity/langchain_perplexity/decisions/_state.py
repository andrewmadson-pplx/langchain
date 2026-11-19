"""Convert LangChain messages inside Decisions state to JSON."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import JsonValue

if TYPE_CHECKING:
    from langchain_perplexity.decisions.types import State


def _serialize_value(value: object) -> JsonValue:
    if isinstance(value, BaseMessage):
        return _serialize_value(convert_to_openai_messages(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            message = "Decisions state object keys must be strings."
            raise TypeError(message)
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    message = f"Unsupported Decisions state value: {type(value).__name__}."
    raise TypeError(message)


def serialize_state(state: State) -> JsonValue:
    """Convert state to the JSON the Decisions API accepts.

    Args:
        state: A string, JSON object, JSON array, `BaseMessage`, or sequence of
            messages. Messages may also be nested inside objects and arrays.

    Returns:
        A string, object, or array. Every message becomes a role/content object,
        with image parts in a `HumanMessage` kept as OpenAI-style `image_url` parts.

    Raises:
        TypeError: If the root is `None` or a non-string scalar, if an object key is
            not a string, or if a nested value cannot be represented as JSON.
    """
    if state is None or isinstance(state, (int, float, bool)):
        message = (
            "Decisions state must be a string, object, array, BaseMessage, or "
            "sequence of BaseMessage objects."
        )
        raise TypeError(message)
    return _serialize_value(state)


__all__ = ["serialize_state"]
