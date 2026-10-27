"""Live-gate behavior for the Decisions integration tests.

Without `--require-live`, a test marked `live_required` skips when its route key is
missing. With the option, that skip becomes a failure and a collection that
contains no `live_required` tests is a usage error. The option itself is registered
in `tests/integration_tests/conftest.py` because pytest only reads options from
conftest files at or above the directory it was pointed at.
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import zlib
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from langchain_perplexity.decisions import (
    Choice,
    DecisionRequest,
    DecisionResponse,
    Noul,
    NoulCriteria,
    Score,
    State,
)

LIVE_MARKER = "live_required"
FIXTURES = Path(__file__).parent / "fixtures"


def _require_live(config: pytest.Config) -> bool:
    return bool(config.getoption("--require-live", default=False))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Fail the live gate when nothing live-required was collected."""
    if not _require_live(config):
        return
    if not any(item.get_closest_marker(LIVE_MARKER) for item in items):
        message = "--require-live was set but no live_required tests were collected."
        raise pytest.UsageError(message)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Turn a missing-key skip into a failure under `--require-live`."""
    marker = item.get_closest_marker(LIVE_MARKER)
    if marker is None or not _require_live(item.config):
        return
    missing = [name for name in marker.args if not os.environ.get(name)]
    if len(missing) == len(marker.args):
        pytest.fail(
            f"--require-live is set but none of {list(marker.args)} is set.",
            pytrace=False,
        )


def _png_data_url() -> str:
    """Return a 64 by 64 PNG data URL: a solid red square on white."""
    size = 64
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            rows.extend((255, 0, 0) if x < 32 and y < 32 else (255, 255, 255))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture(scope="session")
def image_data_url() -> str:
    """Synthetic PNG used by the image-state tests."""
    return _png_data_url()


STATE: State = {
    "title": "Battery died after two weeks",
    "review": (
        "The headphones sound great, but the battery stopped charging after two weeks."
    ),
}
SENTIMENTS = {
    "positive": "Mostly satisfied",
    "mixed": "Praise and complaints in one review",
    "negative": "Mostly dissatisfied",
}
SEVERITY = ["Cosmetic", "Inconvenient", "Product unusable"]
COLORS = {"red": None, "blue": None, "green": None}


@pytest.fixture
def quickstart_request() -> DecisionRequest:
    """Row 1 and 2: the three quickstart questions with instructions."""
    return {
        "state": STATE,
        "questions": {
            "defect": Noul(instructions="Does the review report a product defect?"),
            "sentiment": Choice(
                instructions="What is the overall sentiment of the review?",
                criteria=SENTIMENTS,
            ),
            "severity": Score(
                instructions="How severe is the reported problem?",
                criteria=SEVERITY,
            ),
        },
    }


@pytest.fixture
def criteria_only_request() -> DecisionRequest:
    """Row 3: a `noul` with criteria and no instructions."""
    return {
        "state": STATE,
        "questions": {
            "defect": Noul(
                criteria=NoulCriteria(
                    true="Something is broken or not working.",
                    false="Normal wear or personal preference.",
                )
            )
        },
    }


@pytest.fixture
def single_level_request() -> DecisionRequest:
    """Row 4: a `score` with exactly one level."""
    return {
        "state": STATE,
        "questions": {
            "severity": Score(instructions="How severe?", criteria=["Only level"])
        },
    }


@pytest.fixture
def image_request(image_data_url: str) -> DecisionRequest:
    """Row 5: a `HumanMessage` with one text part and one image part."""
    return {
        "state": HumanMessage(
            content=[
                {"type": "text", "text": "Which color is the square?"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        ),
        "questions": {
            "color": Choice(instructions="What color is the square?", criteria=COLORS)
        },
    }


def assert_quickstart_structure(response: DecisionResponse) -> None:
    """Check keys, types, ranges, and membership. Never values."""
    assert set(response.answers) == {"defect", "sentiment", "severity"}
    defect = response.nouls["defect"]
    assert 0.0 <= defect.noul <= 1.0
    sentiment = response.choices["sentiment"]
    assert sentiment.choice in SENTIMENTS
    if sentiment.probabilities is not None:
        assert set(sentiment.probabilities) == set(SENTIMENTS)
        assert math.isclose(sum(sentiment.probabilities.values()), 1.0, abs_tol=0.05)
    severity = response.scores["severity"]
    assert 0.0 <= severity.score <= len(SEVERITY) - 1
    if severity.legend is not None:
        assert severity.legend == dict(enumerate(SEVERITY))
    if severity.probabilities is not None:
        assert set(severity.probabilities) == set(range(len(SEVERITY)))


@pytest.fixture
def check_quickstart_structure() -> Callable[[DecisionResponse], None]:
    """Structural checks for the quickstart response, shared by both routes."""
    return assert_quickstart_structure


@pytest.fixture
def recorded_response() -> Callable[[str, str], dict[str, object]]:
    """Load the sanitized live response recorded for a route."""

    def load(row: str, route: str) -> dict[str, object]:
        data = json.loads((FIXTURES / f"{row}.{route}.json").read_text())
        response = data["response"]
        assert isinstance(response, dict)
        return response

    return load


def _assert_same_top_level_shape(
    response: DecisionResponse, recorded: dict[str, object]
) -> None:
    body = recorded["json"]
    assert isinstance(body, dict)
    live = {
        key: value
        for key, value in response.model_dump(exclude_none=True).items()
        if key not in {"request_id", "raw"}
    } | response.raw
    assert set(live) == set(body)
    if "usage" in body and isinstance(body["usage"], dict):
        assert response.usage is not None
        assert set(response.usage.model_dump(exclude_none=True)) == set(body["usage"])


@pytest.fixture
def check_same_top_level_shape() -> Callable[
    [DecisionResponse, dict[str, object]], None
]:
    """Check a live response has the same top-level and usage keys as recorded."""
    return _assert_same_top_level_shape
