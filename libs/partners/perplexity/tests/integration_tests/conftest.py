"""Command-line options shared by the integration test tree."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--require-live`, which turns missing-key skips into failures."""
    parser.addoption(
        "--require-live",
        action="store_true",
        default=False,
        help="Fail live Decisions tests that would skip because a key is missing.",
    )
