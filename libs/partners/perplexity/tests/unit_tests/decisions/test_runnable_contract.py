"""`Runnable` behavior: batching, concurrency, composition, callbacks, errors."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

from langchain_perplexity import PerplexityDecisions
from langchain_perplexity.decisions import (
    DecisionRequest,
    DecisionResponse,
    Noul,
    PerplexityDecisionsInvalidRequestError,
)


def _request(text: str) -> DecisionRequest:
    return {"state": text, "questions": {"q": Noul(instructions="Yes?")}}


def _ok(text: str) -> httpx.Response:
    noul = 0.9 if text == "yes" else 0.1
    return httpx.Response(
        200,
        json={
            "answers": {"q": {"type": "noul", "noul": noul}},
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
    )


class _Tracker:
    """Record request count and peak concurrency across a mock transport."""

    def __init__(self, delay: float = 0.0) -> None:
        self.count = 0
        self.active = 0
        self.peak = 0
        self.delay = delay
        self._lock = threading.Lock()

    def sync(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.count += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            return _ok(request.content.decode().split('"state":"')[1].split('"')[0])
        finally:
            with self._lock:
                self.active -= 1

    async def asynchronous(self, request: httpx.Request) -> httpx.Response:
        self.count += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return _ok(request.content.decode().split('"state":"')[1].split('"')[0])


class _Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[Any] = []

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self.starts.append({"input": args[1], "metadata": kwargs.get("metadata")})

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        self.ends.append(outputs)


def test_batch_makes_one_request_per_input() -> None:
    tracker = _Tracker()
    decisions = PerplexityDecisions(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(tracker.sync))
    )
    results = decisions.batch([_request("yes"), _request("no"), _request("yes")])
    assert [r.nouls["q"].noul for r in results] == [0.9, 0.1, 0.9]
    assert tracker.count == 3


async def test_abatch_respects_max_concurrency() -> None:
    tracker = _Tracker(delay=0.01)
    decisions = PerplexityDecisions(
        api_key="k",
        async_client=httpx.AsyncClient(
            transport=httpx.MockTransport(tracker.asynchronous)
        ),
    )
    inputs = [_request("yes")] * 4
    await decisions.abatch(inputs, config={"max_concurrency": 1})
    assert tracker.count == 4
    assert tracker.peak == 1

    tracker = _Tracker(delay=0.01)
    decisions = PerplexityDecisions(
        api_key="k",
        async_client=httpx.AsyncClient(
            transport=httpx.MockTransport(tracker.asynchronous)
        ),
    )
    await decisions.abatch(inputs)
    assert tracker.peak > 1


def test_composes_with_lcel() -> None:
    tracker = _Tracker()
    decisions = PerplexityDecisions(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(tracker.sync))
    )
    chain = (
        RunnableLambda(_request)
        | decisions
        | RunnableLambda(lambda r: r.nouls["q"].noul >= 0.5)
    )
    assert chain.invoke("yes") is True
    assert chain.invoke("no") is False


def test_callbacks_receive_tracing_metadata() -> None:
    tracker = _Tracker()
    decisions = PerplexityDecisions(
        route="native",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(tracker.sync)),
    )
    recorder = _Recorder()
    decisions.invoke(_request("yes"), config={"callbacks": [recorder]})

    [start] = recorder.starts
    assert start["input"] == _request("yes")
    assert start["metadata"] == {
        "ls_provider": "perplexity",
        "ls_model_name": "pplx-decider-v1.1-27b",
        "ls_model_type": "chat",
    }
    [end] = recorder.ends
    assert isinstance(end, DecisionResponse)


def test_batch_error_propagation() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if b'"state":"bad"' in request.content:
            return httpx.Response(400, json={"error": {"message": "bad state"}})
        return _ok("yes")

    decisions = PerplexityDecisions(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    inputs = [_request("yes"), _request("bad")]
    with pytest.raises(PerplexityDecisionsInvalidRequestError):
        decisions.batch(inputs)

    results = decisions.batch(inputs, return_exceptions=True)
    assert isinstance(results[0], DecisionResponse)
    assert isinstance(results[1], PerplexityDecisionsInvalidRequestError)
