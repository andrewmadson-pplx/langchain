"""Status classification, body handling, and error sanitization."""

from __future__ import annotations

import email.utils
import time
from typing import Any

import httpx
import pytest
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)

from langchain_perplexity.decisions import (
    PerplexityDecisionsAPIError,
    PerplexityDecisionsAuthenticationError,
    PerplexityDecisionsConnectionError,
    PerplexityDecisionsError,
    PerplexityDecisionsInvalidRequestError,
    PerplexityDecisionsNotFoundError,
    PerplexityDecisionsPaymentRequiredError,
    PerplexityDecisionsPermissionDeniedError,
    PerplexityDecisionsRateLimitError,
    PerplexityDecisionsResponseValidationError,
    PerplexityDecisionsServerError,
    PerplexityDecisionsTimeoutError,
)
from langchain_perplexity.decisions._client import (
    parse_response,
    parse_retry_after_ms,
    transport_error,
)
from langchain_perplexity.decisions._routes import ROUTES

NATIVE = ROUTES["native"]
OPENROUTER = ROUTES["openrouter"]
PRIVATE_BODY = "Noul question must have criteria or instructions"
AUTH_VALUE = "Bearer sk-secret-value"
HTML_504 = "<html><head><title>504 Gateway Time-out</title></head></html>"


def _response(
    status: int,
    *,
    body: bytes | None = None,
    json: Any = None,
    headers: dict[str, str] | None = None,
    url: str = "https://api.perplexity.ai/v1/decisions",
) -> httpx.Response:
    request = httpx.Request(
        "POST", url, headers={"Authorization": AUTH_VALUE}, json={"state": "private"}
    )
    if json is not None:
        return httpx.Response(status, json=json, headers=headers, request=request)
    return httpx.Response(status, content=body or b"", headers=headers, request=request)


@pytest.mark.parametrize(
    ("response", "provider_type", "langchain_type", "is_retryable"),
    [
        (
            _response(
                400,
                json={
                    "error": {
                        "message": PRIVATE_BODY,
                        "type": "invalid_request",
                        "code": "400",
                    }
                },
            ),
            PerplexityDecisionsInvalidRequestError,
            ModelInvalidRequestError,
            False,
        ),
        (
            _response(401, json={"error": {"message": "Unauthorized"}}),
            PerplexityDecisionsAuthenticationError,
            ModelAuthenticationError,
            False,
        ),
        (
            _response(402, json={"error": {"message": "Insufficient credits"}}),
            PerplexityDecisionsPaymentRequiredError,
            ModelError,
            False,
        ),
        (
            _response(403, json={"error": {"message": "Forbidden"}}),
            PerplexityDecisionsPermissionDeniedError,
            ModelPermissionDeniedError,
            False,
        ),
        (_response(404), PerplexityDecisionsNotFoundError, ModelNotFoundError, False),
        (
            _response(405, headers={"Allow": "POST"}),
            PerplexityDecisionsNotFoundError,
            ModelNotFoundError,
            False,
        ),
        (
            _response(413, body=b"Request Entity Too Large"),
            PerplexityDecisionsInvalidRequestError,
            ModelInvalidRequestError,
            False,
        ),
        (
            _response(422, json={"detail": [{"loc": ["body", "questions"]}]}),
            PerplexityDecisionsInvalidRequestError,
            ModelInvalidRequestError,
            False,
        ),
        (
            _response(429, headers={"Retry-After": "2"}),
            PerplexityDecisionsRateLimitError,
            ModelRateLimitError,
            True,
        ),
        (
            _response(500, json={"error": {"message": "boom"}}),
            PerplexityDecisionsServerError,
            ModelAPIError,
            True,
        ),
        (
            _response(
                504, body=HTML_504.encode(), headers={"Content-Type": "text/html"}
            ),
            PerplexityDecisionsServerError,
            ModelAPIError,
            True,
        ),
        (
            _response(
                307, headers={"Location": "https://elsewhere.example/v1/decisions"}
            ),
            PerplexityDecisionsServerError,
            ModelAPIError,
            True,
        ),
        (_response(418), PerplexityDecisionsAPIError, ModelError, False),
    ],
)
def test_status_is_classified_before_body(
    response: httpx.Response,
    provider_type: type[PerplexityDecisionsAPIError],
    langchain_type: type[ModelError],
    *,
    is_retryable: bool,
) -> None:
    with pytest.raises(provider_type) as exc_info:
        parse_response(response, NATIVE)
    error = exc_info.value
    assert isinstance(error, langchain_type)
    assert isinstance(error, PerplexityDecisionsError)
    assert error.is_retryable is is_retryable
    assert error.status == response.status_code
    text = f"{error!s} {error!r}"
    assert str(response.status_code) in text
    assert "api.perplexity.ai/v1/decisions" in text
    assert PRIVATE_BODY not in text
    assert HTML_504 not in text
    assert AUTH_VALUE not in text
    assert "sk-secret" not in text
    assert "private" not in text


def test_empty_and_html_bodies_are_kept_as_text_or_none() -> None:
    with pytest.raises(PerplexityDecisionsNotFoundError) as not_found:
        parse_response(_response(404), NATIVE)
    assert not_found.value.body is None
    with pytest.raises(PerplexityDecisionsServerError) as gateway:
        parse_response(
            _response(
                504, body=HTML_504.encode(), headers={"Content-Type": "text/html"}
            ),
            NATIVE,
        )
    assert gateway.value.body == HTML_504
    assert "Redirect" not in str(gateway.value)


def test_redirect_error_message_and_json_body_parsing() -> None:
    with pytest.raises(PerplexityDecisionsServerError, match="Redirects are disabled"):
        parse_response(_response(302, headers={"Location": "/elsewhere"}), NATIVE)
    with pytest.raises(PerplexityDecisionsInvalidRequestError) as exc_info:
        parse_response(
            _response(
                400,
                json={"error": {"message": PRIVATE_BODY}},
                headers={"x-request-id": "req-1"},
            ),
            NATIVE,
        )
    assert exc_info.value.body == {"error": {"message": PRIVATE_BODY}}
    assert exc_info.value.request_id == "req-1"
    assert "(request_id=req-1)" in str(exc_info.value)


def test_retry_after_seconds_and_http_date() -> None:
    with pytest.raises(PerplexityDecisionsRateLimitError) as seconds:
        parse_response(_response(429, headers={"Retry-After": "2"}), NATIVE)
    assert seconds.value.retry_after_ms == 2000.0

    http_date = email.utils.formatdate(time.time() + 30, usegmt=True)
    with pytest.raises(PerplexityDecisionsRateLimitError) as dated:
        parse_response(_response(429, headers={"Retry-After": http_date}), NATIVE)
    assert dated.value.retry_after_ms is not None
    assert 0 < dated.value.retry_after_ms <= 30_000

    with pytest.raises(PerplexityDecisionsRateLimitError) as missing:
        parse_response(_response(429), NATIVE)
    assert missing.value.retry_after_ms is None
    assert parse_retry_after_ms(httpx.Headers({"Retry-After": "soon"})) is None
    assert parse_retry_after_ms(httpx.Headers({"Retry-After": "-1"})) is None


@pytest.mark.parametrize(
    ("response", "field_path"),
    [
        (
            _response(
                200,
                body=b"{not json",
                headers={"Content-Type": "application/json"},
            ),
            "response",
        ),
        (
            _response(
                200, body=b"<html>ok</html>", headers={"Content-Type": "text/html"}
            ),
            "response",
        ),
        (_response(200, json={"model": "m"}), "answers"),
        (
            _response(200, json={"answers": {"n": {"type": "noul", "noul": "high"}}}),
            "answers.n.noul.noul",
        ),
    ],
)
def test_malformed_success_raises_validation_error(
    response: httpx.Response, field_path: str
) -> None:
    with pytest.raises(PerplexityDecisionsResponseValidationError) as exc_info:
        parse_response(response, NATIVE)
    error = exc_info.value
    assert isinstance(error, ModelError)
    assert error.field_path == field_path
    assert error.status == 200
    assert error.body is not None
    assert "<html>" not in str(error)
    assert "high" not in str(error)


def test_request_id_source_by_route() -> None:
    body = {"id": "gen-123", "answers": {"n": {"type": "noul", "noul": 0.1}}}
    headers = {"x-request-id": "hdr-456"}
    assert parse_response(_response(200, json=body), OPENROUTER).request_id == "gen-123"
    native = parse_response(_response(200, json=body, headers=headers), NATIVE)
    assert native.request_id == "hdr-456"
    assert native.id == "gen-123"
    with pytest.raises(PerplexityDecisionsServerError) as exc_info:
        parse_response(_response(500, json={"id": "gen-err"}), OPENROUTER)
    assert exc_info.value.request_id == "gen-err"


def test_endpoint_sanitization_drops_userinfo_query_and_fragment() -> None:
    response = _response(
        400, url="https://user:pw@[2001:db8::1]:8443/v1/decisions?token=secret#frag"
    )
    with pytest.raises(PerplexityDecisionsInvalidRequestError) as exc_info:
        parse_response(response, NATIVE)
    assert exc_info.value.endpoint == "POST https://[2001:db8::1]:8443/v1/decisions"
    assert "pw" not in str(exc_info.value)
    assert "token" not in str(exc_info.value)


def test_transport_errors_map_to_timeout_and_connection() -> None:
    timeout = transport_error(httpx.ReadTimeout("slow"), "POST https://h/p")
    assert isinstance(timeout, PerplexityDecisionsTimeoutError)
    assert isinstance(timeout, ModelTimeoutError)
    assert timeout.is_retryable is True
    assert str(timeout) == "Request timed out (POST https://h/p)."

    connect = transport_error(httpx.ConnectError("refused"), None)
    assert isinstance(connect, PerplexityDecisionsConnectionError)
    assert isinstance(connect, ModelConnectionError)
    assert str(connect) == "Request failed."
