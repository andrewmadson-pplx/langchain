"""Request building, response classification and errors for the Decisions API."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
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
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import override

from langchain_perplexity._version import __version__
from langchain_perplexity.decisions import _routes
from langchain_perplexity.decisions._routes import RouteSpec
from langchain_perplexity.decisions._state import serialize_state
from langchain_perplexity.decisions.types import (
    MAX_QUESTIONS,
    DecisionRequest,
    DecisionResponse,
    Noul,
    Question,
)

RESERVED_PAYLOAD_KEYS = frozenset({"model", "state", "questions"})
"""Top-level request keys that `extra` may never override."""

USER_AGENT = f"langchain-perplexity/{__version__}"
"""`User-Agent` header sent on both routes."""

_RETRY_AFTER_HEADER = "retry-after"
_QUESTION_ADAPTER: TypeAdapter[Question] = TypeAdapter(Question)


class PerplexityDecisionsError(Exception):
    """Base class for every error raised by `PerplexityDecisions`."""


class PerplexityDecisionsAPIError(PerplexityDecisionsError, ModelError):
    """An HTTP response that is not a usable decision.

    Attributes:
        status: HTTP status code.
        body: Parsed JSON body, response text, or `None` when the body is empty.
        headers: Response headers.
        endpoint: Request method and URL without credentials, query, or fragment.
        request_id: Request ID taken from the body or a header, when present.

    `body` and `headers` are available for handling code but are never part of
    `str()` or `repr()`, so logs and tracebacks do not carry state, keys, or
    provider error text.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx.Headers,
        message: str | None = None,
        endpoint: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Create an API error.

        Args:
            status: HTTP status code.
            body: Parsed JSON body, response text, or `None`.
            headers: Response headers.
            message: Optional short description that contains no response data.
            endpoint: Sanitized request method and URL.
            request_id: Request ID, when known.
        """
        super().__init__(status, body, headers, message, endpoint, request_id)
        self.status = status
        self.body = body
        self.headers = headers
        self.endpoint = endpoint
        self.request_id = request_id
        self._message = message

    @property
    def status_code(self) -> int:
        """Alias for `status`."""
        return self.status

    @override
    def __str__(self) -> str:
        try:
            reason = HTTPStatus(self.status).phrase
        except ValueError:
            reason = "API request failed"
        text = f"{self.status} {self._message or reason}"
        if self.endpoint is not None:
            text = f"{self.endpoint}: {text}"
        if self.request_id is not None:
            text = f"{text} (request_id={self.request_id})"
        return text

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class PerplexityDecisionsInvalidRequestError(
    PerplexityDecisionsAPIError, ModelInvalidRequestError
):
    """The route rejected the request (HTTP 400, 413 or 422)."""


class PerplexityDecisionsAuthenticationError(
    PerplexityDecisionsAPIError, ModelAuthenticationError
):
    """The API key is missing or invalid (HTTP 401)."""


class PerplexityDecisionsPaymentRequiredError(PerplexityDecisionsAPIError):
    """The account has no credit for the request (HTTP 402).

    `langchain_core` has no payment error type, so this class inherits only the
    base `ModelError`.
    """


class PerplexityDecisionsPermissionDeniedError(
    PerplexityDecisionsAPIError, ModelPermissionDeniedError
):
    """The key cannot perform the request (HTTP 403)."""


class PerplexityDecisionsNotFoundError(PerplexityDecisionsAPIError, ModelNotFoundError):
    """Wrong path or method (HTTP 404 or 405). The body may be empty."""


class PerplexityDecisionsRateLimitError(
    PerplexityDecisionsAPIError, ModelRateLimitError
):
    """The rate limit was exceeded (HTTP 429).

    Attributes:
        retry_after_ms: Delay requested by the `Retry-After` header in milliseconds,
            or `None` when the header is missing or unreadable.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx.Headers,
        message: str | None = None,
        endpoint: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Create a rate-limit error and read its retry delay.

        Args:
            status: HTTP status code.
            body: Parsed JSON body, response text, or `None`.
            headers: Response headers.
            message: Optional short description that contains no response data.
            endpoint: Sanitized request method and URL.
            request_id: Request ID, when known.
        """
        super().__init__(status, body, headers, message, endpoint, request_id)
        self.retry_after_ms = parse_retry_after_ms(headers)


class PerplexityDecisionsServerError(PerplexityDecisionsAPIError, ModelAPIError):
    """The route failed or redirected (HTTP 5xx or 3xx). The body may be HTML."""


class PerplexityDecisionsResponseValidationError(PerplexityDecisionsAPIError):
    """A successful response did not match the response schema.

    Attributes:
        field_path: Dotted path to the first field that failed validation.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx.Headers,
        field_path: str,
        endpoint: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Create a response-validation error.

        Args:
            status: HTTP status code.
            body: Parsed JSON body or response text.
            headers: Response headers.
            field_path: Dotted path to the first invalid field.
            endpoint: Sanitized request method and URL.
            request_id: Request ID, when known.
        """
        self.field_path = field_path
        super().__init__(
            status,
            body,
            headers,
            f"Invalid response data at {field_path!r}",
            endpoint,
            request_id,
        )
        self.args = (status, body, headers, field_path, endpoint, request_id)


class PerplexityDecisionsConnectionError(
    PerplexityDecisionsError, ModelConnectionError
):
    """The request failed before an HTTP response arrived."""


class PerplexityDecisionsTimeoutError(PerplexityDecisionsError, ModelTimeoutError):
    """The request exceeded the client timeout."""


_STATUS_ERRORS: dict[int, type[PerplexityDecisionsAPIError]] = {
    400: PerplexityDecisionsInvalidRequestError,
    401: PerplexityDecisionsAuthenticationError,
    402: PerplexityDecisionsPaymentRequiredError,
    403: PerplexityDecisionsPermissionDeniedError,
    404: PerplexityDecisionsNotFoundError,
    405: PerplexityDecisionsNotFoundError,
    413: PerplexityDecisionsInvalidRequestError,
    422: PerplexityDecisionsInvalidRequestError,
    429: PerplexityDecisionsRateLimitError,
}


def parse_retry_after_ms(headers: httpx.Headers) -> float | None:
    """Read a `Retry-After` header as milliseconds.

    Args:
        headers: Response headers.

    Returns:
        The delay in milliseconds for a seconds value or an HTTP date, or `None`
        when the header is missing, negative, or unreadable.
    """
    raw = headers.get(_RETRY_AFTER_HEADER)
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        try:
            delay = (parsedate_to_datetime(raw).timestamp() - time.time()) * 1000
        except (OverflowError, TypeError, ValueError):
            return None
        return max(0.0, delay)
    if math.isfinite(seconds) and seconds >= 0:
        return seconds * 1000
    return None


def build_headers(
    api_key: str,
    *,
    app_url: str | None = None,
    app_title: str | None = None,
) -> dict[str, str]:
    """Build request headers.

    Args:
        api_key: Bearer token.
        app_url: `HTTP-Referer` value, sent only when not `None`.
        app_title: `X-Title` value, sent only when not `None`.

    Returns:
        Headers for one request.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    if app_url is not None:
        headers["HTTP-Referer"] = app_url
    if app_title is not None:
        headers["X-Title"] = app_title
    return headers


def validate_extra(extra: Mapping[str, JsonValue] | None, route: RouteSpec) -> None:
    """Check `extra` request keys against the route allowlist.

    Args:
        extra: Additional top-level request keys, or `None`.
        route: Route whose allowlist applies.

    Raises:
        ValueError: If a key is reserved or not allowed on the route.
    """
    if not extra:
        return
    reserved = sorted(RESERVED_PAYLOAD_KEYS.intersection(extra))
    if reserved:
        message = f"`extra` may not override reserved request keys: {reserved}."
        raise ValueError(message)
    rejected = sorted(key for key in extra if key not in route.extra_allowlist)
    if rejected:
        if not route.extra_allowlist:
            message = (
                f"The {route.name} route does not accept `extra` keys: {rejected}."
            )
        else:
            message = (
                f"`extra` keys {rejected} are not allowed on the {route.name} route. "
                f"Allowed keys: {sorted(route.extra_allowlist)}."
            )
        raise ValueError(message)


def _validate_questions(
    questions: Mapping[str, object], route: RouteSpec
) -> dict[str, JsonValue]:
    if not isinstance(questions, Mapping):
        message = "`questions` must be a mapping of question names to questions."
        raise TypeError(message)
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        message = f"A request needs 1 to {MAX_QUESTIONS} questions."
        raise ValueError(message)
    payload: dict[str, JsonValue] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            message = "Question names must be non-empty strings."
            raise ValueError(message)
        parsed = _QUESTION_ADAPTER.validate_python(question)
        if (
            _routes.OPENROUTER_REQUIRES_INSTRUCTIONS
            and route.name == "openrouter"
            and isinstance(parsed, Noul)
            and parsed.instructions is None
        ):
            message = "OpenRouter requires instructions on noul questions"
            raise ValueError(message)
        payload[name] = parsed.model_dump(mode="json", exclude_none=True)
    return payload


def build_payload(
    request: DecisionRequest,
    *,
    model: str,
    route: RouteSpec,
    extra: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Build the JSON body for one request.

    Args:
        request: State and questions.
        model: Resolved model name.
        route: Route the request is sent on.
        extra: Allowlisted additional keys, already validated.

    Returns:
        A JSON object with `model`, `state`, `questions`, and any `extra` keys.

    Raises:
        ValueError: If the questions violate a validation rule.
        TypeError: If the state cannot be represented as JSON.
    """
    payload: dict[str, JsonValue] = {
        "model": model,
        "state": serialize_state(request["state"]),
        "questions": _validate_questions(request["questions"], route),
    }
    if extra:
        validate_extra(extra, route)
        payload.update(extra)
    return payload


def sanitized_endpoint(request: httpx.Request | None) -> str | None:
    """Describe a request as `METHOD scheme://host/path` without secrets.

    Args:
        request: The HTTP request, or `None`.

    Returns:
        The method and URL without userinfo, query, or fragment, or `None`.
    """
    if request is None:
        return None
    parts = urlsplit(str(request.url))
    hostname = parts.hostname
    if hostname is None:
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return f"{request.method} {urlunsplit((parts.scheme, netloc, parts.path, '', ''))}"


def _request_of(response: httpx.Response) -> httpx.Request | None:
    try:
        return response.request
    except RuntimeError:
        return None


def _is_json(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def _body(response: httpx.Response) -> Any:
    if not response.content:
        return None
    if _is_json(response):
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


def _request_id(response: httpx.Response, body: Any, route: RouteSpec) -> str | None:
    if route.request_id_header is not None:
        header = response.headers.get(route.request_id_header)
        if header:
            return header
    if route.request_id_in_body and isinstance(body, dict):
        value = body.get("id")
        if isinstance(value, str) and value:
            return value
    return None


def classify_response(
    response: httpx.Response, route: RouteSpec
) -> PerplexityDecisionsAPIError | None:
    """Map a non-success response to an error without reading its body first.

    Args:
        response: The HTTP response.
        route: Route the request was sent on.

    Returns:
        The error to raise, or `None` for a 2xx response.
    """
    status = response.status_code
    if 200 <= status < 300:
        return None
    endpoint = sanitized_endpoint(_request_of(response))
    body = _body(response)
    request_id = _request_id(response, body, route)
    if 300 <= status < 400:
        return PerplexityDecisionsServerError(
            status,
            body,
            response.headers,
            "Redirects are disabled; the route returned a redirect",
            endpoint,
            request_id,
        )
    error_type = _STATUS_ERRORS.get(
        status,
        PerplexityDecisionsServerError
        if status >= 500
        else PerplexityDecisionsAPIError,
    )
    return error_type(status, body, response.headers, None, endpoint, request_id)


def parse_response(response: httpx.Response, route: RouteSpec) -> DecisionResponse:
    """Classify a response and convert a successful one to a `DecisionResponse`.

    Args:
        response: The HTTP response.
        route: Route the request was sent on.

    Returns:
        The validated response with `request_id` filled from the route's source.

    Raises:
        PerplexityDecisionsAPIError: For any non-2xx status, using the subclass
            matched to the status code.
        PerplexityDecisionsResponseValidationError: For a 2xx response whose body
            is not JSON or does not match the response schema.
    """
    error = classify_response(response, route)
    if error is not None:
        raise error
    endpoint = sanitized_endpoint(_request_of(response))
    body = _body(response)
    request_id = _request_id(response, body, route)
    try:
        parsed = DecisionResponse.model_validate(body)
    except ValidationError as validation_error:
        location = validation_error.errors()[0].get("loc", ())
        field_path = ".".join(str(item) for item in location) or "response"
        raise PerplexityDecisionsResponseValidationError(
            response.status_code,
            body,
            response.headers,
            field_path,
            endpoint,
            request_id,
        ) from validation_error
    return parsed.model_copy(update={"request_id": request_id})


def transport_error(
    error: httpx.HTTPError, endpoint: str | None
) -> PerplexityDecisionsError:
    """Convert an `httpx` transport failure to a package error.

    Args:
        error: The exception raised by `httpx`.
        endpoint: Sanitized request method and URL.

    Returns:
        A timeout error for `httpx.TimeoutException`, otherwise a connection error.
        The message never includes the request body or headers.
    """
    where = f" ({endpoint})" if endpoint else ""
    if isinstance(error, httpx.TimeoutException):
        return PerplexityDecisionsTimeoutError(f"Request timed out{where}.")
    return PerplexityDecisionsConnectionError(f"Request failed{where}.")


__all__ = [
    "RESERVED_PAYLOAD_KEYS",
    "USER_AGENT",
    "PerplexityDecisionsAPIError",
    "PerplexityDecisionsAuthenticationError",
    "PerplexityDecisionsConnectionError",
    "PerplexityDecisionsError",
    "PerplexityDecisionsInvalidRequestError",
    "PerplexityDecisionsNotFoundError",
    "PerplexityDecisionsPaymentRequiredError",
    "PerplexityDecisionsPermissionDeniedError",
    "PerplexityDecisionsRateLimitError",
    "PerplexityDecisionsResponseValidationError",
    "PerplexityDecisionsServerError",
    "PerplexityDecisionsTimeoutError",
    "build_headers",
    "build_payload",
    "classify_response",
    "parse_response",
    "parse_retry_after_ms",
    "sanitized_endpoint",
    "transport_error",
    "validate_extra",
]
