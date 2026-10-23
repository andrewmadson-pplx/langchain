"""LangChain runnable for the Perplexity Decisions API."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import httpx
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langsmith.run_helpers import get_current_run_tree
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self, override

from langchain_perplexity.decisions._client import (
    build_headers,
    build_payload,
    parse_response,
    sanitized_endpoint,
    transport_error,
    validate_extra,
)
from langchain_perplexity.decisions._routes import (
    APP_TITLE_ENV_VAR,
    APP_URL_ENV_VAR,
    DEFAULT_APP_TITLE,
    DEFAULT_APP_URL,
    ROUTES,
    RouteSpec,
    join_url,
)
from langchain_perplexity.decisions.types import (
    DecisionRequest,
    DecisionResponse,
    DecisionsEndpoint,
)

logger = logging.getLogger(__name__)

_CUSTOM_PROVIDER = "custom"


class PerplexityDecisions(RunnableSerializable[DecisionRequest, DecisionResponse]):
    """Ask typed questions about text, JSON, or messages and get probabilities back.

    One invocation sends a `state` and one or more named questions in a single HTTP
    request and returns one answer per question. `Noul` answers are the probability
    of yes, `Choice` answers name the most likely option, and `Score` answers give
    the expected level on a rubric. Your code decides what to do with the numbers;
    thresholds belong to the application.

    Two routes share the same wire format and are selected explicitly with `route`:

    - `openrouter` (default): `https://openrouter.ai/api/alpha/decisions` with
      `OPENROUTER_API_KEY` and model `perplexity/pplx-decider-v1.2`. This route
      requires `instructions` on every `Noul`.
    - `native`: `https://api.perplexity.ai/v1/decisions` with `PPLX_API_KEY` or
      `PERPLEXITY_API_KEY` and model `pplx-decider-v1.2`. This route also accepts a
      `Noul` with only `criteria`.

    There is no fallback between routes, no key sniffing, and no retry inside the
    class. One `invoke` makes exactly one HTTP request; `batch` makes one request
    per input.

    Example:
        ```python
        from langchain_perplexity import PerplexityDecisions
        from langchain_perplexity.decisions import Choice, Noul, Score

        decisions = PerplexityDecisions()
        response = decisions.invoke(
            {
                "state": "The headphones sound great, but the battery died.",
                "questions": {
                    "defect": Noul(instructions="Does the review report a defect?"),
                    "sentiment": Choice(
                        instructions="What is the overall sentiment?",
                        criteria={"positive": None, "mixed": None, "negative": None},
                    ),
                    "severity": Score(
                        instructions="How severe is the problem?",
                        criteria=["Cosmetic", "Inconvenient", "Unusable"],
                    ),
                },
            }
        )
        if response.nouls["defect"].noul >= 0.8:
            open_ticket()
        ```

    Args:
        route: `openrouter` or `native`.
        model: Model name. Defaults to the route's model.
        api_key: API key. Defaults to the route's environment variables.
        endpoint: Explicit base URL and path. Requires `api_key`.
        app_url: `HTTP-Referer` on the openrouter route. Defaults to
            `OPENROUTER_APP_URL`, then `https://docs.langchain.com`.
        app_title: `X-Title` on the openrouter route. Defaults to
            `OPENROUTER_APP_TITLE`, then `LangChain`.
        timeout: Seconds applied to clients this class creates.
        extra: Additional top-level request keys. Only `provider`, `session_id`,
            `trace` and `user` are accepted, and only on the openrouter route.
        client: Injected `httpx.Client`. Never closed by this class.
        async_client: Injected `httpx.AsyncClient`. Never closed by this class.

    Raises:
        ValueError: If no key can be resolved for the route, if `endpoint` is set
            without an explicit `api_key`, or if `extra` is not allowed.
    """

    route: Literal["openrouter", "native"] = "openrouter"
    """Which transport to use. The default is `openrouter`."""

    model: str | None = None
    """Model name. `None` means the route default.

    Defaults are `perplexity/pplx-decider-v1.2` on the openrouter route and
    `pplx-decider-v1.2` on the native route. Each route also serves a flash variant
    (`perplexity/pplx-decider-flash-v1.2`, `pplx-decider-flash-v1.2`) and moving
    aliases (`~perplexity/pplx-decider-latest`, `pplx-decider-latest`, and their
    flash forms). Pin a versioned ID for reproducible behavior. OpenRouter echoes
    the model back as a dated snapshot such as `perplexity/pplx-decider-v1.2-20261016`.
    """

    api_key: SecretStr | None = Field(default=None, repr=False)
    """API key. `None` means the route's environment variables are read."""

    endpoint: DecisionsEndpoint | None = None
    """Explicit destination. When set, `api_key` must be passed explicitly."""

    app_url: str | None = None
    """`HTTP-Referer` for OpenRouter app attribution. Ignored on the native route."""

    app_title: str | None = None
    """`X-Title` for OpenRouter app attribution. Ignored on the native route."""

    timeout: float = Field(default=30.0, gt=0)
    """Timeout in seconds for clients created by this class."""

    extra: dict[str, JsonValue] | None = None
    """Allowlisted additional request keys for the openrouter route."""

    client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    """Injected synchronous client used by `invoke` and `batch`."""

    async_client: httpx.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )
    """Injected asynchronous client used by `ainvoke` and `abatch`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _owned_client: httpx.Client | None = PrivateAttr(default=None)
    _owned_async_client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _resolved_app_url: str | None = PrivateAttr(default=None)
    _resolved_app_title: str | None = PrivateAttr(default=None)

    @field_validator("model")
    @classmethod
    def _strip_model(cls, model: str | None) -> str | None:
        if model is None:
            return None
        model = model.strip()
        if not model:
            message = "`model` must not be empty."
            raise ValueError(message)
        return model

    @model_validator(mode="after")
    def _resolve_configuration(self) -> Self:
        route = self._route_spec
        validate_extra(self.extra, route)
        if self.endpoint is not None:
            if self.api_key is None:
                message = (
                    "`endpoint` requires an explicit `api_key`. Keys from "
                    f"{list(route.api_key_env_vars)} are never sent to a custom host."
                )
                raise ValueError(message)
        elif self.api_key is None:
            for name in route.api_key_env_vars:
                value = os.environ.get(name)
                if value:
                    self.api_key = SecretStr(value)
                    break
            else:
                names = " or ".join(route.api_key_env_vars)
                message = (
                    f"No API key for the {route.name} route. Pass `api_key` or set "
                    f"{names}."
                )
                raise ValueError(message)
        if route.attribution_headers:
            self._resolved_app_url = (
                self.app_url
                if self.app_url is not None
                else os.environ.get(APP_URL_ENV_VAR) or DEFAULT_APP_URL
            )
            self._resolved_app_title = (
                self.app_title
                if self.app_title is not None
                else os.environ.get(APP_TITLE_ENV_VAR) or DEFAULT_APP_TITLE
            )
        return self

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map `api_key` to the route's primary environment variable."""
        return {"api_key": self._route_spec.api_key_env_vars[0]}

    @property
    def _route_spec(self) -> RouteSpec:
        return ROUTES[self.route]

    @property
    def resolved_model(self) -> str:
        """Return the model that will be sent."""
        return self.model or self._route_spec.default_model

    @property
    def url(self) -> str:
        """Return the full request URL."""
        if self.endpoint is not None:
            return join_url(self.endpoint.base_url, self.endpoint.path)
        return self._route_spec.url

    @property
    def ls_provider(self) -> str:
        """Return the `ls_provider` tracing value for this configuration."""
        if self.endpoint is not None:
            return _CUSTOM_PROVIDER
        return self._route_spec.ls_provider

    def close(self) -> None:
        """Close the synchronous client this instance created, if any.

        Injected clients are left open.
        """
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    async def aclose(self) -> None:
        """Close both clients this instance created, if any.

        Injected clients are left open.
        """
        self.close()
        if self._owned_async_client is not None:
            await self._owned_async_client.aclose()
            self._owned_async_client = None

    def __del__(self) -> None:
        """Close an owned synchronous client when the instance is collected."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            logger.debug("Could not close owned httpx client.", exc_info=True)

    @override
    def invoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Send one request synchronously.

        Args:
            input: State and questions.
            config: Runnable configuration for callbacks, tags, and metadata.
            **_: Accepted for `Runnable` compatibility and ignored.

        Returns:
            One answer per question plus any metadata the route returned.

        Raises:
            PerplexityDecisionsAPIError: For a non-2xx response.
            PerplexityDecisionsResponseValidationError: For a malformed 2xx body.
            PerplexityDecisionsTimeoutError: If the request times out.
            PerplexityDecisionsConnectionError: If no response is received.
            ValueError: If the questions fail validation.
        """
        return self._call_with_config(
            self._decide, input, self._traced_config(config), run_type="llm"
        )

    @override
    async def ainvoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Send one request asynchronously.

        Args:
            input: State and questions.
            config: Runnable configuration for callbacks, tags, and metadata.
            **_: Accepted for `Runnable` compatibility and ignored.

        Returns:
            One answer per question plus any metadata the route returned.

        Raises:
            PerplexityDecisionsAPIError: For a non-2xx response.
            PerplexityDecisionsResponseValidationError: For a malformed 2xx body.
            PerplexityDecisionsTimeoutError: If the request times out.
            PerplexityDecisionsConnectionError: If no response is received.
            ValueError: If the questions fail validation.
        """
        return await self._acall_with_config(
            self._adecide, input, self._traced_config(config), run_type="llm"
        )

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        payload = self._payload(request)
        client = self._sync_client()
        try:
            response = client.post(
                self.url, json=payload, headers=self._headers(), follow_redirects=False
            )
        except httpx.HTTPError as error:
            raise transport_error(error, self._endpoint_description()) from error
        return self._record_usage(parse_response(response, self._route_spec))

    async def _adecide(self, request: DecisionRequest) -> DecisionResponse:
        payload = self._payload(request)
        client = self._async_client()
        try:
            response = await client.post(
                self.url, json=payload, headers=self._headers(), follow_redirects=False
            )
        except httpx.HTTPError as error:
            raise transport_error(error, self._endpoint_description()) from error
        return self._record_usage(parse_response(response, self._route_spec))

    def _payload(self, request: DecisionRequest) -> dict[str, JsonValue]:
        return build_payload(
            request,
            model=self.resolved_model,
            route=self._route_spec,
            extra=self.extra,
        )

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:  # pragma: no cover - guaranteed by validation
            message = "API key was not resolved."
            raise ValueError(message)
        return build_headers(
            self.api_key.get_secret_value(),
            app_url=self._resolved_app_url,
            app_title=self._resolved_app_title,
        )

    def _endpoint_description(self) -> str | None:
        return sanitized_endpoint(httpx.Request("POST", self.url))

    def _sync_client(self) -> httpx.Client:
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            self._owned_client = httpx.Client(
                timeout=self.timeout, follow_redirects=False
            )
        return self._owned_client

    def _async_client(self) -> httpx.AsyncClient:
        if self.async_client is not None:
            return self.async_client
        if self._owned_async_client is None:
            self._owned_async_client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False
            )
        return self._owned_async_client

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": self.ls_provider,
            "ls_model_name": self.resolved_model,
            "ls_model_type": "chat",
        }
        return config

    def _record_usage(self, response: DecisionResponse) -> DecisionResponse:
        usage = response.usage
        if usage is None:
            return response
        if usage.input_tokens is None and usage.output_tokens is None:
            return response
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        usage_metadata: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if usage.cost is not None:
            usage_metadata["total_cost"] = usage.cost
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                run_tree.extra.setdefault("metadata", {})["usage_metadata"] = (
                    usage_metadata
                )
        except Exception:  # noqa: BLE001
            logger.debug("Could not attach Decisions usage.", exc_info=True)
        return response


__all__ = ["PerplexityDecisions"]
