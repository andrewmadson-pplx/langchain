"""Fixed transport defaults for each `PerplexityDecisions` route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

RouteName: TypeAlias = Literal["openrouter", "native"]
"""Name of a supported route."""

OPENROUTER_REQUIRES_INSTRUCTIONS = True
"""Whether the openrouter route rejects `Noul` questions that carry only criteria.

The live contract spike on 2026-10-19 showed OpenRouter returns 400 for a `noul`
without `instructions` while the native route accepts it. The check is applied at
request-build time and only on the openrouter route; instructions are never
synthesized from criteria. Set to `False` if OpenRouter starts accepting them.
"""

OPENROUTER_TOP_LEVEL_IMAGES_ONLY = True
"""Whether the openrouter route rejects image parts outside the top-level `state`.

The live run on 2026-10-27 showed OpenRouter returns 400 for an `image_url` part
nested inside a `state` object or array, or inside a question's `instructions`
or `criteria`, while the native route reads image parts anywhere in `state`. The
check is applied at request-build time and only on the openrouter route; image
parts are never moved. Set to `False` if OpenRouter starts accepting them.
"""

DEFAULT_APP_URL = "https://docs.langchain.com"
"""Default `HTTP-Referer` sent on the openrouter route."""

DEFAULT_APP_TITLE = "LangChain"
"""Default `X-Title` sent on the openrouter route."""

APP_URL_ENV_VAR = "OPENROUTER_APP_URL"
"""Environment variable that overrides `DEFAULT_APP_URL`."""

APP_TITLE_ENV_VAR = "OPENROUTER_APP_TITLE"
"""Environment variable that overrides `DEFAULT_APP_TITLE`."""


@dataclass(frozen=True)
class RouteSpec:
    """Transport defaults for one route.

    Attributes:
        name: Route name, `openrouter` or `native`.
        base_url: Root URL without a trailing slash.
        path: Request path, starting with `/`.
        api_key_env_vars: Environment variables consulted for the key, in order.
        default_model: Model sent when `PerplexityDecisions.model` is `None`.
        ls_provider: Value of the `ls_provider` tracing metadata key.
        extra_allowlist: Top-level request keys accepted through `extra`.
        request_id_header: Response header carrying the request ID, or `None`.
        request_id_in_body: Whether the response body `id` field is the request ID.
            When `True` the body is checked first and the header is the fallback.
        attribution_headers: Whether `HTTP-Referer` and `X-Title` are sent.
    """

    name: RouteName
    base_url: str
    path: str
    api_key_env_vars: tuple[str, ...]
    default_model: str
    ls_provider: str
    extra_allowlist: frozenset[str]
    request_id_header: str | None
    request_id_in_body: bool
    attribution_headers: bool

    @property
    def url(self) -> str:
        """Return the full request URL for the route defaults."""
        return join_url(self.base_url, self.path)


def join_url(base_url: str, path: str) -> str:
    """Join a base URL and a path without duplicating or dropping slashes.

    Args:
        base_url: Root URL. A trailing slash is removed.
        path: Request path. Must start with `/`.

    Returns:
        The joined URL.

    Raises:
        ValueError: If `path` does not start with `/`.
    """
    if not path.startswith("/"):
        message = "Decisions endpoint path must start with '/'."
        raise ValueError(message)
    return base_url.rstrip("/") + path


ROUTES: dict[RouteName, RouteSpec] = {
    "openrouter": RouteSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api",
        path="/alpha/decisions",
        api_key_env_vars=("OPENROUTER_API_KEY",),
        default_model="perplexity/pplx-decider-v1.2",
        ls_provider="openrouter",
        extra_allowlist=frozenset({"provider", "session_id", "trace", "user"}),
        request_id_header="x-request-id",
        request_id_in_body=True,
        attribution_headers=True,
    ),
    "native": RouteSpec(
        name="native",
        base_url="https://api.perplexity.ai",
        path="/v1/decisions",
        api_key_env_vars=("PPLX_API_KEY", "PERPLEXITY_API_KEY"),
        default_model="pplx-decider-v1.2",
        ls_provider="perplexity",
        extra_allowlist=frozenset(),
        request_id_header="x-request-id",
        request_id_in_body=False,
        attribution_headers=False,
    ),
}
"""Route defaults keyed by route name."""


__all__ = [
    "APP_TITLE_ENV_VAR",
    "APP_URL_ENV_VAR",
    "DEFAULT_APP_TITLE",
    "DEFAULT_APP_URL",
    "OPENROUTER_REQUIRES_INSTRUCTIONS",
    "OPENROUTER_TOP_LEVEL_IMAGES_ONLY",
    "ROUTES",
    "RouteName",
    "RouteSpec",
    "join_url",
]
