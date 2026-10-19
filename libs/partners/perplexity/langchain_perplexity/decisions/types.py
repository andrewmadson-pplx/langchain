"""Question, answer, request and response types for the Decisions API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self, TypedDict

MAX_QUESTIONS = 128
"""Largest number of questions accepted in one request."""

MAX_CHOICE_OPTIONS = 255
"""Largest number of options accepted by a `Choice` question."""

MAX_SCORE_LEVELS = 10
"""Largest number of levels accepted by a `Score` question."""

_QuestionContent: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
_StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | BaseMessage
    | Sequence["_StateValue"]
    | dict[str, "_StateValue"]
    | None
)

State: TypeAlias = str | BaseMessage | Sequence[_StateValue] | dict[str, _StateValue]
"""Content every question in a request refers to.

The API accepts a string, a JSON object, or a JSON array. LangChain `BaseMessage`
objects and message sequences may appear at the root or nested inside objects and
arrays; they are converted to role/content JSON before the request is sent.
"""


class NoulCriteria(BaseModel):
    """Descriptions of the yes and no outcomes of a `Noul` question.

    Either description may be omitted. Values accept any JSON-compatible content.
    """

    model_config = ConfigDict(populate_by_name=True)

    true: JsonValue = None
    """What counts as yes, or `None` when no clarification is needed."""

    false: JsonValue = None
    """What counts as no, or `None` when no clarification is needed."""

    def has_content(self) -> bool:
        """Return whether at least one outcome is described."""
        return self.true is not None or self.false is not None


class Noul(BaseModel):
    """A yes or no question answered with the probability of yes.

    Give `instructions`, `criteria`, or both. The answer is a probability from `0`
    to `1`; values near `0.5` mean the model is unsure. The threshold you compare
    against belongs to your application.

    Example:
        ```python
        Noul(instructions="Does the review report a product defect?")
        ```
    """

    type: Literal["noul"] = "noul"
    """Wire discriminator for a yes or no question."""

    instructions: _QuestionContent | None = None
    """The question to answer, as text or structured JSON."""

    criteria: NoulCriteria | None = None
    """Optional descriptions of what counts as yes and as no."""

    @model_validator(mode="after")
    def _require_instructions_or_criteria(self) -> Self:
        if self.instructions is None and not (
            self.criteria is not None and self.criteria.has_content()
        ):
            message = "Noul requires `instructions`, `criteria`, or both."
            raise ValueError(message)
        return self


class Choice(BaseModel):
    """Pick one option from a fixed set.

    The answer names the most likely option and, when the route returns them, a
    probability for every option and a confidence value.

    Example:
        ```python
        Choice(
            instructions="Which team should handle this ticket?",
            criteria={"billing": "Charges and refunds", "other": None},
        )
        ```
    """

    type: Literal["choice"] = "choice"
    """Wire discriminator for a categorical question."""

    instructions: _QuestionContent
    """The decision to make, as text or structured JSON."""

    criteria: dict[str, JsonValue] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)
    """Option names mapped to descriptions. Use `None` to let the name speak."""

    @model_validator(mode="after")
    def _require_non_empty_option_names(self) -> Self:
        if any(not name for name in self.criteria):
            message = "Choice option names must be non-empty strings."
            raise ValueError(message)
        return self


class Score(BaseModel):
    """Rate content on an ordered rubric.

    Levels are numbered from zero in the order given. The answer is the expected
    level and may fall between two integers. A single level is legal; the answer is
    then always `0` with probability `1`.

    Example:
        ```python
        Score(
            instructions="How severe is the reported problem?",
            criteria=["Cosmetic", "Inconvenient", "Product unusable"],
        )
        ```
    """

    type: Literal["score"] = "score"
    """Wire discriminator for an ordinal question."""

    instructions: _QuestionContent
    """The judgment to make, as text or structured JSON."""

    criteria: list[JsonValue] = Field(min_length=1, max_length=MAX_SCORE_LEVELS)
    """One to ten ordered level descriptions, starting at level zero."""


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]
"""Any question type accepted by `PerplexityDecisions`."""


class DecisionsEndpoint(BaseModel):
    """Explicit destination for advanced deployments.

    Setting an endpoint on `PerplexityDecisions` replaces the route's base URL and
    path. The class then requires an explicit `api_key` and reports
    `ls_provider="custom"` in traces.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(min_length=1)
    """Root URL, for example `https://gateway.example.com/api`."""

    path: str = Field(min_length=1)
    """Request path. Must start with `/`."""

    @model_validator(mode="after")
    def _validate_parts(self) -> Self:
        if not self.base_url.startswith(("http://", "https://")):
            message = "DecisionsEndpoint.base_url must start with http:// or https://."
            raise ValueError(message)
        if not self.path.startswith("/"):
            message = "DecisionsEndpoint.path must start with '/'."
            raise ValueError(message)
        return self


class DecisionRequest(TypedDict):
    """Input for one `PerplexityDecisions` invocation."""

    state: State
    """Text, JSON, or LangChain messages to decide about."""

    questions: dict[str, Question]
    """One to 128 questions keyed by non-empty, unique names."""


class NoulAnswer(BaseModel):
    """Answer to a `Noul` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"]
    """Wire discriminator for a yes or no answer."""

    noul: float = Field(ge=0.0, le=1.0)
    """Probability of yes, from `0` to `1`."""


class ChoiceAnswer(BaseModel):
    """Answer to a `Choice` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"]
    """Wire discriminator for a categorical answer."""

    choice: str
    """The option with the highest probability."""

    probabilities: dict[str, float] | None = None
    """Probability per option, when returned."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """The model's certainty from `0` to `1`, when returned."""


class ScoreAnswer(BaseModel):
    """Answer to a `Score` question.

    `legend` and `probabilities` arrive keyed by string level numbers on the wire
    and are coerced to integer keys here.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["score"]
    """Wire discriminator for an ordinal answer."""

    score: float
    """Probability-weighted average of the level indices."""

    legend: dict[int, JsonValue] | None = None
    """Level index mapped back to the rubric entry, when returned."""

    probabilities: dict[int, float] | None = None
    """Probability per level index, when returned."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """The model's certainty from `0` to `1`, when returned."""


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
"""Any answer type returned by `PerplexityDecisions`."""


class Usage(BaseModel):
    """Token usage and cost reported for one request.

    Every field is optional; a missing value means the route did not report it.
    """

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    """Input tokens processed."""

    output_tokens: int | None = None
    """Output tokens produced."""

    cost: float | None = None
    """Cost reported by the route, in its own currency units."""


class DecisionResponse(BaseModel):
    """Answers and metadata from one request.

    Only `answers` is required. Fields the route did not return are `None`. Top-level
    fields this model does not know about are kept in `raw`.
    """

    answers: dict[str, Answer]
    """One answer per question, keyed by the question name."""

    id: str | None = None
    """Response ID from the body, when returned."""

    model: str | None = None
    """Model that answered, when returned."""

    provider: str | None = None
    """Upstream provider name, when returned."""

    usage: Usage | None = None
    """Token usage, when returned."""

    request_id: str | None = None
    """Request ID taken from the body or a response header, by route."""

    raw: dict[str, JsonValue] = Field(default_factory=dict)
    """Top-level response fields not modelled above."""

    @model_validator(mode="before")
    @classmethod
    def _collect_unknown_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        known = cls.model_fields
        unknown = {k: v for k, v in data.items() if k not in known}
        if not unknown:
            return data
        kept = {k: v for k, v in data.items() if k in known}
        kept["raw"] = {**unknown, **(kept.get("raw") or {})}
        return kept

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Return `Noul` answers keyed by question name."""
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Return `Choice` answers keyed by question name."""
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Return `Score` answers keyed by question name."""
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}


__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_QUESTIONS",
    "MAX_SCORE_LEVELS",
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionsEndpoint",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
]
