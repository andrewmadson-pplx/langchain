"""Perplexity Decisions API: typed questions answered with probabilities.

`PerplexityDecisions` is a LangChain runnable that sends `Noul`, `Choice` and
`Score` questions about text, JSON, or messages and returns one typed answer per
question. It reaches the model through OpenRouter by default or through the
Perplexity API with `route="native"`.
"""

from langchain_perplexity.decisions._client import (
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
from langchain_perplexity.decisions.base import PerplexityDecisions
from langchain_perplexity.decisions.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    DecisionsEndpoint,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
    State,
    Usage,
)

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionsEndpoint",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "PerplexityDecisions",
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
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
]
