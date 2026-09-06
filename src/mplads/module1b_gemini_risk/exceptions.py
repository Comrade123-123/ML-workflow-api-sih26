"""Domain-specific exceptions for Module 1B (AI-Powered Project Cost & Risk
Assessment).

Mirrors the rest of the repository's exception design: explicit, typed
failures at each pipeline boundary, mapped by the router to an honest HTTP
error -- never a silently fabricated assessment.
"""


class Module1BError(Exception):
    """Base class for all Module 1B domain errors."""


class UnderspendModelUnavailableError(Module1BError):
    """Raised when the existing underspend-risk model artifact is not
    available. Module 1B does not train or substitute a model of its own --
    this simply propagates the existing model's own unavailability."""


class GeminiUnavailableError(Module1BError):
    """Raised when the Gemini API cannot be reached or used at all: missing
    API key, client construction failure, network timeout, or a non-2xx
    API error. Never converted into a fabricated assessment or a silent
    'LOW RISK' default."""


class GeminiResponseError(Module1BError):
    """Raised when Gemini responds, but its output is not usable: not valid
    JSON, or valid JSON that fails Pydantic validation against the expected
    AIAssessment schema. Also never converted into a fabricated assessment."""
