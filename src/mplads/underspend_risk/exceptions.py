"""Domain-specific exceptions for the MPLADS Expenditure Utilization Risk
("underspend_risk") capability.

Mirrors Module 1 and Module 2's exception design: explicit, typed failures
at each pipeline boundary instead of generic exceptions or silent fallbacks.
"""


class UnderspendRiskError(Exception):
    """Base class for all underspend_risk domain errors."""


class DataValidationError(UnderspendRiskError):
    """Raised when input data does not match the expected structural schema."""


class ModelNotAvailableError(UnderspendRiskError):
    """Raised when a prediction is requested but no trained model artifact
    (and its accompanying historical-rate lookup table) exists.

    The API must surface this as an explicit "unavailable" response rather
    than fabricating a prediction.
    """
