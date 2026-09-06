"""Domain-specific exceptions for Module 1 (Cost Overrun Detection).

These are raised deliberately at clear boundaries (schema checks, target
validation, model lookup) so callers get an explicit, typed failure instead
of a generic exception or a silently wrong/fabricated result.
"""


class Module1Error(Exception):
    """Base class for all Module 1 domain errors."""


class DataValidationError(Module1Error):
    """Raised when input data does not match the expected structural schema.

    Examples: a required column is missing, a column has an unexpected
    dtype, or a join key does not resolve.
    """


class TargetValidationError(Module1Error):
    """Raised when the regression target does not qualify for training.

    This is the fail-fast gate: it fires when the historical data does not
    contain enough genuine positive-overrun examples (actual cost >
    sanctioned cost) to support training a cost-overrun regression model.
    Training must stop here rather than substituting a different target.
    """


class ModelNotAvailableError(Module1Error):
    """Raised when a prediction is requested but no trained model artifact exists.

    The API must surface this as an explicit "unavailable" response rather
    than fabricating a prediction.
    """
