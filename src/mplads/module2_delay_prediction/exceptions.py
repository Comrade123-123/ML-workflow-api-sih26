"""Domain-specific exceptions for Module 2 (Delayed Project Prediction).

Mirrors Module 1's exception design: explicit, typed failures at each
pipeline boundary instead of generic exceptions or silent fallbacks.
"""


class Module2Error(Exception):
    """Base class for all Module 2 domain errors."""


class DataValidationError(Module2Error):
    """Raised when input data does not match the expected structural schema."""


class TargetValidationError(Module2Error):
    """Raised when the peer-relative delay labeling does not qualify for training.

    Fires when, after labeling, one or more of ON_TRACK/AT_RISK/LIKELY_DELAYED
    does not meet the minimum sample count -- training must stop rather than
    proceed on a degenerate class split.
    """


class ModelNotAvailableError(Module2Error):
    """Raised when a prediction is requested but no trained model artifact exists."""
