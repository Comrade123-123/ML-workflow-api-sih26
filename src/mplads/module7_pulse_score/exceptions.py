"""Domain-specific exceptions for Module 7 (Development Pulse Score)."""


class Module7Error(Exception):
    """Base class for all Module 7 domain errors."""


class EntityNotFoundError(Module7Error):
    """Raised when the given constituency/MP identifier does not resolve to
    any known LokSabha constituency in the MPLADS works data.

    This includes Rajya Sabha MP names: they represent states, not
    constituencies, and no work record in this dataset is attributed to
    them, so there is no project data to score.
    """
