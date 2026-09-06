"""Domain-specific exceptions for Module 8 (Report Generation).

Entity-not-found is intentionally NOT redefined here -- it reuses Module 7's
EntityNotFoundError directly, since Module 8 resolves entities via Module 7's
public resolve_entity() rather than its own copy of that logic.
"""


class Module8Error(Exception):
    """Base class for all Module 8 domain errors."""


class UnknownReportTypeError(Module8Error):
    """Raised if a report_type has no registered generator (defensive --
    ReportType is a Pydantic enum, so FastAPI rejects an invalid value
    before this would ever fire)."""


class UnknownSubscriptionError(Module8Error):
    """Raised when a subscription_id is not in the static catalog."""
