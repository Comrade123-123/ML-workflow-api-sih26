"""Domain-specific exceptions for Module 9 (Alert Generation & Dispatch)."""


class Module9Error(Exception):
    """Base class for all Module 9 domain errors."""


class ProjectNotFoundError(Module9Error):
    """Raised when a project_id does not resolve to exactly one row in
    Works_Recommended.

    This covers both a genuinely unknown Work ID and the ambiguous case: the
    ~5,440 not-yet-sanctioned "NA-" placeholder rows share duplicate Work ID
    strings (one placeholder string can cover over a thousand distinct real
    recommendations). A project without a confirmed, uniquely-identified
    sanctioned Work ID cannot be alerted on or dispatched to -- there is no
    single project to act on.
    """


class InspectorNotFoundError(Module9Error):
    """Raised when an inspector_id is not in the roster."""


class InspectorNotAvailableError(Module9Error):
    """Raised when a dispatch is attempted against an inspector who is not
    currently AVAILABLE."""
