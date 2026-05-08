"""
SynError hierarchy – all synthflow exceptions inherit from SynError
so callers can catch broadly or narrowly.

Usage:
    from synthflow.exceptions import SynConfigError
    raise SynConfigError("domain must be one of: industrial, medical, iot")
"""


class SynError(Exception):
    """Base exception for all synthflow errors."""

    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(message)

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} – {self.detail}"
        return self.message


class SynConfigError(SynError):
    """Raised when SynConfig is invalid or missing required fields."""
    pass


class SynIngestError(SynError):
    """Raised when data ingestion fails (bad file, wrong format, etc.)."""
    pass


class SynImpError(SynError):
    """Raised when imputation fails or produces invalid output."""
    pass


class SynBackendError(SynError):
    """Raised when a synthesis backend fails (OOM, bad params, etc.)."""
    pass


class SynRouterError(SynError):
    """Raised when no suitable backend can be selected for the config."""
    pass
