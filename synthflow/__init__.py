"""
synthflow – Synthetic sensor data generation.
"""

__version__ = "0.1.0"
__author__ = "synthflow"

from synthflow.core import SynFlow
from synthflow.exceptions import (
    SynError,
    SynConfigError,
    SynIngestError,
    SynImpError,
    SynBackendError,
    SynRouterError,
)

__all__ = [
    "SynFlow",
    "SynError",
    "SynConfigError",
    "SynIngestError",
    "SynImpError",
    "SynBackendError",
    "SynRouterError",
]
