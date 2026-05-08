"""
SynBackend – abstract base class for all synthesis backends.

All backends must implement fit() and generate().
Data arrives as numpy arrays of shape (n_samples, seq_len, n_features).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SynBackend(ABC):
    """
    Abstract base class for synthflow synthesis backends.

    Subclasses implement fit() and generate() for a specific
    generative modelling library (TSGM, SDV, Gretel, etc.).
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        **kwargs,
    ) -> None:
        """
        Train the generative model.

        Args:
            X      : training data, shape (n_samples, seq_len, n_features)
            epochs : training epochs / steps
            **kwargs: model-specific parameters
        """

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """
        Generate synthetic sequences.

        Args:
            n : number of sequences to generate

        Returns:
            np.ndarray of shape (n, seq_len, n_features)
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
