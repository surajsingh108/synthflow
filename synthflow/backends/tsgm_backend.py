"""
TsgmBackend – TSGM-based synthesis backend.

Handles:
  - Data preparation: DataFrame → windowed numpy array
  - Model selection and training: TimeGAN, TimeVAE, RCGAN,
    WaveGAN, GaussianProcess, AR
  - Output building: numpy array → DataFrame with synthetic_ prefix
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthflow.backends.base import SynBackend
from synthflow.exceptions import SynBackendError


# ──── Data preparation utilities ──────────────────────────────────────────────

def window_dataframe(
    df: pd.DataFrame,
    signal_cols: list[str],
    seq_len: int,
    step: int | None = None,
) -> np.ndarray:
    """
    Slice a DataFrame into overlapping windows.

    Args:
        df          : input DataFrame
        signal_cols : columns to include (must be numeric)
        seq_len     : length of each window in rows
        step        : stride between windows (default: seq_len // 2)

    Returns:
        np.ndarray of shape (n_windows, seq_len, n_features)

    Raises:
        SynBackendError if fewer than 1 window can be produced.
    """
    if step is None:
        step = max(1, seq_len // 2)

    data = df[signal_cols].values.astype(np.float32)
    n_rows, n_features = data.shape

    if n_rows < seq_len:
        raise SynBackendError(
            f"Dataset has {n_rows} rows but seq_len={seq_len} requires "
            f"at least {seq_len} rows.",
            detail="Reduce seq_len or provide more data.",
        )

    windows = []
    start = 0
    while start + seq_len <= n_rows:
        windows.append(data[start : start + seq_len])
        start += step

    if not windows:
        raise SynBackendError(
            "No windows could be produced from the dataset.",
            detail=f"n_rows={n_rows}, seq_len={seq_len}, step={step}",
        )

    return np.stack(windows, axis=0)  # (n_windows, seq_len, n_features)


def dewindow_array(
    arr: np.ndarray,
    n_target_rows: int | None = None,
) -> np.ndarray:
    """
    Flatten windowed array back to 2D by taking non-overlapping slices.

    Args:
        arr           : shape (n_samples, seq_len, n_features)
        n_target_rows : if set, trim or pad output to this row count

    Returns:
        np.ndarray of shape (n_rows, n_features)
    """
    n_samples, seq_len, n_features = arr.shape
    # concatenate seq_len dimension: take each full window
    flat = arr.reshape(n_samples * seq_len, n_features)
    if n_target_rows is not None:
        if len(flat) >= n_target_rows:
            flat = flat[:n_target_rows]
        else:
            # pad by repeating last row
            pad = np.tile(flat[-1:], (n_target_rows - len(flat), 1))
            flat = np.vstack([flat, pad])
    return flat


def build_output_df(
    arr: np.ndarray,
    signal_cols: list[str],
    timestamp_col: str | None,
    source_df: pd.DataFrame,
    n_rows: int,
) -> pd.DataFrame:
    """
    Build output DataFrame from generated array.

    - Signal columns get synthetic_ prefix
    - Timestamp column is copied from source unchanged
    - Shape is (n_rows, len(signal_cols) + (1 if timestamp_col else 0))

    Args:
        arr           : generated array (n_samples, seq_len, n_features)
                        OR already flat (n_rows, n_features)
        signal_cols   : original signal column names
        timestamp_col : timestamp column name, or None
        source_df     : original DataFrame (for timestamp values)
        n_rows        : target row count in output

    Returns:
        DataFrame with synthetic_ prefixed signal columns.
    """
    # flatten if 3D
    if arr.ndim == 3:
        flat = dewindow_array(arr, n_target_rows=n_rows)
    else:
        flat = arr[:n_rows] if len(arr) >= n_rows else np.vstack(
            [arr, np.tile(arr[-1:], (n_rows - len(arr), 1))]
        )

    # build column mapping
    synth_cols = {f"synthetic_{col}": flat[:, i]
                  for i, col in enumerate(signal_cols)}
    out = pd.DataFrame(synth_cols)

    # add timestamp column unchanged
    if timestamp_col is not None and timestamp_col in source_df.columns:
        ts = source_df[timestamp_col].values
        if len(ts) >= n_rows:
            out.insert(0, timestamp_col, ts[:n_rows])
        else:
            # repeat timestamps if generated more rows than source
            repeats = (n_rows // len(ts)) + 1
            extended = np.tile(ts, repeats)[:n_rows]
            out.insert(0, timestamp_col, extended)

    return out.reset_index(drop=True)


def infer_seq_len(n_rows: int, sampling_rate_hz: float) -> int:
    """
    Choose a reasonable window length based on dataset size and frequency.

    Targets roughly 1-2 seconds of data per window, bounded between 8-128.
    """
    # aim for ~1 second of data
    target = int(sampling_rate_hz)
    # clamp to sensible range
    seq_len = max(8, min(128, target))
    # ensure at least 10 windows possible
    while n_rows // (seq_len // 2) < 10 and seq_len > 8:
        seq_len = seq_len // 2
    return seq_len


# ──── Model wrappers ─────────────────────────────────────────────────────────

class _TimeGANWrapper(SynBackend):
    """Wraps TSGM TimeGAN."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 24,
        num_layer: int = 3,
        batch_size: int = 128,
        device: str = "auto",
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.batch_size = batch_size
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 100, **kwargs) -> None:
        try:
            from tsgm.models.timeGAN import TimeGAN
        except ImportError as e:
            raise SynBackendError("TSGM TimeGAN not available.", detail=str(e)) from e
        try:
            self._model = TimeGAN(
                seq_len=self.seq_len,
                module="gru",
                hidden_dim=self.hidden_dim,
                n_features=self.n_features,
                num_layer=self.num_layer,
                batch_size=self.batch_size,
            )
            self._model.fit(X, epochs=epochs)
            self._fitted = True
        except Exception as exc:
            raise SynBackendError("TimeGAN training failed.", detail=str(exc)) from exc

    def generate(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise SynBackendError("TimeGAN must be fitted before generate().")
        try:
            return self._model.generate(n=n)
        except Exception as exc:
            raise SynBackendError("TimeGAN generation failed.", detail=str(exc)) from exc

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def name(self) -> str:
        return "TimeGAN"


class _TimeVAEWrapper(SynBackend):
    """Wraps TSGM TimeVAE (VAE-based)."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        latent_dim: int = 8,
        batch_size: int = 128,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 100, **kwargs) -> None:
        try:
            import tsgm
        except ImportError as e:
            raise SynBackendError("TSGM not available.", detail=str(e)) from e
        try:
            # TimeVAE is accessed via tsgm.models.cvae or tsgm.models.vae
            # Try both import paths for compatibility
            try:
                from tsgm.models.cvae import BetaVAE as TimeVAE
            except ImportError:
                from tsgm.models.vae import TimeVAE  # noqa: F401
            self._model = TimeVAE(
                seq_len=self.seq_len,
                feat_dim=self.n_features,
                latent_dim=self.latent_dim,
                batch_size=self.batch_size,
            )
            self._model.fit(X, epochs=epochs)
            self._fitted = True
        except Exception as exc:
            raise SynBackendError("TimeVAE training failed.", detail=str(exc)) from exc

    def generate(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise SynBackendError("TimeVAE must be fitted before generate().")
        try:
            return self._model.generate(n=n)
        except Exception as exc:
            raise SynBackendError("TimeVAE generation failed.", detail=str(exc)) from exc

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def name(self) -> str:
        return "TimeVAE"


class _GaussianProcessWrapper(SynBackend):
    """
    Probabilistic GP-based synthesis.
    Fits a multivariate Gaussian to the data and samples from it.
    Fast, CPU-only, no TSGM model needed.
    """

    def __init__(self, seq_len: int, n_features: int, random_seed: int = 42):
        self.seq_len = seq_len
        self.n_features = n_features
        self.random_seed = random_seed
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 1, **kwargs) -> None:
        # X: (n_windows, seq_len, n_features) → flatten to (n_windows, seq_len * n_features)
        flat = X.reshape(len(X), -1).astype(np.float64)
        self._mean = flat.mean(axis=0)
        self._cov = np.cov(flat, rowvar=False)
        # ensure positive definite
        self._cov += np.eye(self._cov.shape[0]) * 1e-6
        self._fitted = True

    def generate(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise SynBackendError("GaussianProcess must be fitted before generate().")
        rng = np.random.default_rng(self.random_seed)
        flat = rng.multivariate_normal(self._mean, self._cov, size=n)
        return flat.reshape(n, self.seq_len, self.n_features).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def name(self) -> str:
        return "GaussianProcess"


class _ARWrapper(SynBackend):
    """
    Autoregressive synthesis – fits AR(1) per feature and generates sequences.
    Fast, CPU-only, no TSGM model needed.
    """

    def __init__(self, seq_len: int, n_features: int, random_seed: int = 42):
        self.seq_len = seq_len
        self.n_features = n_features
        self.random_seed = random_seed
        self._phi: np.ndarray | None = None   # AR coefficients (n_features,)
        self._sigma: np.ndarray | None = None  # noise std (n_features,)
        self._mu: np.ndarray | None = None     # mean (n_features,)
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 1, **kwargs) -> None:
        # X: (n_windows, seq_len, n_features)
        # fit AR(1) per feature: x_t = phi * x_{t-1} + eps
        n_windows, seq_len, n_features = X.shape
        flat = X.reshape(-1, n_features)  # (n_windows * seq_len, n_features)
        self._mu = flat.mean(axis=0)
        centered = flat - self._mu

        phi = np.zeros(n_features)
        sigma = np.ones(n_features)
        for f in range(n_features):
            x = centered[:, f]
            if len(x) > 1 and np.std(x) > 0:
                # AR(1) via OLS
                x_lag = x[:-1]
                x_cur = x[1:]
                phi[f] = np.dot(x_lag, x_cur) / (np.dot(x_lag, x_lag) + 1e-8)
                residuals = x_cur - phi[f] * x_lag
                sigma[f] = max(np.std(residuals), 1e-6)

        self._phi = phi
        self._sigma = sigma
        self._fitted = True

    def generate(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise SynBackendError("AR must be fitted before generate().")
        rng = np.random.default_rng(self.random_seed)
        out = np.zeros((n, self.seq_len, self.n_features), dtype=np.float32)
        for i in range(n):
            x = np.zeros((self.seq_len, self.n_features))
            x[0] = self._mu
            for t in range(1, self.seq_len):
                noise = rng.normal(0, self._sigma)
                x[t] = self._mu + self._phi * (x[t - 1] - self._mu) + noise
            out[i] = x
        return out

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def name(self) -> str:
        return "AR"


# ──── Model factory ──────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type] = {
    "TimeGAN":        _TimeGANWrapper,
    "TimeVAE":        _TimeVAEWrapper,
    "GaussianProcess": _GaussianProcessWrapper,
    "AR":             _ARWrapper,
}


# ──── Main backend class ─────────────────────────────────────────────────────

class TsgmBackend:
    """
    Full TSGM synthesis pipeline.

    Handles data preparation, model fitting, generation, and output
    formatting. Returns a DataFrame with synthetic_ prefixed columns.

    Usage:
        backend = TsgmBackend()
        result_df = backend.run(
            df=clean_df,
            signal_cols=["accel_x", "accel_y"],
            timestamp_col="timestamp",
            model_name="GaussianProcess",
            n_samples=1000,
            random_seed=42,
        )
    """

    def run(
        self,
        df: pd.DataFrame,
        signal_cols: list[str],
        timestamp_col: str | None = None,
        model_name: str = "GaussianProcess",
        n_samples: int = 1000,
        epochs: int = 100,
        batch_size: int = 128,
        random_seed: int = 42,
        sampling_rate_hz: float = 100.0,
        seq_len: int | None = None,
    ) -> pd.DataFrame:
        """
        Run the full synthesis pipeline.

        Args:
            df              : clean imputed DataFrame (zero NaNs)
            signal_cols     : columns to synthesize
            timestamp_col   : timestamp column (preserved unchanged)
            model_name      : one of the _MODEL_REGISTRY keys
            n_samples       : number of output rows to generate
            epochs          : training epochs
            batch_size      : training batch size
            random_seed     : for reproducibility
            sampling_rate_hz: used to infer seq_len if not provided
            seq_len         : window size (inferred if None)

        Returns:
            DataFrame with synthetic_ prefixed signal columns.
        """
        if model_name not in _MODEL_REGISTRY:
            raise SynBackendError(
                f"Unknown model: '{model_name}'",
                detail=f"Available: {list(_MODEL_REGISTRY.keys())}",
            )

        # verify no NaNs
        if df[signal_cols].isna().any().any():
            raise SynBackendError(
                "Input DataFrame contains NaNs.",
                detail="Run SynImputer before TsgmBackend.",
            )

        n_rows = len(df)
        n_features = len(signal_cols)

        # infer seq_len
        if seq_len is None:
            seq_len = infer_seq_len(n_rows, sampling_rate_hz)

        # prepare training data
        X = window_dataframe(df, signal_cols, seq_len)

        # n_samples to generate in terms of windows
        n_windows_needed = max(1, (n_samples // seq_len) + 1)

        # instantiate model
        model_cls = _MODEL_REGISTRY[model_name]
        if model_name in ("GaussianProcess", "AR"):
            model = model_cls(
                seq_len=seq_len,
                n_features=n_features,
                random_seed=random_seed,
            )
        elif model_name == "TimeGAN":
            model = model_cls(
                seq_len=seq_len,
                n_features=n_features,
                batch_size=batch_size,
            )
        elif model_name == "TimeVAE":
            model = model_cls(
                seq_len=seq_len,
                n_features=n_features,
                batch_size=batch_size,
            )
        else:
            model = model_cls(
                seq_len=seq_len,
                n_features=n_features,
            )

        # fit
        model.fit(X, epochs=epochs)

        # generate
        X_synth = model.generate(n=n_windows_needed)

        # build output DataFrame
        return build_output_df(
            arr=X_synth,
            signal_cols=signal_cols,
            timestamp_col=timestamp_col,
            source_df=df,
            n_rows=n_samples,
        )
