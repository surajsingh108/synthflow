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
    When seq_len=1 (tabular mode), each row becomes its own window.
    """
    if step is None:
        step = max(1, seq_len // 2) if seq_len > 1 else 1

    data = df[signal_cols].values.astype(np.float32)
    n_rows, n_features = data.shape

    if n_rows < seq_len:
        raise SynBackendError(
            f"Dataset has {n_rows} rows but seq_len={seq_len} requires "
            f"at least {seq_len} rows.",
            detail="Reduce seq_len or provide more data.",
        )

    # seq_len=1: each row is its own window (tabular mode)
    if seq_len == 1:
        return data.reshape(n_rows, 1, n_features)  # (n_rows, 1, n_features)

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


def resolve_seq_len(
    n_rows: int,
    sampling_rate_hz: float,
    data_type: str = "auto",
    seq_len_override: int | None = None,
) -> int:
    """
    Determine the window length to use.

    Priority:
      1. Explicit seq_len_override (from SynConfig.seq_len)
      2. data_type == "tabular" → always 1
      3. Auto-infer from sampling_rate_hz
    """
    if seq_len_override is not None:
        return seq_len_override

    if data_type == "tabular":
        return 1

    # auto-infer for timeseries
    target = int(sampling_rate_hz)
    seq_len = max(8, min(128, target))
    while n_rows // (max(seq_len // 2, 1)) < 10 and seq_len > 8:
        seq_len = seq_len // 2
    return seq_len


def infer_seq_len(n_rows: int, sampling_rate_hz: float) -> int:
    """Legacy alias for resolve_seq_len. Kept for backward compatibility."""
    return resolve_seq_len(n_rows, sampling_rate_hz)


def enforce_bounds(
    synth_df: pd.DataFrame,
    signal_cols: list[str],
    real_df: pd.DataFrame,
    auto_bounds: bool = True,
    column_bounds: dict | None = None,
) -> pd.DataFrame:
    """
    Clip synthetic columns to physical bounds.

    Priority per column:
      1. column_bounds[col] – explicit user-specified (lo, hi)
      2. auto_bounds – infer from real data:
           lo = 0 if real_min >= 0, else real_min
           hi = real_max
      3. No bounds – leave as-is

    Args:
        synth_df      : DataFrame with synthetic_ prefixed columns
        signal_cols   : original column names (without synthetic_)
        real_df       : real DataFrame used to infer bounds
        auto_bounds   : whether to auto-infer bounds
        column_bounds : explicit {col: (lo, hi)} overrides

    Returns:
        DataFrame with bounds enforced.
    """
    col_bounds = column_bounds or {}
    out = synth_df.copy()

    for col in signal_cols:
        synth_col = f"synthetic_{col}"
        if synth_col not in out.columns:
            continue

        if col in col_bounds:
            lo, hi = col_bounds[col]
        elif auto_bounds and col in real_df.columns:
            real_vals = real_df[col].dropna()
            if len(real_vals) == 0:
                continue
            lo = 0.0 if real_vals.min() >= 0 else float(real_vals.min())
            hi = float(real_vals.max())
        else:
            continue

        out[synth_col] = out[synth_col].clip(lo, hi)

    return out


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


class _GaussianCopulaWrapper(SynBackend):
    """
    Gaussian Copula synthesizer for tabular data.

    Steps:
      fit()      : quantile-transform each feature to N(0,1),
                   fit a multivariate Gaussian to the transformed data
      generate() : sample from the Gaussian, inverse-transform back
                   – preserves true marginal distributions (heavy tails,
                     bimodal peaks, etc.) AND linear correlations
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        random_seed: int = 42,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.random_seed = random_seed
        self._qt: object | None = None
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 1, **kwargs) -> None:
        """
        X: (n_samples, seq_len, n_features) – flattened to 2D for fitting.
        """
        try:
            from sklearn.preprocessing import QuantileTransformer
        except ImportError as e:
            raise SynBackendError(
                "scikit-learn required for GaussianCopula.",
                detail="pip install scikit-learn",
            ) from e

        flat = X.reshape(-1, self.n_features).astype(np.float64)
        n = len(flat)

        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(n, 1000),
            random_state=self.random_seed,
        )
        flat_normal = self._qt.fit_transform(flat)

        self._mean = flat_normal.mean(axis=0)
        self._cov  = np.cov(flat_normal, rowvar=False)
        if self._cov.ndim == 0:
            self._cov = np.array([[float(self._cov)]])
        self._cov += np.eye(self._cov.shape[0]) * 1e-6
        self._fitted = True

    def generate(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise SynBackendError("GaussianCopula must be fitted before generate().")
        rng = np.random.default_rng(self.random_seed)
        flat_normal = rng.multivariate_normal(self._mean, self._cov, size=n)
        flat_normal  = np.clip(flat_normal, -4.0, 4.0)
        flat_original = self._qt.inverse_transform(flat_normal)
        return flat_original.reshape(n, self.seq_len, self.n_features).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def name(self) -> str:
        return "GaussianCopula"


class _TabularVAEWrapper(SynBackend):
    """
    Variational Autoencoder for tabular / cross-sectional data.

    Fixes vs original:
      - QuantileTransformer normalisation (robust to heavy tails and extreme ranges)
      - KL annealing: weight ramps from 0 → 0.0001 over first 30% of epochs
        (prevents posterior collapse – decoder learns to reconstruct before
         being forced toward N(0,1))
      - Smaller latent_dim: n_features // 3, capped at 8
        (reduces underdetermination)
      - Deeper network with LayerNorm for stability
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        latent_dim: int | None = None,
        hidden_dim: int = 128,
        batch_size: int = 128,
        random_seed: int = 42,
    ):
        self.seq_len     = seq_len
        self.n_features  = n_features
        # smaller latent_dim prevents underconstrained collapse
        self.latent_dim  = latent_dim if latent_dim is not None else max(2, min(n_features // 3, 8))
        self.hidden_dim  = hidden_dim
        self.batch_size  = batch_size
        self.random_seed = random_seed
        self._qt         = None    # QuantileTransformer
        self._enc        = None
        self._dec        = None
        self._mu_lay     = None
        self._lv_lay     = None
        self._losses: list[float] = []
        self._fitted = False

    def fit(self, X: np.ndarray, epochs: int = 200, **kwargs) -> None:
        """
        X: (n_samples, seq_len, n_features).
        Flattened to 2D for training.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import torch.nn.functional as F
        except ImportError as e:
            raise SynBackendError(
                "PyTorch is required for TabularVAE.",
                detail="pip install torch",
            ) from e

        flat = X.reshape(-1, self.n_features).astype(np.float32)
        n, p = flat.shape

        # – QuantileTransformer: robust to heavy tails and extreme ranges –
        try:
            from sklearn.preprocessing import QuantileTransformer
        except ImportError as e:
            raise SynBackendError(
                "scikit-learn required for TabularVAE normalisation.",
                detail=str(e),
            ) from e

        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(n, 1000),
            random_state=self.random_seed,
        )
        Xn = self._qt.fit_transform(flat).astype(np.float32)

        # – Architecture ––––––––––––––––––––––––––––––––––––––––––––––––––––
        torch.manual_seed(self.random_seed)
        ld, hd = self.latent_dim, self.hidden_dim

        # Deeper encoder/decoder with LayerNorm for training stability
        enc = nn.Sequential(
            nn.Linear(p, hd), nn.LayerNorm(hd), nn.ReLU(),
            nn.Linear(hd, hd), nn.LayerNorm(hd), nn.ReLU(),
        )
        mu_layer = nn.Linear(hd, ld)
        lv_layer = nn.Linear(hd, ld)
        dec = nn.Sequential(
            nn.Linear(ld, hd), nn.LayerNorm(hd), nn.ReLU(),
            nn.Linear(hd, hd), nn.LayerNorm(hd), nn.ReLU(),
            nn.Linear(hd, p),
        )

        params = (
            list(enc.parameters()) +
            list(mu_layer.parameters()) +
            list(lv_layer.parameters()) +
            list(dec.parameters())
        )
        opt = optim.Adam(params, lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        Xt = torch.tensor(Xn)
        losses = []

        # KL annealing: ramp from 0 to KL_MAX over the first 30% of epochs
        KL_MAX    = 0.0001
        KL_WARMUP = int(epochs * 0.30)

        for ep in range(epochs):
            idx = torch.randperm(n)[: self.batch_size]
            xb  = Xt[idx]

            # forward
            h        = enc(xb)
            mu_z     = mu_layer(h)
            lv_z     = lv_layer(h).clamp(-4, 4)   # clamp for stability
            std_z    = torch.exp(0.5 * lv_z)
            z        = mu_z + std_z * torch.randn_like(std_z)
            recon    = dec(z)

            # losses
            recon_loss = F.mse_loss(recon, xb)
            kl_loss    = -0.5 * (1 + lv_z - mu_z.pow(2) - lv_z.exp()).mean()

            # KL annealing weight
            if KL_WARMUP > 0:
                kl_weight = KL_MAX * min(1.0, ep / KL_WARMUP)
            else:
                kl_weight = KL_MAX

            loss = recon_loss + kl_weight * kl_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            scheduler.step()

            losses.append(float(loss.item()))

        self._enc    = enc
        self._dec    = dec
        self._mu_lay = mu_layer
        self._lv_lay = lv_layer
        self._losses = losses
        self._fitted = True

    def generate(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise SynBackendError("TabularVAE must be fitted before generate().")
        try:
            import torch
        except ImportError as e:
            raise SynBackendError("PyTorch required.", detail=str(e)) from e

        torch.manual_seed(self.random_seed)
        z = torch.randn(n, self.latent_dim)
        with torch.no_grad():
            out_normal = self._dec(z).numpy()

        # inverse-transform from Gaussian back to original distribution
        out_original = self._qt.inverse_transform(out_normal)
        return out_original.reshape(n, self.seq_len, self.n_features).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def name(self) -> str:
        return "TabularVAE"

    @property
    def losses(self) -> list[float]:
        return self._losses


# ──── Model factory ──────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type] = {
    "TimeGAN":         _TimeGANWrapper,
    "TimeVAE":         _TimeVAEWrapper,
    "GaussianProcess": _GaussianProcessWrapper,
    "AR":              _ARWrapper,
    "GaussianCopula":  _GaussianCopulaWrapper,
    "TabularVAE":      _TabularVAEWrapper,
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
        data_type: str = "auto",
        auto_bounds: bool = True,
        column_bounds: dict | None = None,
    ) -> pd.DataFrame:
        """
        Run the full synthesis pipeline.

        Args:
            df              : clean imputed DataFrame (zero NaNs)
            signal_cols     : columns to synthesize
            timestamp_col   : timestamp column (preserved unchanged)
            model_name      : model to use
            n_samples       : number of output rows to generate
            epochs          : training epochs
            batch_size      : training batch size
            random_seed     : for reproducibility
            sampling_rate_hz: used to infer seq_len if not provided
            seq_len         : explicit window length (1 = tabular mode)
            data_type       : "timeseries", "tabular", or "auto"
            auto_bounds     : infer and enforce physical bounds from real data
            column_bounds   : explicit {col: (lo, hi)} bounds per column
        """
        if model_name not in _MODEL_REGISTRY:
            raise SynBackendError(
                f"Unknown model: '{model_name}'",
                detail=f"Available: {list(_MODEL_REGISTRY.keys())}",
            )

        if df[signal_cols].isna().any().any():
            raise SynBackendError(
                "Input DataFrame contains NaNs.",
                detail="Run SynImputer before TsgmBackend.",
            )

        n_rows     = len(df)
        n_features = len(signal_cols)

        # resolve seq_len
        # tabular models always use seq_len=1
        tabular_models = {"GaussianCopula", "TabularVAE"}
        if model_name in tabular_models:
            effective_seq_len = 1
        else:
            effective_seq_len = resolve_seq_len(
                n_rows, sampling_rate_hz, data_type, seq_len
            )

        # prepare training data
        X = window_dataframe(df, signal_cols, effective_seq_len)

        # number of windows to generate so we have enough rows
        if effective_seq_len == 1:
            n_windows_needed = n_samples
        else:
            n_windows_needed = max(1, (n_samples // effective_seq_len) + 1)

        # instantiate model
        model_cls = _MODEL_REGISTRY[model_name]
        common = dict(seq_len=effective_seq_len, n_features=n_features)

        if model_name in ("GaussianProcess", "AR"):
            model = model_cls(**common, random_seed=random_seed)
        elif model_name == "GaussianCopula":
            model = model_cls(**common, random_seed=random_seed)
        elif model_name == "TabularVAE":
            model = model_cls(
                **common,
                # latent_dim computed inside class: max(2, min(n_features//3, 8))
                hidden_dim=min(n_features * 8, 512),
                batch_size=batch_size,
                random_seed=random_seed,
            )
        elif model_name == "TimeGAN":
            model = model_cls(**common, batch_size=batch_size)
        elif model_name == "TimeVAE":
            model = model_cls(**common, batch_size=batch_size)
        else:
            model = model_cls(**common)

        # fit and generate
        model.fit(X, epochs=epochs)
        X_synth = model.generate(n=n_windows_needed)

        # build output DataFrame
        out_df = build_output_df(
            arr=X_synth,
            signal_cols=signal_cols,
            timestamp_col=timestamp_col,
            source_df=df,
            n_rows=n_samples,
        )

        # enforce bounds
        if auto_bounds or column_bounds:
            out_df = enforce_bounds(
                synth_df=out_df,
                signal_cols=signal_cols,
                real_df=df,
                auto_bounds=auto_bounds,
                column_bounds=column_bounds or {},
            )

        return out_df
