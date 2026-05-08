"""
Domain scoring tables – define how each domain influences model scores.

Each entry maps model_name → score (0-3).
Higher score = better fit for this domain.
"""

from __future__ import annotations

# All supported models in scoring order
ALL_MODELS = [
    "GaussianProcess",
    "AR",
    "TimeVAE",
    "RCGAN",
    "TimeGAN",
    "WaveGAN",
]

# VRAM requirements in GB (approximate, at default batch size 128)
MODEL_VRAM_GB: dict[str, float] = {
    "GaussianProcess": 0.0,   # CPU only
    "AR":              0.0,   # CPU only
    "TimeVAE":         2.5,
    "RCGAN":           3.5,
    "TimeGAN":         5.0,
    "WaveGAN":         7.0,
}

# OOM fallback chains – if primary model OOMs, try these in order
OOM_FALLBACK: dict[str, list[str]] = {
    "TimeGAN":        ["TimeVAE", "GaussianProcess"],
    "WaveGAN":        ["TimeGAN", "TimeVAE", "GaussianProcess"],
    "RCGAN":          ["TimeVAE", "GaussianProcess"],
    "TimeVAE":        ["GaussianProcess"],
    "GaussianProcess":[],
    "AR":             [],
}

# ── Factor 1: Dataset size scoring ──────────────────────────────────────────
# n_rows thresholds
SIZE_TINY   = 500
SIZE_SMALL  = 5_000
SIZE_LARGE  = 50_000

def score_size(model: str, n_rows: int) -> int:
    if n_rows < SIZE_TINY:
        return {
            "GaussianProcess": 3, "AR": 3,
            "TimeVAE": 1, "RCGAN": 0,
            "TimeGAN": 0, "WaveGAN": 0,
        }[model]
    elif n_rows < SIZE_SMALL:
        return {
            "GaussianProcess": 2, "AR": 2,
            "TimeVAE": 3, "RCGAN": 2,
            "TimeGAN": 1, "WaveGAN": 1,
        }[model]
    elif n_rows < SIZE_LARGE:
        return {
            "GaussianProcess": 1, "AR": 1,
            "TimeVAE": 3, "RCGAN": 3,
            "TimeGAN": 3, "WaveGAN": 2,
        }[model]
    else:
        return {
            "GaussianProcess": 0, "AR": 0,
            "TimeVAE": 2, "RCGAN": 3,
            "TimeGAN": 3, "WaveGAN": 3,
        }[model]


# ── Factor 2: Signal complexity scoring ─────────────────────────────────────
# driven by sampling_rate_hz and n_signal_cols

FREQ_LOW  = 10.0   # Hz
FREQ_HIGH = 500.0  # Hz
COLS_MANY = 5      # channels

def score_complexity(
    model: str,
    sampling_rate_hz: float,
    n_signal_cols: int,
) -> int:
    # audio / very high frequency
    if sampling_rate_hz > FREQ_HIGH:
        return {
            "GaussianProcess": 0, "AR": 0,
            "TimeVAE": 1, "RCGAN": 1,
            "TimeGAN": 2, "WaveGAN": 3,
        }[model]
    # low frequency, simple signal
    if sampling_rate_hz <= FREQ_LOW and n_signal_cols <= 2:
        return {
            "GaussianProcess": 3, "AR": 3,
            "TimeVAE": 2, "RCGAN": 1,
            "TimeGAN": 1, "WaveGAN": 0,
        }[model]
    # many correlated channels
    if n_signal_cols > COLS_MANY:
        return {
            "GaussianProcess": 0, "AR": 0,
            "TimeVAE": 2, "RCGAN": 2,
            "TimeGAN": 3, "WaveGAN": 1,
        }[model]
    # mid-range – the sweet spot for TimeVAE
    return {
        "GaussianProcess": 1, "AR": 1,
        "TimeVAE": 3, "RCGAN": 2,
        "TimeGAN": 2, "WaveGAN": 1,
    }[model]


# ── Factor 3: Domain scoring ────────────────────────────────────────────────

DOMAIN_SCORES: dict[str, dict[str, int]] = {
    "industrial": {
        "GaussianProcess": 1, "AR": 1,
        "TimeVAE": 3, "RCGAN": 2,
        "TimeGAN": 3, "WaveGAN": 0,
    },
    "iot": {
        "GaussianProcess": 2, "AR": 2,
        "TimeVAE": 3, "RCGAN": 2,
        "TimeGAN": 2, "WaveGAN": 0,
    },
    "medical": {
        "GaussianProcess": 1, "AR": 1,
        "TimeVAE": 3, "RCGAN": 2,
        "TimeGAN": 2, "WaveGAN": 0,
    },
    "audio": {
        "GaussianProcess": 0, "AR": 0,
        "TimeVAE": 1, "RCGAN": 1,
        "TimeGAN": 2, "WaveGAN": 3,
    },
    "financial": {
        "GaussianProcess": 1, "AR": 3,
        "TimeVAE": 2, "RCGAN": 2,
        "TimeGAN": 3, "WaveGAN": 0,
    },
    "generic": {
        "GaussianProcess": 1, "AR": 1,
        "TimeVAE": 3, "RCGAN": 2,
        "TimeGAN": 2, "WaveGAN": 1,
    },
}

def score_domain(model: str, domain: str) -> int:
    table = DOMAIN_SCORES.get(domain, DOMAIN_SCORES["generic"])
    return table.get(model, 1)


# ── Factor 4: VRAM scoring ─────────────────────────────────────────────────

def score_vram(model: str, available_vram_gb: float) -> int:
    """
    Score a model based on whether it fits in available VRAM.
    Models that fit comfortably score 3, tight fit scores 1,
    won't fit scores 0.
    """
    required = MODEL_VRAM_GB[model]
    if required == 0.0:
        return 3  # CPU models always fit
    headroom = available_vram_gb - required
    if headroom >= 2.0:
        return 3   # fits with room to spare
    elif headroom >= 0.5:
        return 2   # fits but tight
    elif headroom >= 0.0:
        return 1   # barely fits
    else:
        return 0   # won't fit
