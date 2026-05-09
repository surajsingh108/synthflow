"""
SynConfig – the single source of truth for a synthflow pipeline run.

Every field has:
  - a type annotation
  - a validated default (where sensible)
  - a description (used by the LM parser prompt)

Validation rules are intentionally strict so that bad configs
fail immediately at construction, not silently later in the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# – Allowed literals –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

DOMAINS = Literal["industrial", "medical", "iot", "audio", "financial", "generic"]

SENSOR_TYPES = Literal[
    "accelerometer",
    "gyroscope",
    "temperature",
    "pressure",
    "humidity",
    "current",
    "voltage",
    "vibration",
    "microphone",
    "ecg",
    "eeg",
    "generic",
]

MISSING_PATTERNS = Literal["MCAR", "MAR", "MNAR", "auto"]

IMPUTATION_STRATEGIES = Literal[
    "auto",
    "forward_fill",
    "spline",
    "knn",
    "mice",
    "missforest",
    "hyperimpute",
]

BACKENDS = Literal["tsgm", "sdv", "gretel", "timesynth"]

MODELS = Literal[
    "TimeGAN",
    "TimeVAE",
    "RCGAN",
    "WaveGAN",
    "GaussianProcess",
    "AR",
    "GaussianCopula",
    "TabularVAE",
    "auto",
]

DEVICES = Literal["auto", "cuda", "cpu"]


# – SynConfig ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class SynConfig(BaseModel):
    """
    Full configuration for a synthflow pipeline run.

    Fields are grouped into four sections:
      1. Data description  – what the sensor data represents
      2. Missing data      – how to handle NaNs
      3. Synthesis         – which backend and model to use
      4. Hardware          – device and performance settings
    """

    # – 1. Data description ––––––––––––––––––––––––––––––––––––––––––––––––––

    domain: DOMAINS = Field(
        default="generic",
        description="High-level domain the sensor data comes from.",
    )

    sensor_type: SENSOR_TYPES = Field(
        default="generic",
        description="Type of physical sensor that produced the data.",
    )

    sampling_rate_hz: float = Field(
        default=100.0,
        gt=0,
        description="Sampling frequency in Hz. Must be > 0.",
    )

    description: str = Field(
        default="",
        description="Free-text description provided by the user (auto mode only).",
        max_length=1000,
    )

    # – 2. Missing data ––––––––––––––––––––––––––––––––––––––––––––––––––––––

    missing_pattern: MISSING_PATTERNS = Field(
        default="auto",
        description=(
            "Missing data mechanism. "
            "'auto' runs statistical detection per column."
        ),
    )

    imputation_strategy: IMPUTATION_STRATEGIES = Field(
        default="auto",
        description=(
            "Imputation method. "
            "'auto' selects based on detected missing_pattern."
        ),
    )

    imputation_overrides: dict[str, IMPUTATION_STRATEGIES] = Field(
        default_factory=dict,
        description=(
            "Per-column imputation overrides. "
            "Keys are column names, values are strategy names. "
            "Example: {'rpm': 'missforest', 'temp': 'spline'}"
        ),
    )

    # – 3. Synthesis –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    backend: BACKENDS = Field(
        default="tsgm",
        description="Synthesis backend to use.",
    )

    model: MODELS = Field(
        default="auto",
        description=(
            "Generative model to use. "
            "'auto' scores all models and picks the best for the config."
        ),
    )

    n_samples: int = Field(
        default=1000,
        gt=0,
        le=1_000_000,
        description="Number of synthetic samples to generate.",
    )

    augmentations: list[str] = Field(
        default_factory=list,
        description=(
            "List of augmentation names to apply. "
            "Valid values: jitter, window_warp, magnitude_warp, "
            "slice_and_shuffle, time_warp."
        ),
    )

    # – Data type –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    data_type: Literal["timeseries", "tabular", "auto"] = Field(
        default="auto",
        description=(
            "Whether the data is a time series or independent tabular rows. "
            "'auto' detects from sampling_rate_hz: "
            "  <= 1 Hz with no strong autocorrelation → tabular "
            "  > 1 Hz or explicit timestamp sequence → timeseries. "
            "Set 'tabular' explicitly for SECOM-style data."
        ),
    )

    seq_len: int | None = Field(
        default=None,
        ge=1,
        le=512,
        description=(
            "Sequence window length. "
            "Set to 1 for tabular data (each row is an independent sample). "
            "If None, inferred from sampling_rate_hz and data_type."
        ),
    )

    # – Bounds –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    auto_bounds: bool = Field(
        default=True,
        description=(
            "Automatically detect per-column physical bounds from real data. "
            "Columns where real_min >= 0 get a lower bound of 0. "
            "All columns are capped at their observed max. "
            "Overridden per-column by column_bounds."
        ),
    )

    column_bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description=(
            "Explicit per-column (lo, hi) bounds enforced after synthesis. "
            "Keys are original column names (no synthetic_ prefix). "
            "Example: {'sensor_01': (0.0, 5000.0)}"
        ),
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility.",
    )

    training_epochs: int = Field(
        default=200,
        gt=0,
        le=10_000,
        description=(
            "Training epochs for neural generative models "
            "(TabularVAE, TimeGAN, TimeVAE, RCGAN, WaveGAN). "
            "Ignored for statistical models (GaussianProcess, AR, GaussianCopula). "
            "Recommended: TabularVAE=200-500, TimeGAN=500-2000."
        ),
    )

    # – 4. Hardware ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    device: DEVICES = Field(
        default="auto",
        description=(
            "Compute device. "
            "'auto' uses CUDA if available, otherwise CPU."
        ),
    )

    batch_size: int = Field(
        default=128,
        gt=0,
        le=4096,
        description="Training batch size. Reduce if CUDA OOM.",
    )

    # – Validators –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    @field_validator("augmentations")
    @classmethod
    def validate_augmentations(cls, v: list[str]) -> list[str]:
        valid = {
            "jitter",
            "window_warp",
            "magnitude_warp",
            "slice_and_shuffle",
            "time_warp",
        }
        invalid = set(v) - valid
        if invalid:
            raise ValueError(
                f"Unknown augmentations: {invalid}. "
                f"Valid options: {valid}"
            )
        return v

    @field_validator("imputation_overrides")
    @classmethod
    def validate_imputation_overrides(
        cls, v: dict[str, str]
    ) -> dict[str, str]:
        valid = {
            "auto", "forward_fill", "spline",
            "knn", "mice", "missforest", "hyperimpute",
        }
        for col, strategy in v.items():
            if strategy not in valid:
                raise ValueError(
                    f"Column '{col}': unknown strategy '{strategy}'. "
                    f"Valid options: {valid}"
                )
        return v

    @model_validator(mode="after")
    def validate_model_backend_compatibility(self) -> SynConfig:
        """
        WaveGAN is only available on the tsgm backend.
        GaussianProcess and AR are only available on tsgm and timesynth.
        """
        if self.model == "WaveGAN" and self.backend != "tsgm":
            raise ValueError(
                f"WaveGAN is only available with backend='tsgm', "
                f"got backend='{self.backend}'"
            )
        stat_models = {"GaussianProcess", "AR"}
        if self.model in stat_models and self.backend not in ("tsgm", "timesynth"):
            raise ValueError(
                f"Model '{self.model}' is only available with "
                f"backend='tsgm' or backend='timesynth', "
                f"got backend='{self.backend}'"
            )
        return self

    @model_validator(mode="after")
    def validate_tabular_model(self) -> SynConfig:
        """
        GaussianCopula and TabularVAE require tabular or auto data_type.
        They cannot be used with explicit timeseries data_type.
        """
        tabular_models = {"GaussianCopula", "TabularVAE"}
        if self.model in tabular_models and self.data_type == "timeseries":
            raise ValueError(
                f"Model '{self.model}' is designed for tabular data. "
                f"Set data_type='tabular' or data_type='auto'."
            )
        return self

    # – Serialisation helpers –––––––––––––––––––––––––––––––––––––––––––––––

    def to_json(self, path: str | Path | None = None) -> str:
        """
        Serialise config to JSON string.
        If path is given, also write to file.
        """
        data = self.model_dump()
        json_str = json.dumps(data, indent=2)
        if path is not None:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, source: str | Path) -> SynConfig:
        """
        Load config from a JSON string or file path.
        """
        source = str(source)
        if source.strip().startswith("{"):
            # treat as raw JSON string
            data = json.loads(source)
        else:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        return cls(**data)

    def patch(self, **kwargs) -> SynConfig:
        """
        Return a new SynConfig with specific fields overridden.
        Does not mutate the original.

        Usage:
            new_cfg = cfg.patch(model="TimeVAE", n_samples=500)
        """
        current = self.model_dump()
        current.update(kwargs)
        return SynConfig(**current)

    def summary(self) -> str:
        """
        Return a short human-readable summary of the config.
        Used by the chat state machine to show the user what was parsed.
        """
        lines = [
            f"  domain          – {self.domain}",
            f"  sensor_type     – {self.sensor_type}",
            f"  sampling_rate   – {self.sampling_rate_hz} Hz",
            f"  missing_pattern – {self.missing_pattern}",
            f"  imputation      – {self.imputation_strategy}",
            f"  backend         – {self.backend}",
            f"  model           – {self.model}",
            f"  n_samples       – {self.n_samples}",
            f"  augmentations   – {self.augmentations or 'none'}",
            f"  data_type       – {self.data_type}",
            f"  seq_len         – {self.seq_len or 'auto'}",
            f"  auto_bounds     – {self.auto_bounds}",
            f"  device          – {self.device}",
            f"  batch_size      – {self.batch_size}",
            f"  training_epochs – {self.training_epochs}",
            f"  random_seed     – {self.random_seed}",
        ]
        return "\n".join(lines)
