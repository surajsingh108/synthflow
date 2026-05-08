"""
SynRouter – scores all models and selects the best one
given the dataset characteristics and available hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthflow.exceptions import SynRouterError
from synthflow.router.domain_rules import (
    ALL_MODELS,
    MODEL_VRAM_GB,
    OOM_FALLBACK,
    score_size,
    score_complexity,
    score_domain,
    score_vram,
)


# ── VRAM detection ──────────────────────────────────────────────────────────

# Default VRAM assumed when detection fails (conservative)
_DEFAULT_VRAM_GB = 4.0

def _detect_available_vram_gb() -> float:
    """
    Detect available GPU VRAM in GB.
    Returns 0.0 if no CUDA device found (will favour CPU models).
    Returns _DEFAULT_VRAM_GB if detection fails unexpectedly.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, _ = torch.cuda.mem_get_info()
        return round(free_bytes / (1024 ** 3), 2)
    except Exception:
        return _DEFAULT_VRAM_GB


# ── Batch size tuning ───────────────────────────────────────────────────────

def _tune_batch_size(
    model: str,
    available_vram_gb: float,
    default_batch_size: int = 128,
) -> int:
    """
    Return a safe batch size for the given model and available VRAM.
    Reduces batch size when VRAM headroom is tight.
    """
    required = MODEL_VRAM_GB[model]
    if required == 0.0:
        return default_batch_size  # CPU models – batch size doesn't affect VRAM

    headroom = available_vram_gb - required
    if headroom >= 2.0:
        return default_batch_size        # 128
    elif headroom >= 1.0:
        return max(64, default_batch_size // 2)   # 64
    elif headroom >= 0.0:
        return max(32, default_batch_size // 4)   # 32
    else:
        return max(16, default_batch_size // 8)   # 16 (shouldn't be selected)


# ── Result type ────────────────────────────────────────────────────────────

@dataclass
class ModelScore:
    """Score breakdown for a single model."""
    model: str
    total: int
    size_score: int
    complexity_score: int
    domain_score: int
    vram_score: int


@dataclass
class SelectionResult:
    """
    Result returned by SynRouter.select().

    Attributes:
        model           : recommended model name
        fallback        : fallback model if primary OOMs
        fallback_chain  : full ordered fallback list
        batch_size      : tuned batch size for the selected model
        available_vram_gb: detected available VRAM (0.0 = CPU only)
        scores          : full scoring breakdown for all models
        reason          : human-readable explanation of the selection
    """
    model: str
    fallback: str
    fallback_chain: list[str]
    batch_size: int
    available_vram_gb: float
    scores: list[ModelScore] = field(default_factory=list)
    reason: str = ""

    def summary(self) -> str:
        lines = [
            f"  selected model  – {self.model}",
            f"  fallback        – {self.fallback}",
            f"  batch_size      – {self.batch_size}",
            f"  available_vram  – {self.available_vram_gb:.1f} GB",
            f"  reason          – {self.reason}",
        ]
        return "\n".join(lines)


# ── Router ──────────────────────────────────────────────────────────────────

class SynRouter:
    """
    Selects the best generative model for a given dataset and config.

    Usage:
        router = SynRouter()
        result = router.select(
            domain="industrial",
            n_rows=5000,
            sampling_rate_hz=500,
            n_signal_cols=6,
        )
        print(result.model)        # "TimeGAN"
        print(result.batch_size)   # 64
    """

    def __init__(self, available_vram_gb: float | None = None):
        """
        Args:
            available_vram_gb: override VRAM detection (useful for testing).
                               If None, auto-detects from torch.
        """
        if available_vram_gb is not None:
            self._vram = available_vram_gb
        else:
            self._vram = _detect_available_vram_gb()

    # ── public API ──────────────────────────────────────────────────────────

    def select(
        self,
        domain: str = "generic",
        n_rows: int = 1000,
        sampling_rate_hz: float = 100.0,
        n_signal_cols: int = 3,
        preferred_model: str = "auto",
        default_batch_size: int = 128,
    ) -> SelectionResult:
        """
        Score all models and return the best selection.

        Args:
            domain            : data domain (matches SynConfig.domain)
            n_rows            : number of rows in the dataset
            sampling_rate_hz  : sampling frequency in Hz
            n_signal_cols     : number of signal columns
            preferred_model   : "auto" or explicit model name to force
            default_batch_size: starting batch size before VRAM tuning

        Returns:
            SelectionResult with model, fallback, batch_size, scores.

        Raises:
            SynRouterError if preferred_model is not a known model.
        """
        # ── user specified a model explicitly ────────────────────────────────
        if preferred_model != "auto":
            if preferred_model not in ALL_MODELS:
                raise SynRouterError(
                    f"Unknown model: '{preferred_model}'",
                    detail=f"Valid models: {ALL_MODELS}",
                )
            fallback_chain = OOM_FALLBACK.get(preferred_model, [])
            fallback = fallback_chain[0] if fallback_chain else preferred_model
            batch_size = _tune_batch_size(
                preferred_model, self._vram, default_batch_size
            )
            return SelectionResult(
                model=preferred_model,
                fallback=fallback,
                fallback_chain=fallback_chain,
                batch_size=batch_size,
                available_vram_gb=self._vram,
                reason=f"user-specified model '{preferred_model}'",
            )

        # ── score all models ────────────────────────────────────────────────
        scores: list[ModelScore] = []
        cpu_only_models = ["GaussianProcess", "AR"]

        for model in ALL_MODELS:
            s_size = score_size(model, n_rows)
            s_comp = score_complexity(model, sampling_rate_hz, n_signal_cols)
            s_dom  = score_domain(model, domain)
            s_vram = score_vram(model, self._vram)

            # if no VRAM available, exclude GPU models entirely
            if self._vram == 0.0 and model not in cpu_only_models:
                total = -1  # will sort to bottom
            else:
                total  = s_size + s_comp + s_dom + s_vram

            scores.append(ModelScore(
                model=model,
                total=total,
                size_score=s_size,
                complexity_score=s_comp,
                domain_score=s_dom,
                vram_score=s_vram,
            ))

        # sort descending by total score
        scores.sort(key=lambda s: s.total, reverse=True)
        best = scores[0]

        # ── build fallback chain ────────────────────────────────────────────
        fallback_chain = OOM_FALLBACK.get(best.model, [])
        fallback = fallback_chain[0] if fallback_chain else best.model

        # ── tune batch size ─────────────────────────────────────────────────
        batch_size = _tune_batch_size(
            best.model, self._vram, default_batch_size
        )

        reason = (
            f"scored {best.total}/12 "
            f"(size={best.size_score}, complexity={best.complexity_score}, "
            f"domain={best.domain_score}, vram={best.vram_score})"
        )

        return SelectionResult(
            model=best.model,
            fallback=fallback,
            fallback_chain=fallback_chain,
            batch_size=batch_size,
            available_vram_gb=self._vram,
            scores=scores,
            reason=reason,
        )

    def next_fallback(self, failed_model: str) -> str | None:
        """
        Given a model that just OOMed, return the next fallback.
        Returns None if no fallback is available.

        Usage in backend:
            model = result.model
            while True:
                try:
                    run(model)
                    break
                except OOM:
                    model = router.next_fallback(model)
                    if model is None:
                        raise SynRouterError("All models OOMed")
        """
        chain = OOM_FALLBACK.get(failed_model, [])
        return chain[0] if chain else None
