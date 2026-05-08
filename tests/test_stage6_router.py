"""
Stage 6 tests – SynRouter (Model Selector)

Tests that SynRouter:
  - selects correct model for each domain/size combination
  - penalises models that exceed available VRAM
  - tunes batch size based on VRAM headroom
  - respects user-specified model (bypasses scoring)
  - returns correct fallback chain
  - next_fallback() traverses OOM chain correctly
  - SelectionResult has all required fields
  - scores all models and sorts them

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import pytest


# ── 1. Import ───────────────────────────────────────────────────────────────

class TestImport:
    def test_syn_router_importable(self):
        from synthflow.router import SynRouter
        assert SynRouter is not None

    def test_selection_result_importable(self):
        from synthflow.router import SelectionResult
        assert SelectionResult is not None

    def test_importable_from_backend_router(self):
        from synthflow.router.backend_router import SynRouter, SelectionResult
        assert SynRouter is not None

    def test_domain_rules_importable(self):
        from synthflow.router.domain_rules import ALL_MODELS, MODEL_VRAM_GB
        assert len(ALL_MODELS) > 0
        assert len(MODEL_VRAM_GB) > 0


# ── 2. Domain + size combinations ───────────────────────────────────────────

class TestDomainSizeSelection:
    def test_audio_domain_selects_wavegan(self):
        """Audio with high freq and large dataset – WaveGAN."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="audio",
            n_rows=50_000,
            sampling_rate_hz=44_100,
            n_signal_cols=2,
        )
        assert result.model == "WaveGAN"

    def test_tiny_dataset_selects_probabilistic(self):
        """Very small dataset – GaussianProcess or AR."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="generic",
            n_rows=100,
            sampling_rate_hz=10.0,
            n_signal_cols=1,
        )
        assert result.model in ("GaussianProcess", "AR")

    def test_large_industrial_selects_timegan_or_timevae(self):
        """Large industrial dataset – TimeGAN or TimeVAE."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="industrial",
            n_rows=20_000,
            sampling_rate_hz=500.0,
            n_signal_cols=6,
        )
        assert result.model in ("TimeGAN", "TimeVAE", "RCGAN")

    def test_iot_medium_dataset_selects_timevae(self):
        """Medium IoT dataset – TimeVAE (sweet spot)."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="iot",
            n_rows=2_000,
            sampling_rate_hz=100.0,
            n_signal_cols=3,
        )
        assert result.model in ("TimeVAE", "TimeGAN", "RCGAN")

    def test_financial_ar_model_preferred_for_low_freq(self):
        """Low freq financial data – AR scores high."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="financial",
            n_rows=200,
            sampling_rate_hz=1.0,
            n_signal_cols=1,
        )
        assert result.model in ("AR", "GaussianProcess")


# ── 3. VRAM constraints ─────────────────────────────────────────────────────

class TestVRAMConstraints:
    def test_no_vram_selects_cpu_model(self):
        """No GPU – only GaussianProcess or AR (0 VRAM required)."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=0.0)
        result = router.select(
            domain="generic",
            n_rows=5_000,
            sampling_rate_hz=100.0,
            n_signal_cols=3,
        )
        assert result.model in ("GaussianProcess", "AR")

    def test_2gb_vram_excludes_timegan_and_wavegan(self):
        """2GB VRAM – TimeGAN (5GB) and WaveGAN (7GB) heavily penalised."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=2.0)
        result = router.select(
            domain="generic",
            n_rows=5_000,
            sampling_rate_hz=100.0,
            n_signal_cols=3,
        )
        assert result.model not in ("TimeGAN", "WaveGAN")

    def test_8gb_vram_allows_all_models(self):
        """8GB VRAM – all models are viable, no VRAM penalty."""
        from synthflow.router import SynRouter
        from synthflow.router.domain_rules import MODEL_VRAM_GB
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="audio",
            n_rows=50_000,
            sampling_rate_hz=44_100,
            n_signal_cols=2,
        )
        # WaveGAN needs 7GB, 8GB available – should be selectable
        assert result.available_vram_gb == 8.0

    def test_vram_score_zero_for_model_exceeding_vram(self):
        """A model needing more VRAM than available scores 0 on VRAM factor."""
        from synthflow.router.domain_rules import score_vram
        # WaveGAN needs 7GB, only 2GB available
        score = score_vram("WaveGAN", 2.0)
        assert score == 0

    def test_cpu_model_always_scores_max_vram(self):
        """GP and AR need 0GB – always score 3 on VRAM factor."""
        from synthflow.router.domain_rules import score_vram
        assert score_vram("GaussianProcess", 0.0) == 3
        assert score_vram("AR", 0.0) == 3


# ── 4. Batch size tuning ────────────────────────────────────────────────────

class TestBatchSizeTuning:
    def test_default_batch_size_with_plenty_of_vram(self):
        """Lots of headroom – use default batch size (128)."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="generic",
            n_rows=5_000,
            sampling_rate_hz=100.0,
            n_signal_cols=3,
            preferred_model="TimeVAE",
        )
        assert result.batch_size == 128

    def test_reduced_batch_size_with_tight_vram(self):
        """Tight VRAM – batch size reduced."""
        from synthflow.router import SynRouter
        # TimeVAE needs 2.5GB, give it 3.0GB (0.5 headroom – tight)
        router = SynRouter(available_vram_gb=3.0)
        result = router.select(
            preferred_model="TimeVAE",
        )
        assert result.batch_size < 128

    def test_cpu_model_keeps_default_batch_size(self):
        """CPU models ignore VRAM – batch size unchanged."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=0.0)
        result = router.select(preferred_model="GaussianProcess")
        assert result.batch_size == 128

    def test_batch_size_at_least_16(self):
        """Batch size never goes below 16."""
        from synthflow.router.backend_router import _tune_batch_size
        size = _tune_batch_size("WaveGAN", available_vram_gb=0.5)
        assert size >= 16

    def test_batch_size_is_power_of_two_or_reasonable(self):
        """Batch sizes should be 16, 32, 64, or 128."""
        from synthflow.router.backend_router import _tune_batch_size
        for vram in [0.0, 1.0, 2.0, 4.0, 8.0]:
            size = _tune_batch_size("TimeGAN", vram)
            assert size in (16, 32, 64, 128)


# ── 5. User-specified model ─────────────────────────────────────────────────

class TestUserSpecifiedModel:
    def test_preferred_model_bypasses_scoring(self):
        """User says TimeVAE – get TimeVAE regardless of other factors."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(
            domain="audio",         # would normally pick WaveGAN
            n_rows=50_000,
            sampling_rate_hz=44_100,
            preferred_model="TimeVAE",
        )
        assert result.model == "TimeVAE"

    def test_preferred_model_still_tunes_batch_size(self):
        """Even with forced model, batch size is VRAM-tuned."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=3.0)  # tight for TimeGAN
        result = router.select(preferred_model="TimeGAN")
        # TimeGAN needs 5GB, only 3GB available – batch should reduce
        assert result.batch_size <= 64

    def test_invalid_preferred_model_raises(self):
        """Unknown model name raises SynRouterError."""
        from synthflow.router import SynRouter
        from synthflow.exceptions import SynRouterError
        router = SynRouter(available_vram_gb=8.0)
        with pytest.raises(SynRouterError):
            router.select(preferred_model="GPT5")

    def test_preferred_model_reason_mentions_user(self):
        """Selection reason should mention user-specified."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(preferred_model="TimeVAE")
        assert "user" in result.reason.lower()


# ── 6. Fallback chain ───────────────────────────────────────────────────────

class TestFallbackChain:
    def test_timegan_has_fallback(self):
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(preferred_model="TimeGAN")
        assert result.fallback != "TimeGAN"
        assert len(result.fallback_chain) > 0

    def test_wavegan_fallback_chain_length(self):
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(preferred_model="WaveGAN")
        assert len(result.fallback_chain) >= 2

    def test_gaussian_process_has_no_fallback(self):
        """GaussianProcess is the end of the chain."""
        from synthflow.router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        result = router.select(preferred_model="GaussianProcess")
        assert result.fallback_chain == []

    def test_next_fallback_returns_correct_model(self):
        from synthflow.router.backend_router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        fallback = router.next_fallback("TimeGAN")
        assert fallback == "TimeVAE"

    def test_next_fallback_from_timevae(self):
        from synthflow.router.backend_router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        fallback = router.next_fallback("TimeVAE")
        assert fallback == "GaussianProcess"

    def test_next_fallback_returns_none_at_end_of_chain(self):
        from synthflow.router.backend_router import SynRouter
        router = SynRouter(available_vram_gb=8.0)
        fallback = router.next_fallback("GaussianProcess")
        assert fallback is None


# ── 7. SelectionResult fields ───────────────────────────────────────────────

class TestSelectionResult:
    def test_result_has_model(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        assert isinstance(result.model, str)
        assert len(result.model) > 0

    def test_result_has_fallback(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        assert isinstance(result.fallback, str)

    def test_result_has_batch_size(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        assert isinstance(result.batch_size, int)
        assert result.batch_size > 0

    def test_result_has_vram(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=6.0).select()
        assert result.available_vram_gb == 6.0

    def test_result_has_scores_list(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        assert isinstance(result.scores, list)
        assert len(result.scores) == 6  # one per model

    def test_scores_sorted_descending(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        totals = [s.total for s in result.scores]
        assert totals == sorted(totals, reverse=True)

    def test_result_has_reason(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_summary_returns_string(self):
        from synthflow.router import SynRouter
        result = SynRouter(available_vram_gb=8.0).select()
        assert isinstance(result.summary(), str)
        assert len(result.summary()) > 0


# ── 8. Scoring functions ────────────────────────────────────────────────────

class TestScoringFunctions:
    def test_score_size_tiny_favours_probabilistic(self):
        from synthflow.router.domain_rules import score_size
        assert score_size("GaussianProcess", 100) == 3
        assert score_size("TimeGAN", 100) == 0

    def test_score_size_large_favours_gans(self):
        from synthflow.router.domain_rules import score_size
        assert score_size("TimeGAN", 100_000) == 3
        assert score_size("GaussianProcess", 100_000) == 0

    def test_score_complexity_high_freq_favours_wavegan(self):
        from synthflow.router.domain_rules import score_complexity
        assert score_complexity("WaveGAN", 44_100, 2) == 3
        assert score_complexity("GaussianProcess", 44_100, 2) == 0

    def test_score_complexity_low_freq_favours_ar(self):
        from synthflow.router.domain_rules import score_complexity
        assert score_complexity("AR", 1.0, 1) == 3
        assert score_complexity("WaveGAN", 1.0, 1) == 0

    def test_score_domain_audio_favours_wavegan(self):
        from synthflow.router.domain_rules import score_domain
        assert score_domain("WaveGAN", "audio") == 3
        assert score_domain("AR", "audio") == 0

    def test_score_domain_industrial_favours_timevae_or_timegan(self):
        from synthflow.router.domain_rules import score_domain
        assert score_domain("TimeVAE", "industrial") == 3
        assert score_domain("TimeGAN", "industrial") == 3

    def test_all_scores_in_range_0_to_3(self):
        from synthflow.router.domain_rules import (
            score_size, score_complexity, score_domain, score_vram, ALL_MODELS
        )
        for model in ALL_MODELS:
            assert 0 <= score_size(model, 1000) <= 3
            assert 0 <= score_complexity(model, 100.0, 3) <= 3
            assert 0 <= score_domain(model, "generic") <= 3
            assert 0 <= score_vram(model, 8.0) <= 3
