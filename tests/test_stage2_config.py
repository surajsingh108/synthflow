"""
Stage 2 tests – SynConfig

Tests that SynConfig:
  - accepts valid inputs and applies correct defaults
  - rejects invalid inputs with clear errors
  - serialises and deserialises correctly (JSON roundtrip)
  - patch() returns new config without mutating original
  - summary() returns a non-empty string
  - cross-field validation (model/backend compatibility) works

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError


# – helpers –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def make_config(**kwargs):
    from synthflow.parser import SynConfig
    return SynConfig(**kwargs)


# – 1. Import ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestImport:
    def test_synconfig_importable_from_parser(self):
        from synthflow.parser import SynConfig
        assert SynConfig is not None

    def test_synconfig_importable_from_schema(self):
        from synthflow.parser.schema import SynConfig
        assert SynConfig is not None


# – 2. Defaults –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestDefaults:
    def test_default_domain(self):
        cfg = make_config()
        assert cfg.domain == "generic"

    def test_default_sensor_type(self):
        cfg = make_config()
        assert cfg.sensor_type == "generic"

    def test_default_sampling_rate(self):
        cfg = make_config()
        assert cfg.sampling_rate_hz == 100.0

    def test_default_missing_pattern(self):
        cfg = make_config()
        assert cfg.missing_pattern == "auto"

    def test_default_imputation_strategy(self):
        cfg = make_config()
        assert cfg.imputation_strategy == "auto"

    def test_default_backend(self):
        cfg = make_config()
        assert cfg.backend == "tsgm"

    def test_default_model(self):
        cfg = make_config()
        assert cfg.model == "auto"

    def test_default_n_samples(self):
        cfg = make_config()
        assert cfg.n_samples == 1000

    def test_default_augmentations_empty_list(self):
        cfg = make_config()
        assert cfg.augmentations == []

    def test_default_random_seed(self):
        cfg = make_config()
        assert cfg.random_seed == 42

    def test_default_device(self):
        cfg = make_config()
        assert cfg.device == "auto"

    def test_default_batch_size(self):
        cfg = make_config()
        assert cfg.batch_size == 128

    def test_default_imputation_overrides_empty(self):
        cfg = make_config()
        assert cfg.imputation_overrides == {}

    def test_default_description_empty(self):
        cfg = make_config()
        assert cfg.description == ""


# – 3. Valid inputs ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestValidInputs:
    def test_valid_domain(self):
        for domain in ["industrial", "medical", "iot", "audio", "financial", "generic"]:
            cfg = make_config(domain=domain)
            assert cfg.domain == domain

    def test_valid_sensor_types(self):
        for st in ["accelerometer", "gyroscope", "temperature", "ecg", "eeg", "generic"]:
            cfg = make_config(sensor_type=st)
            assert cfg.sensor_type == st

    def test_valid_sampling_rate(self):
        cfg = make_config(sampling_rate_hz=500.0)
        assert cfg.sampling_rate_hz == 500.0

    def test_valid_missing_patterns(self):
        for p in ["MCAR", "MAR", "MNAR", "auto"]:
            cfg = make_config(missing_pattern=p)
            assert cfg.missing_pattern == p

    def test_valid_imputation_strategies(self):
        for s in ["auto", "forward_fill", "spline", "knn", "mice", "missforest", "hyperimpute"]:
            cfg = make_config(imputation_strategy=s)
            assert cfg.imputation_strategy == s

    def test_valid_backends(self):
        for b in ["tsgm", "sdv", "gretel", "timesynth"]:
            cfg = make_config(backend=b)
            assert cfg.backend == b

    def test_valid_models(self):
        for m in ["TimeGAN", "TimeVAE", "RCGAN", "WaveGAN", "GaussianProcess", "AR", "auto"]:
            cfg = make_config(model=m)
            assert cfg.model == m

    def test_valid_n_samples(self):
        cfg = make_config(n_samples=5000)
        assert cfg.n_samples == 5000

    def test_valid_augmentations(self):
        cfg = make_config(augmentations=["jitter", "window_warp"])
        assert "jitter" in cfg.augmentations
        assert "window_warp" in cfg.augmentations

    def test_valid_device(self):
        for d in ["auto", "cuda", "cpu"]:
            cfg = make_config(device=d)
            assert cfg.device == d

    def test_valid_imputation_overrides(self):
        cfg = make_config(imputation_overrides={"rpm": "missforest", "temp": "spline"})
        assert cfg.imputation_overrides["rpm"] == "missforest"
        assert cfg.imputation_overrides["temp"] == "spline"

    def test_valid_description(self):
        cfg = make_config(description="accelerometer from wind turbine")
        assert cfg.description == "accelerometer from wind turbine"


# – 4. Invalid inputs ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestInvalidInputs:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValidationError):
            make_config(domain="space")

    def test_invalid_sensor_type_raises(self):
        with pytest.raises(ValidationError):
            make_config(sensor_type="sonar")

    def test_zero_sampling_rate_raises(self):
        with pytest.raises(ValidationError):
            make_config(sampling_rate_hz=0)

    def test_negative_sampling_rate_raises(self):
        with pytest.raises(ValidationError):
            make_config(sampling_rate_hz=-100)

    def test_invalid_missing_pattern_raises(self):
        with pytest.raises(ValidationError):
            make_config(missing_pattern="RANDOM")

    def test_invalid_imputation_strategy_raises(self):
        with pytest.raises(ValidationError):
            make_config(imputation_strategy="magic")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError):
            make_config(backend="mybackend")

    def test_invalid_model_raises(self):
        with pytest.raises(ValidationError):
            make_config(model="GPT5")

    def test_zero_n_samples_raises(self):
        with pytest.raises(ValidationError):
            make_config(n_samples=0)

    def test_n_samples_too_large_raises(self):
        with pytest.raises(ValidationError):
            make_config(n_samples=2_000_000)

    def test_invalid_augmentation_raises(self):
        with pytest.raises(ValidationError):
            make_config(augmentations=["invalid_aug"])

    def test_invalid_device_raises(self):
        with pytest.raises(ValidationError):
            make_config(device="tpu")

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValidationError):
            make_config(batch_size=0)

    def test_batch_size_too_large_raises(self):
        with pytest.raises(ValidationError):
            make_config(batch_size=9999)

    def test_invalid_imputation_override_value_raises(self):
        with pytest.raises(ValidationError):
            make_config(imputation_overrides={"col": "magic_method"})

    def test_description_too_long_raises(self):
        with pytest.raises(ValidationError):
            make_config(description="x" * 1001)


# – 5. Cross-field validation ––––––––––––––––––––––––––––––––––––––––––––––––

class TestCrossFieldValidation:
    def test_wavegan_requires_tsgm_backend(self):
        with pytest.raises(ValidationError):
            make_config(model="WaveGAN", backend="sdv")

    def test_wavegan_with_tsgm_passes(self):
        cfg = make_config(model="WaveGAN", backend="tsgm")
        assert cfg.model == "WaveGAN"

    def test_gaussian_process_with_sdv_raises(self):
        with pytest.raises(ValidationError):
            make_config(model="GaussianProcess", backend="sdv")

    def test_gaussian_process_with_tsgm_passes(self):
        cfg = make_config(model="GaussianProcess", backend="tsgm")
        assert cfg.model == "GaussianProcess"

    def test_gaussian_process_with_timesynth_passes(self):
        cfg = make_config(model="GaussianProcess", backend="timesynth")
        assert cfg.model == "GaussianProcess"

    def test_ar_with_gretel_raises(self):
        with pytest.raises(ValidationError):
            make_config(model="AR", backend="gretel")


# – 6. Serialisation ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestSerialisation:
    def test_to_json_returns_string(self):
        cfg = make_config(domain="industrial", sensor_type="accelerometer")
        result = cfg.to_json()
        assert isinstance(result, str)

    def test_to_json_is_valid_json(self):
        cfg = make_config()
        data = json.loads(cfg.to_json())
        assert "domain" in data
        assert "model" in data

    def test_to_json_writes_file(self):
        cfg = make_config(domain="medical")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        cfg.to_json(path=path)
        content = Path(path).read_text()
        data = json.loads(content)
        assert data["domain"] == "medical"

    def test_from_json_string_roundtrip(self):
        cfg = make_config(
            domain="iot",
            sensor_type="temperature",
            sampling_rate_hz=10.0,
            n_samples=500,
        )
        from synthflow.parser import SynConfig
        restored = SynConfig.from_json(cfg.to_json())
        assert restored.domain == "iot"
        assert restored.sensor_type == "temperature"
        assert restored.sampling_rate_hz == 10.0
        assert restored.n_samples == 500

    def test_from_json_file_roundtrip(self):
        cfg = make_config(domain="audio", model="WaveGAN")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        cfg.to_json(path=path)
        from synthflow.parser import SynConfig
        restored = SynConfig.from_json(path)
        assert restored.domain == "audio"
        assert restored.model == "WaveGAN"

    def test_roundtrip_preserves_all_fields(self):
        cfg = make_config(
            domain="industrial",
            sensor_type="vibration",
            sampling_rate_hz=500.0,
            missing_pattern="MCAR",
            imputation_strategy="spline",
            backend="tsgm",
            model="TimeVAE",
            n_samples=2000,
            augmentations=["jitter", "window_warp"],
            random_seed=99,
            device="cpu",
            batch_size=64,
        )
        from synthflow.parser import SynConfig
        restored = SynConfig.from_json(cfg.to_json())
        assert restored == cfg


# – 7. patch() ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestPatch:
    def test_patch_returns_new_instance(self):
        cfg = make_config()
        new_cfg = cfg.patch(model="TimeVAE")
        assert new_cfg is not cfg

    def test_patch_does_not_mutate_original(self):
        cfg = make_config(model="TimeGAN")
        cfg.patch(model="TimeVAE")
        assert cfg.model == "TimeGAN"

    def test_patch_applies_single_field(self):
        cfg = make_config(model="TimeGAN")
        new_cfg = cfg.patch(model="TimeVAE")
        assert new_cfg.model == "TimeVAE"

    def test_patch_preserves_other_fields(self):
        cfg = make_config(domain="industrial", model="TimeGAN", n_samples=500)
        new_cfg = cfg.patch(model="TimeVAE")
        assert new_cfg.domain == "industrial"
        assert new_cfg.n_samples == 500

    def test_patch_multiple_fields(self):
        cfg = make_config()
        new_cfg = cfg.patch(n_samples=2000, device="cpu", batch_size=32)
        assert new_cfg.n_samples == 2000
        assert new_cfg.device == "cpu"
        assert new_cfg.batch_size == 32

    def test_patch_validates_new_values(self):
        cfg = make_config()
        with pytest.raises(ValidationError):
            cfg.patch(domain="invalid_domain")


# – 8. summary() ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class TestSummary:
    def test_summary_returns_string(self):
        cfg = make_config()
        assert isinstance(cfg.summary(), str)

    def test_summary_non_empty(self):
        cfg = make_config()
        assert len(cfg.summary()) > 0

    def test_summary_contains_domain(self):
        cfg = make_config(domain="industrial")
        assert "industrial" in cfg.summary()

    def test_summary_contains_model(self):
        cfg = make_config(model="TimeVAE")
        assert "TimeVAE" in cfg.summary()

    def test_summary_contains_sampling_rate(self):
        cfg = make_config(sampling_rate_hz=500.0)
        assert "500" in cfg.summary()
