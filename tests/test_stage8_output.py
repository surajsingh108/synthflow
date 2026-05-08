import json

import numpy as np
import pandas as pd
import pytest


def make_real_df(n=200):
    t = np.linspace(0, 4 * np.pi, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="10ms"),
        "accel_x": np.sin(t) + 0.05 * np.random.randn(n),
        "accel_y": np.cos(t) + 0.05 * np.random.randn(n),
        "temperature": np.linspace(20, 80, n) + np.random.randn(n),
    })


def make_synth_df(n=200, signal_cols=None):
    if signal_cols is None:
        signal_cols = ["accel_x", "accel_y", "temperature"]
    t = np.linspace(0, 4 * np.pi, n)
    data = {"timestamp": pd.date_range("2024-01-01", periods=n, freq="10ms")}
    for col in signal_cols:
        data[f"synthetic_{col}"] = np.sin(t) + 0.1 * np.random.randn(n)
    return pd.DataFrame(data)


SIGNAL_COLS = ["accel_x", "accel_y", "temperature"]
STUB_CONFIG = {"domain": "industrial", "model": "GaussianProcess"}
STUB_IMP = {
    "total_missing_before": 10,
    "missing_pct_before": 2.5,
    "missing_pct_after": 0.0,
    "pattern_detected": "MCAR",
    "strategy_used": "forward_fill",
    "columns_affected": {},
}
STUB_QUALITY = {
    "distribution_similarity": {"accel_x": 0.9, "overall": 0.9},
    "autocorrelation_score": 0.85,
    "pca_overlap_score": 0.88,
    "train_synth_test_real": 0.80,
}
STUB_DESC = {
    "synthetic_accel_x": {
        "original_col": "accel_x",
        "dtype": "float32",
        "range": [-1.0, 1.0],
        "mean": 0.0,
        "std": 0.5,
        "unit": "unknown",
    }
}


class TestImport:
    def test_syn_result_importable(self):
        from synthflow.output import SynResult

        assert SynResult is not None

    def test_quality_module_importable(self):
        from synthflow.output.quality import compute_all_metrics

        assert compute_all_metrics is not None

    def test_report_importable(self):
        from synthflow.output.report import SynReport

        assert SynReport is not None

    def test_writer_importable(self):
        from synthflow.output.writer import SynWriter

        assert SynWriter is not None

    def test_console_importable(self):
        from synthflow.output.console import print_summary

        assert print_summary is not None


class TestQualityMetrics:
    def test_distribution_similarity_identical(self):
        from synthflow.output.quality import distribution_similarity

        s = pd.Series(np.random.randn(200))
        assert distribution_similarity(s, s) == pytest.approx(1.0, abs=0.05)

    def test_distribution_similarity_different(self):
        from synthflow.output.quality import distribution_similarity

        real = pd.Series(np.random.randn(200))
        synth = pd.Series(np.random.randn(200) + 10)
        assert distribution_similarity(real, synth) < 0.5

    def test_distribution_similarity_in_range(self):
        from synthflow.output.quality import distribution_similarity

        real = pd.Series(np.random.randn(200))
        synth = pd.Series(np.random.randn(200))
        assert 0.0 <= distribution_similarity(real, synth) <= 1.0

    def test_distribution_similarity_all_has_overall(self):
        from synthflow.output.quality import distribution_similarity_all

        scores = distribution_similarity_all(make_real_df(), make_synth_df(), SIGNAL_COLS)
        assert "overall" in scores

    def test_distribution_similarity_all_per_column(self):
        from synthflow.output.quality import distribution_similarity_all

        scores = distribution_similarity_all(make_real_df(), make_synth_df(), SIGNAL_COLS)
        assert "accel_x" in scores and "accel_y" in scores

    def test_autocorrelation_score_in_range(self):
        from synthflow.output.quality import autocorrelation_score

        assert (
            0.0
            <= autocorrelation_score(make_real_df(), make_synth_df(), SIGNAL_COLS)
            <= 1.0
        )

    def test_pca_overlap_score_in_range(self):
        from synthflow.output.quality import pca_overlap_score

        assert (
            0.0 <= pca_overlap_score(make_real_df(), make_synth_df(), SIGNAL_COLS) <= 1.0
        )

    def test_pca_overlap_identical_data_high_score(self):
        from synthflow.output.quality import pca_overlap_score

        real = make_real_df()
        synth = real.rename(columns={col: f"synthetic_{col}" for col in SIGNAL_COLS})
        assert pca_overlap_score(real, synth, SIGNAL_COLS) > 0.7

    def test_tstr_score_in_range(self):
        from synthflow.output.quality import tstr_score

        assert 0.0 <= tstr_score(make_real_df(), make_synth_df(), SIGNAL_COLS) <= 1.0

    def test_compute_all_metrics_has_four_keys(self):
        from synthflow.output.quality import compute_all_metrics

        metrics = compute_all_metrics(make_real_df(), make_synth_df(), SIGNAL_COLS)
        for key in [
            "distribution_similarity",
            "autocorrelation_score",
            "pca_overlap_score",
            "train_synth_test_real",
        ]:
            assert key in metrics

    def test_compute_all_metrics_values_in_range(self):
        from synthflow.output.quality import compute_all_metrics

        metrics = compute_all_metrics(make_real_df(), make_synth_df(), SIGNAL_COLS)
        assert 0.0 <= metrics["autocorrelation_score"] <= 1.0
        assert 0.0 <= metrics["pca_overlap_score"] <= 1.0
        assert 0.0 <= metrics["train_synth_test_real"] <= 1.0


class TestSynReport:
    def _make(self):
        from synthflow.output.report import SynReport

        return SynReport(
            "id", "ts", 100, STUB_CONFIG, STUB_IMP, STUB_QUALITY, STUB_DESC
        )

    def test_as_dict_has_required_keys(self):
        d = self._make().as_dict()
        for k in [
            "run_id",
            "generated_at",
            "n_samples_generated",
            "config",
            "imputation_report",
            "quality_metrics",
            "column_descriptions",
        ]:
            assert k in d

    def test_to_json_valid(self):
        assert json.loads(self._make().to_json())["run_id"] == "id"

    def test_to_json_writes_file(self, tmp_path):
        path = tmp_path / "r.json"
        self._make().to_json(path=path)
        assert path.exists()
        assert json.loads(path.read_text())["n_samples_generated"] == 100

    def test_make_run_id_format(self):
        from synthflow.output.report import SynReport

        assert SynReport.make_run_id().startswith("synthflow_run_")

    def test_build_column_descriptions(self):
        from synthflow.output.report import SynReport

        desc = SynReport.build_column_descriptions(make_synth_df(), SIGNAL_COLS)
        assert "synthetic_accel_x" in desc
        col = desc["synthetic_accel_x"]
        for k in ["original_col", "dtype", "range", "mean", "std"]:
            assert k in col


class TestSynWriter:
    def test_write_creates_run_folder(self, tmp_path):
        from synthflow.output.writer import SynWriter

        run_dir = SynWriter().write(
            make_synth_df(50), '{"run_id":"t"}', "synthflow_run_test", "dataframe", tmp_path
        )
        assert run_dir.exists() and run_dir.is_dir()

    def test_write_creates_data_file(self, tmp_path):
        from synthflow.output.writer import SynWriter

        run_dir = SynWriter().write(
            make_synth_df(50), "{}", "synthflow_run_test", "dataframe", tmp_path
        )
        assert len(list(run_dir.glob("*.csv"))) > 0

    def test_write_creates_report_json(self, tmp_path):
        from synthflow.output.writer import SynWriter

        run_dir = SynWriter().write(
            make_synth_df(50), "{}", "synthflow_run_test", "sensor.csv", tmp_path
        )
        assert (run_dir / "run_report.json").exists()

    def test_csv_source_writes_csv(self, tmp_path):
        from synthflow.output.writer import SynWriter

        run_dir = SynWriter().write(
            make_synth_df(50), "{}", "synthflow_run_test", "readings.csv", tmp_path
        )
        assert (run_dir / "readings_synthetic.csv").exists()

    def test_json_source_writes_json(self, tmp_path):
        from synthflow.output.writer import SynWriter

        run_dir = SynWriter().write(
            make_synth_df(50), "{}", "synthflow_run_test", "sensor_data.json", tmp_path
        )
        assert (run_dir / "sensor_data_synthetic.json").exists()

    def test_written_csv_is_readable(self, tmp_path):
        from synthflow.output.writer import SynWriter

        run_dir = SynWriter().write(
            make_synth_df(50), "{}", "synthflow_run_test", "sensor.csv", tmp_path
        )
        loaded = pd.read_csv(run_dir / "sensor_synthetic.csv")
        assert len(loaded) == 50


class TestSynResult:
    def _make(self):
        from synthflow.output import SynResult
        from synthflow.output.report import SynReport

        return SynResult(
            data=make_synth_df(100),
            config=STUB_CONFIG,
            imputation_report=STUB_IMP,
            quality_metrics=STUB_QUALITY,
            column_descriptions=STUB_DESC,
            run_id=SynReport.make_run_id(),
            signal_cols=SIGNAL_COLS,
            timestamp_col="timestamp",
            source="sensor.csv",
            model="GaussianProcess",
            backend="tsgm",
            sampling_rate_hz=100.0,
        )

    def test_data_is_dataframe(self):
        assert isinstance(self._make().data, pd.DataFrame)

    def test_signal_cols_stored(self):
        assert self._make().signal_cols == SIGNAL_COLS

    def test_run_id_stored(self):
        assert self._make().run_id.startswith("synthflow_run_")

    def test_as_dict_has_required_keys(self):
        d = self._make().as_dict()
        for k in [
            "run_id",
            "model",
            "backend",
            "n_samples",
            "config",
            "imputation_report",
            "quality_metrics",
            "column_descriptions",
        ]:
            assert k in d

    def test_as_dict_n_samples_correct(self):
        assert self._make().as_dict()["n_samples"] == 100

    def test_summary_returns_string(self):
        s = self._make().summary()
        assert isinstance(s, str) and len(s) > 0

    def test_summary_contains_model_name(self):
        assert "GaussianProcess" in self._make().summary()

    def test_save_creates_run_folder(self, tmp_path):
        assert self._make().save(output_dir=tmp_path).exists()

    def test_save_creates_report_json(self, tmp_path):
        run_dir = self._make().save(output_dir=tmp_path)
        assert (run_dir / "run_report.json").exists()

    def test_save_creates_data_file(self, tmp_path):
        run_dir = self._make().save(output_dir=tmp_path)
        assert len(list(run_dir.glob("*.csv"))) > 0

    def test_save_report_is_valid_json(self, tmp_path):
        run_dir = self._make().save(output_dir=tmp_path)
        parsed = json.loads((run_dir / "run_report.json").read_text())
        for k in ["config", "quality_metrics", "imputation_report"]:
            assert k in parsed
