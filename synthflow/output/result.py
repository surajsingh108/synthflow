from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from synthflow.output.console import print_summary
from synthflow.output.report import SynReport
from synthflow.output.writer import SynWriter


@dataclass
class SynResult:
    data: pd.DataFrame
    config: dict
    imputation_report: dict
    quality_metrics: dict
    column_descriptions: dict
    run_id: str
    signal_cols: list
    timestamp_col: str | None = None
    source: str = "dataframe"
    model: str = "GaussianProcess"
    backend: str = "tsgm"
    sampling_rate_hz: float = 100.0

    def summary(self) -> str:
        return print_summary(
            run_id=self.run_id,
            model=self.model,
            backend=self.backend,
            n_samples=len(self.data),
            sampling_rate_hz=self.sampling_rate_hz,
            imputation_report=self.imputation_report,
            quality_metrics=self.quality_metrics,
            signal_cols=self.signal_cols,
            synth_df=self.data,
        )

    def save(self, output_dir="./outputs"):
        report = SynReport(
            run_id=self.run_id,
            generated_at=self.run_id.replace("synthflow_run_", ""),
            n_samples_generated=len(self.data),
            config=self.config,
            imputation_report=self.imputation_report,
            quality_metrics=self.quality_metrics,
            column_descriptions=self.column_descriptions,
        )
        return SynWriter().write(
            data=self.data,
            report_json=report.to_json(),
            run_id=self.run_id,
            source=self.source,
            output_dir=output_dir,
        )

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "backend": self.backend,
            "n_samples": len(self.data),
            "config": self.config,
            "imputation_report": self.imputation_report,
            "quality_metrics": self.quality_metrics,
            "column_descriptions": self.column_descriptions,
        }
