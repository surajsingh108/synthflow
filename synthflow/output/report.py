from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SynReport:
    run_id: str
    generated_at: str
    n_samples_generated: int
    config: dict
    imputation_report: dict
    quality_metrics: dict
    column_descriptions: dict

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "n_samples_generated": self.n_samples_generated,
            "config": self.config,
            "imputation_report": self.imputation_report,
            "quality_metrics": self.quality_metrics,
            "column_descriptions": self.column_descriptions,
        }

    def to_json(self, path=None):
        json_str = json.dumps(self.as_dict(), indent=2, default=str)
        if path is not None:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    @staticmethod
    def make_run_id():
        return "synthflow_run_" + datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def build_column_descriptions(synth_df, signal_cols):
        import numpy as np

        desc = {}
        for col in signal_cols:
            synth_col = f"synthetic_{col}"
            if synth_col not in synth_df.columns:
                continue
            series = synth_df[synth_col].dropna()
            desc[synth_col] = {
                "original_col": col,
                "dtype": str(synth_df[synth_col].dtype),
                "range": [
                    round(float(series.min()), 4),
                    round(float(series.max()), 4),
                ],
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "unit": "unknown",
            }
        return desc
