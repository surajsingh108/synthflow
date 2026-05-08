from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthflow.exceptions import SynError


class SynWriter:
    def write(
        self, data, report_json, run_id, source="dataframe", output_dir="./outputs"
    ):
        run_dir = Path(output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        data_path = self._data_path(run_dir, source)
        self._write_data(data, data_path)
        (run_dir / "run_report.json").write_text(report_json, encoding="utf-8")
        return run_dir

    def _data_path(self, run_dir, source):
        if source == "dataframe":
            return run_dir / "data_synthetic.csv"
        src = Path(source)
        suffix = src.suffix.lower() if src.suffix else ".csv"
        if suffix not in (".csv", ".json", ".xlsx"):
            suffix = ".csv"
        return run_dir / f"{src.stem}_synthetic{suffix}"

    def _write_data(self, data, path):
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                data.to_csv(path, index=False)
            elif suffix == ".json":
                data.to_json(path, orient="records", indent=2)
            elif suffix in (".xlsx", ".xls"):
                data.to_excel(path, index=False)
            else:
                data.to_csv(path, index=False)
        except Exception as exc:
            raise SynError(
                f"Failed to write output file: {path}", detail=str(exc)
            ) from exc
