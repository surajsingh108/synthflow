"""
SynIngestor – main entry point for data ingestion.

Accepts a file path (CSV, Excel, JSON) or a pandas DataFrame directly.
Runs detection and validation, then returns a LoadResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from synthflow.exceptions import SynIngestError
from synthflow.ingestor.loaders import load_csv, load_excel, load_json
from synthflow.ingestor.detector import SynDetector
from synthflow.ingestor.validator import SynValidator


@dataclass
class LoadResult:
    """
    Result returned by SynIngestor.load().

    Attributes:
        data            : cleaned DataFrame (all columns preserved)
        timestamp_col   : name of detected timestamp column, or None
        signal_cols     : list of numeric (signal) column names
        sampling_rate_hz: inferred sampling rate, or None if not detected
        n_rows          : number of rows
        n_cols          : number of columns
        missing_pct     : overall % of missing values (0.0 – 100.0)
        source          : original file path string, or "dataframe"
        warnings        : list of non-fatal warning strings
    """

    data: pd.DataFrame
    timestamp_col: str | None
    signal_cols: list[str]
    sampling_rate_hz: float | None
    n_rows: int
    n_cols: int
    missing_pct: float
    source: str
    warnings: list[str] = field(default_factory=list)


class SynIngestor:
    """
    Loads sensor data from file or DataFrame.

    Supported formats:
        .csv        – comma-separated values
        .xlsx/.xls  – Excel workbook (first sheet)
        .json       – JSON array of records or time-series DB export

    Usage:
        ingestor = SynIngestor()
        result = ingestor.load("sensor_data.csv")
        print(result.sampling_rate_hz)
        print(result.signal_cols)
    """

    def load(
        self,
        source: str | Path | pd.DataFrame,
        timestamp_col: str | None = None,
        signal_cols: list[str] | None = None,
    ) -> LoadResult:
        """
        Load data from file path or DataFrame.

        Args:
            source        : file path string, Path object, or DataFrame
            timestamp_col : hint – name of the timestamp column.
                            If None, SynDetector tries to find it.
            signal_cols   : hint – list of signal column names.
                            If None, SynDetector infers them.

        Returns:
            LoadResult with data and metadata.

        Raises:
            SynIngestError: if file not found, format unsupported,
                            or data is empty / unreadable.
        """
        warnings: list[str] = []

        # – load raw DataFrame ––––––––––––––––––––––––––––––––––––––––––––––
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            source_label = "dataframe"
        else:
            path = Path(source)
            source_label = str(path)
            if not path.exists():
                raise SynIngestError(
                    f"File not found: {path}",
                    detail="Check the path and try again.",
                )
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = load_csv(path)
            elif suffix in (".xlsx", ".xls"):
                df = load_excel(path)
            elif suffix == ".json":
                df = load_json(path)
            else:
                raise SynIngestError(
                    f"Unsupported file format: '{suffix}'",
                    detail="Supported formats: .csv, .xlsx, .xls, .json",
                )

        # – basic emptiness check –––––––––––––––––––––––––––––––––––––––––––
        if df.empty:
            raise SynIngestError(
                "Loaded dataset is empty.",
                detail=f"Source: {source_label}",
            )

        if len(df.columns) < 1:
            raise SynIngestError(
                "Dataset has no columns.",
                detail=f"Source: {source_label}",
            )

        # – detect ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        detector = SynDetector(df)

        ts_col = timestamp_col or detector.find_timestamp_col()
        sig_cols = signal_cols or detector.find_signal_cols(exclude=[ts_col])
        sampling_rate = detector.infer_sampling_rate(ts_col)

        if not sig_cols:
            warnings.append(
                "No numeric signal columns detected. "
                "Check that your data contains numeric sensor readings."
            )

        if sampling_rate is None and ts_col is not None:
            warnings.append(
                "Could not infer sampling rate from the timestamp column. "
                "The timestamps may be irregular or non-numeric."
            )

        # – validate ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        validator = SynValidator(df)
        validation_warnings = validator.run(
            timestamp_col=ts_col,
            signal_cols=sig_cols,
        )
        warnings.extend(validation_warnings)

        # – compute summary stats –––––––––––––––––––––––––––––––––––––––––––
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        missing_pct = (
            round(df.isna().sum().sum() / total_cells * 100, 2)
            if total_cells > 0
            else 0.0
        )

        return LoadResult(
            data=df,
            timestamp_col=ts_col,
            signal_cols=sig_cols,
            sampling_rate_hz=sampling_rate,
            n_rows=n_rows,
            n_cols=n_cols,
            missing_pct=missing_pct,
            source=source_label,
            warnings=warnings,
        )
