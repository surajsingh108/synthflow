"""
File format loaders – each returns a raw pandas DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthflow.exceptions import SynIngestError


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file.

    Tries UTF-8 first, falls back to latin-1 if encoding fails.
    Raises SynIngestError on parse failure.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise SynIngestError(
                f"Failed to parse CSV: {path.name}",
                detail=str(exc),
            ) from exc
    raise SynIngestError(
        f"Could not decode CSV file: {path.name}",
        detail="Tried utf-8 and latin-1 encodings.",
    )


def load_excel(path: Path) -> pd.DataFrame:
    """
    Load the first sheet of an Excel file (.xlsx or .xls).
    Raises SynIngestError on failure.
    """
    try:
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except Exception as exc:
        raise SynIngestError(
            f"Failed to parse Excel file: {path.name}",
            detail=str(exc),
        ) from exc


def load_json(path: Path) -> pd.DataFrame:
    """
    Load a JSON file.

    Supports two formats:
      1. Array of records: [{"ts": 0, "val": 1.2}, ...]
      2. Time-series DB export: {"data": [...], "columns": [...]}
         (common in InfluxDB / TimescaleDB JSON exports)

    Raises SynIngestError on failure.
    """
    import json

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SynIngestError(
            f"Failed to read JSON file: {path.name}",
            detail=str(exc),
        ) from exc

    # format 1: top-level list
    if isinstance(raw, list):
        try:
            return pd.DataFrame(raw)
        except Exception as exc:
            raise SynIngestError(
                "JSON array could not be converted to DataFrame.",
                detail=str(exc),
            ) from exc

    # format 2: dict with "data" key
    if isinstance(raw, dict):
        if "data" in raw:
            records = raw["data"]
            columns = raw.get("columns")
            try:
                if columns:
                    return pd.DataFrame(records, columns=columns)
                return pd.DataFrame(records)
            except Exception as exc:
                raise SynIngestError(
                    "JSON 'data' field could not be converted to DataFrame.",
                    detail=str(exc),
                ) from exc
        # flat dict – treat as single-row
        try:
            return pd.DataFrame([raw])
        except Exception as exc:
            raise SynIngestError(
                "JSON object could not be converted to DataFrame.",
                detail=str(exc),
            ) from exc

    raise SynIngestError(
        "Unrecognised JSON structure.",
        detail="Expected a list of records or a dict with a 'data' key.",
    )
