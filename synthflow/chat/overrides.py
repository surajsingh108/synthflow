"""
Override detection – identifies when a user is correcting a specific field
rather than describing new data, and returns the field + value to patch.

Examples of override messages:
  "change model to TimeVAE"       – {"model": "TimeVAE"}
  "actually sampling rate is 1kHz" – {"sampling_rate_hz": 1000.0}
  "add jitter augmentation"       – patch augmentations list

This runs BEFORE the LM parser. If a direct override is detected,
we skip the full LM parse and apply the patch directly.
"""

from __future__ import annotations

import re


# Simple regex-based override detectors
_OVERRIDES = [
    # model
    (r"(?:change|set|use|switch to)\s+model\s+(?:to\s+)?(\w+)",
     "model", str),
    # sampling rate: "1kHz" or "500 Hz" or "sampling rate is 100"
    (r"sampling\s*rate\s*(?:is|to|=)?\s*([\d.]+)\s*(?:khz|hz)?",
     "sampling_rate_hz",
     lambda v: float(v) * 1000 if "k" in v.lower() else float(v)),
    # n_samples
    (r"(?:generate|make|produce)\s+([\d,]+)\s+samples?",
     "n_samples", lambda v: int(v.replace(",", ""))),
    # domain
    (r"(?:change|set)\s+domain\s+(?:to\s+)?(\w+)",
     "domain", str),
    # device
    (r"(?:use|run on|switch to)\s+(cpu|cuda|gpu)",
     "device", lambda v: "cuda" if v.lower() == "gpu" else v.lower()),
    # batch size
    (r"batch\s*size\s*(?:to|=|is)?\s*(\d+)",
     "batch_size", int),
    # random seed
    (r"(?:seed|random\s*seed)\s*(?:to|=|is)?\s*(\d+)",
     "random_seed", int),
]


def detect_override(message: str) -> dict | None:
    """
    Return a patch dict if a direct field override is detected, else None.
    Only returns the first override found.
    """
    for pattern, field, converter in _OVERRIDES:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            try:
                value = converter(match.group(1))
                return {field: value}
            except (ValueError, TypeError):
                continue
    return None
