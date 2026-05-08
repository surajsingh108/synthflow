"""
Guardrails – JSON repair, retry logic, and field validation.
"""

from __future__ import annotations

import json
import re


def repair_json(text: str) -> str:
    """
    Extract and repair JSON from LM output.
    Handles: markdown fences, leading/trailing text, single quotes.
    """
    # strip markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start:end + 1]

    # replace single quotes with double quotes (naive but often works)
    text = text.replace("'", '"')

    return text


def validate_patch(patch: dict, valid_fields: set) -> dict:
    """
    Remove fields not in the SynConfig schema.
    Returns a cleaned patch dict.
    """
    return {k: v for k, v in patch.items() if k in valid_fields and v is not None}


VALID_SYNCONFIG_FIELDS = {
    "domain", "sensor_type", "sampling_rate_hz", "description",
    "missing_pattern", "imputation_strategy", "imputation_overrides",
    "backend", "model", "n_samples", "augmentations",
    "random_seed", "device", "batch_size",
}


def parse_llm_response(response_text: str) -> dict:
    """
    Parse LM response text into a patch dict.
    Returns empty dict on failure (caller retries).
    """
    try:
        repaired = repair_json(response_text)
        patch = json.loads(repaired)
        if not isinstance(patch, dict):
            return {}
        return validate_patch(patch, VALID_SYNCONFIG_FIELDS)
    except (json.JSONDecodeError, Exception):
        return {}
