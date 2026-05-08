"""
SynFlow – main entry point for the synthflow package.

This is a stub for Stage 1. Full implementation comes in Stage 9.
The class is importable and instantiable now so that all other
modules can reference it without circular import issues.
"""

from __future__ import annotations

from synthflow.exceptions import SynConfigError


class SynFlow:
    """
    Main synthflow interface.

    Modes:
        "auto"   – natural language input via Claude API
        "manual" – direct SynConfig object, no API calls

    Args:
        mode     : "auto" or "manual"
        api_key  : Anthropic API key (required for auto mode)
        config   : SynConfig instance (required for manual mode)
        data     : path to input file, or None
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        mode: str = "auto",
        api_key: str | None = None,
        config=None,
        data: str | None = None,
    ):
        if mode not in ("auto", "manual"):
            raise SynConfigError(
                f"Invalid mode: '{mode}'",
                detail="mode must be 'auto' or 'manual'",
            )

        if mode == "auto" and api_key is None:
            # allow env var fallback – full logic in Stage 9
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise SynConfigError(
                    "api_key is required for auto mode",
                    detail="pass api_key= or set ANTHROPIC_API_KEY env var",
                )

        if mode == "manual" and config is None:
            raise SynConfigError(
                "config is required for manual mode",
                detail="pass a SynConfig instance",
            )

        self.mode = mode
        self.api_key = api_key
        self.config = config
        self.data = data
        self._result = None

    def chat(self, message: str) -> str:
        """
        Send a natural language message (auto mode only).
        Full implementation in Stage 9.
        """
        # TODO: Stage 9 – wire to SynParser and SynState
        raise NotImplementedError("chat() implemented in Stage 9")

    def generate(self):
        """
        Run the full pipeline and return a SynResult.
        Full implementation in Stage 9.
        """
        # TODO: Stage 9 – wire all modules together
        raise NotImplementedError("generate() implemented in Stage 9")

    def __repr__(self) -> str:
        return (
            f"SynFlow(mode='{self.mode}', "
            f"data='{self.data}', "
            f"config={'set' if self.config else 'not set'})"
        )
