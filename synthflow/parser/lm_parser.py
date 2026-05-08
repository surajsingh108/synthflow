"""
SynParser – calls Claude API to parse natural language into SynConfig patches.

Two-attempt retry: if first response produces empty/invalid JSON, retry once.
Falls back to an empty patch (no change) if both attempts fail.
"""

from __future__ import annotations

from synthflow.parser.prompts import EXTRACTION_SYSTEM_PROMPT
from synthflow.parser.guardrails import parse_llm_response
from synthflow.exceptions import SynConfigError


class SynParser:
    """
    Wraps the Claude API for natural language → SynConfig extraction.

    Usage:
        parser = SynParser(api_key="sk-ant-...")
        patch = parser.parse("accelerometer, wind turbine, 500Hz")
        # → {"domain": "industrial", "sensor_type": "accelerometer", ...}
    """

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 512
    MAX_RETRIES = 2

    def __init__(self, api_key: str):
        self.api_key = api_key

    def parse(
        self,
        message: str,
        current_config=None,
    ) -> dict:
        """
        Parse a user message into a SynConfig patch dict.

        Args:
            message        : user's natural language input
            current_config : current SynConfig (for context), or None

        Returns:
            dict of field → value to patch into current config.
            Empty dict if parsing fails.
        """
        try:
            import anthropic
        except ImportError as e:
            raise SynConfigError(
                "anthropic package not installed.",
                detail="Run: pip install anthropic",
            ) from e

        client = anthropic.Anthropic(api_key=self.api_key)

        # build user message with current config context
        if current_config is not None:
            try:
                ctx = current_config.model_dump()
            except Exception:
                ctx = str(current_config)
            user_content = (
                f"Current config: {ctx}\n\n"
                f"User says: {message}"
            )
        else:
            user_content = f"User says: {message}"

        # attempt parse with retry
        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                text = response.content[0].text
                patch = parse_llm_response(text)
                if patch:
                    return patch
                # empty patch → retry with explicit instruction
                user_content = (
                    f"{user_content}\n\n"
                    "IMPORTANT: Return ONLY a JSON object. No text before or after."
                )
            except anthropic.AuthenticationError as e:
                raise SynConfigError(
                    "Invalid Anthropic API key.",
                    detail="Check your ANTHROPIC_API_KEY.",
                ) from e
            except Exception:
                continue

        # both attempts failed → return empty patch
        return {}

    def explain(self, config) -> str:
        """
        Ask Claude to explain the current config decisions.
        Returns explanation string.
        """
        try:
            import anthropic
            from synthflow.parser.prompts import EXPLAIN_SYSTEM_PROMPT
        except ImportError:
            return "Cannot explain: anthropic not installed."

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.MODEL,
                max_tokens=1024,
                system=EXPLAIN_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Current config:\n{config.summary()}"
                }],
            )
            return response.content[0].text
        except Exception as exc:
            return f"Could not explain config: {exc}"
