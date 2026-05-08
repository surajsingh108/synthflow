"""
Magic word detection – identifies special commands in user messages.

Magic words trigger pipeline actions rather than LM parsing:
  "generate"    – lock config and run the full synthesis pipeline
  "reset"       – wipe config and start over
  "show config" – print current config dict
  "explain"     – LM explains each config decision

Detection is case-insensitive and tolerant of surrounding text.
"""

from __future__ import annotations

_MAGIC_MAP = {
    "reset":      ["start over", "reset", "clear", "wipe", "restart"],
    "show config":["show config", "show me the config", "what's the config",
                   "print config", "current config", "display config"],
    "explain":    ["explain", "why did you", "how did you decide", "reasoning"],
    "generate":   ["generate", "run", "go", "start", "execute", "synthesize", "make it"],
}


def detect_magic_word(message: str) -> str | None:
    """
    Return the canonical magic word if found, else None.

    Checks each magic word category in priority order:
      reset > show config > explain > generate
    (Longer phrases like "start over" checked before shorter ones like "start")
    """
    lower = message.strip().lower()
    for canonical, triggers in _MAGIC_MAP.items():
        for trigger in triggers:
            if trigger in lower:
                return canonical
    return None
