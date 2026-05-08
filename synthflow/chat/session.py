"""
SynSession – stores the conversation history for an auto-mode session.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single conversation turn."""
    user: str
    assistant: str


class SynSession:
    """
    Maintains ordered conversation history.

    Usage:
        session = SynSession()
        session.add("accelerometer, 500Hz", "Got it. Domain: industrial...")
        print(len(session))   # 1
        print(session.turns)  # [Turn(...)]
        session.clear()
    """

    def __init__(self):
        self._turns: list[Turn] = []

    def add(self, user_message: str, assistant_response: str) -> None:
        self._turns.append(Turn(user=user_message, assistant=assistant_response))

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def as_messages(self) -> list[dict]:
        """Return history in Anthropic messages format."""
        messages = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def __len__(self) -> int:
        return len(self._turns)
