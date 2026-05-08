"""
SynState – Finite State Machine for SynFlow auto mode.

States:
  COLLECTING  – receiving user messages, building config
  EXECUTING   – pipeline is running (transient)

Transitions:
  COLLECTING + message    – parse/update config – COLLECTING
  COLLECTING + "generate" – EXECUTING
  COLLECTING + "reset"    – clear config – COLLECTING
  EXECUTING – automatic   – run pipeline – COLLECTING (done)
"""

from __future__ import annotations

from synthflow.chat.session import SynSession
from synthflow.exceptions import SynConfigError


class SynState:
    """
    Manages conversation state for SynFlow auto mode.

    Usage:
        state = SynState()
        state.state          # "COLLECTING"
        state.config         # None until first parse
        state.session        # SynSession instance

        state.update_config(patched_config)
        state.transition("EXECUTING")
        state.reset()
    """

    COLLECTING = "COLLECTING"
    EXECUTING  = "EXECUTING"

    def __init__(self):
        self._state: str = self.COLLECTING
        self._config = None
        self._session: SynSession = SynSession()

    @property
    def state(self) -> str:
        return self._state

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value) -> None:
        self._config = value

    @property
    def session(self) -> SynSession:
        return self._session

    def transition(self, new_state: str) -> None:
        valid = {self.COLLECTING, self.EXECUTING}
        if new_state not in valid:
            raise SynConfigError(
                f"Invalid state transition to '{new_state}'",
                detail=f"Valid states: {valid}",
            )
        self._state = new_state

    def update_config(self, new_config) -> None:
        """Replace or patch the current config."""
        self._config = new_config

    def reset(self) -> None:
        """Wipe config and session, return to COLLECTING."""
        self._state = self.COLLECTING
        self._config = None
        self._session.clear()

    def is_ready(self) -> bool:
        """True if config is set and state is COLLECTING (ready to generate)."""
        return self._config is not None and self._state == self.COLLECTING
