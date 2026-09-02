"""Map confirmed gestures to laptop actions, with hold/cooldown semantics.

Design notes
------------
* The dispatcher is OS-agnostic. Sound is produced through a small
  :class:`SoundBackend` protocol; Windows uses :mod:`winsound`, everything else
  falls back to a logging backend so the app never crashes for lack of audio.
* "Fire once per hold": after an action fires for a gesture, it will not fire
  again until the confirmed gesture changes (including returning to
  ``UNKNOWN``).
* "Cooldown": a global minimum time between two fired actions, so switching
  quickly between gestures cannot produce a burst.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .gesture_classifier import Gesture

logger = logging.getLogger(__name__)


class SoundBackend:
    """Interface for producing a beep. Implementations must not raise."""

    def beep(self, frequency_hz: int, duration_ms: int) -> None:  # pragma: no cover
        raise NotImplementedError


class WinsoundBackend(SoundBackend):
    def __init__(self) -> None:
        import winsound

        self._winsound = winsound

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        try:
            self._winsound.Beep(int(frequency_hz), int(duration_ms))
        except Exception:  # pragma: no cover - hardware/driver dependent
            logger.warning("winsound.Beep failed", exc_info=True)


class LoggingBackend(SoundBackend):
    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        logger.info("BEEP %d Hz for %d ms (no audio backend available)",
                    frequency_hz, duration_ms)


def create_default_sound_backend() -> SoundBackend:
    if platform.system() == "Windows":
        try:
            return WinsoundBackend()
        except Exception:  # pragma: no cover
            logger.warning("winsound unavailable; using logging backend", exc_info=True)
    return LoggingBackend()


@dataclass(frozen=True)
class Action:
    name: str
    run: Callable[[], None]


def beep_action(name: str, backend: SoundBackend,
                frequency_hz: int, duration_ms: int) -> Action:
    def _run() -> None:
        backend.beep(frequency_hz, duration_ms)

    return Action(name=name, run=_run)


class ActionDispatcher:
    def __init__(
        self,
        mapping: Dict[Gesture, Action],
        cooldown_seconds: float = 1.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mapping = dict(mapping)
        self._cooldown = cooldown_seconds
        self._now = time_source
        self._armed_gesture: Optional[Gesture] = None
        self._last_trigger_time: float = float("-inf")
        self.last_action_name: Optional[str] = None

    def on_cooldown(self) -> bool:
        return (self._now() - self._last_trigger_time) < self._cooldown

    def handle(self, confirmed: Optional[Gesture]) -> Optional[Action]:
        """Process the confirmed gesture for this frame; maybe fire an action."""
        if confirmed is None or confirmed == Gesture.UNKNOWN:
            self._armed_gesture = None
            return None

        if confirmed == self._armed_gesture:
            return None  # still held; already fired for this hold

        action = self._mapping.get(confirmed)
        if action is None:
            self._armed_gesture = confirmed
            return None

        if self.on_cooldown():
            return None

        action.run()
        self._armed_gesture = confirmed
        self._last_trigger_time = self._now()
        self.last_action_name = action.name
        logger.info("action triggered: %s (gesture=%s)", action.name, confirmed.value)
        return action
