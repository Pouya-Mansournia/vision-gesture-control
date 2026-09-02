"""Temporal validation of raw per-frame gestures.

A single noisy frame must never confirm a gesture. A gesture is confirmed only
after it has been observed for ``confirmation_frames`` consecutive frames. Once
confirmed, it survives short bursts of noise and is only cleared after the same
number of consecutive contradicting frames.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from .gesture_classifier import Gesture


class GestureStabilizer:
    def __init__(self, confirmation_frames: int = 5, window_size: int = 12) -> None:
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be >= 1")
        self.confirmation_frames = confirmation_frames
        self.window_size = max(window_size, confirmation_frames)
        self._window: Deque[Gesture] = deque(maxlen=self.window_size)
        self._confirmed: Gesture = Gesture.UNKNOWN

    @property
    def confirmed(self) -> Gesture:
        return self._confirmed

    def update(self, gesture: Gesture) -> Gesture:
        """Feed one raw gesture; return the currently confirmed gesture."""
        self._window.append(gesture)

        run = 0
        for past in reversed(self._window):
            if past == gesture:
                run += 1
            else:
                break

        if run >= self.confirmation_frames:
            self._confirmed = gesture
        return self._confirmed

    def reset(self) -> None:
        self._window.clear()
        self._confirmed = Gesture.UNKNOWN
