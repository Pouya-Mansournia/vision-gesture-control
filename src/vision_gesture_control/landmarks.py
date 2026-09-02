"""Backend-agnostic hand landmark representation.

This module depends only on :mod:`numpy` on purpose: the gesture logic and its
unit tests must never need to import OpenCV or MediaPipe. Third-party detector
output is converted into :class:`HandLandmarks` at the edge of the system
(see :mod:`vision_gesture_control.hand_tracker`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

NUM_LANDMARKS = 21


class LandmarkIndex(IntEnum):
    """Indices of the 21 hand landmarks (MediaPipe Hands topology)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Pairs of landmark indices that form the drawable hand skeleton.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class HandLandmarks:
    """Normalized hand landmarks in image space.

    ``points`` has shape ``(21, 3)``. ``x`` and ``y`` are normalized to
    ``[0, 1]`` relative to frame width/height; ``z`` is the detector's relative
    depth (smaller = closer to the camera). Image convention: ``y`` grows
    downward, so "up" in the physical world means a *smaller* ``y``.
    """

    points: np.ndarray
    handedness: str = "Unknown"
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.points.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"expected points of shape ({NUM_LANDMARKS}, 3), got {self.points.shape}"
            )

    def point(self, index: LandmarkIndex) -> np.ndarray:
        """Return the ``(x, y, z)`` vector for a single landmark."""
        return self.points[int(index)]

    @property
    def xy(self) -> np.ndarray:
        """All landmark positions projected to the image plane, shape ``(21, 2)``."""
        return self.points[:, :2]

    @classmethod
    def from_iterable(
        cls,
        coords: Iterable[Iterable[float]],
        handedness: str = "Right",
        score: float = 1.0,
    ) -> "HandLandmarks":
        """Build from an iterable of ``(x, y)`` or ``(x, y, z)`` tuples."""
        arr = np.asarray(list(coords), dtype=float)
        if arr.shape == (NUM_LANDMARKS, 2):
            arr = np.hstack([arr, np.zeros((NUM_LANDMARKS, 1))])
        return cls(points=arr, handedness=handedness, score=score)
