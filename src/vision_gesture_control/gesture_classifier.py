"""Geometric (landmark-based) gesture classification.

The classifier is a pure function of hand landmarks. It performs **no** I/O and
triggers **no** actions, which keeps it deterministic and unit-testable with
synthetic coordinates.

Operating assumptions for V1 (documented on purpose):

* One hand, roughly facing the camera, fingers pointing broadly up or down.
* The hand is not heavily rotated in-plane (no full rotation invariance).
* Distances are normalized by the wrist -> middle-finger-MCP length so the
  logic is independent of frame resolution and the hand's distance to camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .landmarks import HandLandmarks
from .landmarks import LandmarkIndex as LM

_EPS = 1e-9


class Gesture(str, Enum):
    """All gestures the architecture is designed to support.

    Only ``UNKNOWN``, ``THUMBS_UP`` and ``THUMBS_DOWN`` are implemented in V1;
    the rest are reserved so new gestures can be added without touching the
    application wiring.
    """

    UNKNOWN = "UNKNOWN"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    INDEX_UP = "INDEX_UP"
    INDEX_DOWN = "INDEX_DOWN"
    PEACE = "PEACE"
    OK = "OK"


@dataclass(frozen=True)
class ClassificationResult:
    gesture: Gesture
    confidence: float


# Non-thumb fingers described as (tip, pip, mcp) landmark indices.
_NON_THUMB_FINGERS: tuple[tuple[LM, LM, LM], ...] = (
    (LM.INDEX_TIP, LM.INDEX_PIP, LM.INDEX_MCP),
    (LM.MIDDLE_TIP, LM.MIDDLE_PIP, LM.MIDDLE_MCP),
    (LM.RING_TIP, LM.RING_PIP, LM.RING_MCP),
    (LM.PINKY_TIP, LM.PINKY_PIP, LM.PINKY_MCP),
)


@dataclass(frozen=True)
class GestureClassifier:
    """Classify a single :class:`HandLandmarks` into a :class:`Gesture`.

    Parameters
    ----------
    thumb_extension_threshold:
        Minimum thumb length (tip -> CMC), normalized by hand scale, for the
        thumb to count as "extended".
    finger_fold_threshold:
        A non-thumb finger is "folded" when ``dist(tip, wrist)`` is below
        ``finger_fold_threshold * dist(mcp, wrist)`` (the curled tip moves back
        toward the palm).
    thumb_vertical_ratio:
        The vertical component of the thumb direction must exceed this multiple
        of the horizontal component for the thumb to be considered up/down.
    """

    thumb_extension_threshold: float = 0.6
    finger_fold_threshold: float = 1.1
    thumb_vertical_ratio: float = 1.2

    def classify(self, hand: HandLandmarks) -> ClassificationResult:
        pts = hand.points
        wrist = pts[LM.WRIST][:2]
        middle_mcp = pts[LM.MIDDLE_MCP][:2]
        scale = float(np.linalg.norm(middle_mcp - wrist))
        if scale < 1e-4:
            return ClassificationResult(Gesture.UNKNOWN, 0.0)

        folded_flags = self._folded_flags(pts, wrist)
        fold_fraction = float(np.mean(folded_flags))
        all_folded = bool(np.all(folded_flags))

        thumb_tip = pts[LM.THUMB_TIP][:2]
        thumb_cmc = pts[LM.THUMB_CMC][:2]
        thumb_mcp = pts[LM.THUMB_MCP][:2]

        thumb_len_ratio = float(np.linalg.norm(thumb_tip - thumb_cmc)) / scale
        thumb_extended = thumb_len_ratio >= self.thumb_extension_threshold

        direction = thumb_tip - thumb_mcp  # image space: +y is downward
        dx, dy = abs(float(direction[0])), float(direction[1])
        vertical_enough = abs(dy) >= self.thumb_vertical_ratio * dx

        if not (thumb_extended and all_folded and vertical_enough):
            return ClassificationResult(Gesture.UNKNOWN, 0.0)

        conf_thumb = _saturate((thumb_len_ratio - self.thumb_extension_threshold)
                               / self.thumb_extension_threshold)
        conf_dir = _saturate((abs(dy) - self.thumb_vertical_ratio * dx) / (abs(dy) + _EPS))
        confidence = float((conf_thumb + conf_dir + fold_fraction) / 3.0)

        if dy < 0 and thumb_tip[1] < wrist[1]:
            return ClassificationResult(Gesture.THUMBS_UP, confidence)
        if dy > 0 and thumb_tip[1] > wrist[1]:
            return ClassificationResult(Gesture.THUMBS_DOWN, confidence)
        return ClassificationResult(Gesture.UNKNOWN, 0.0)

    def _folded_flags(self, pts: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        flags = []
        for tip, _pip, mcp in _NON_THUMB_FINGERS:
            tip_to_wrist = float(np.linalg.norm(pts[tip][:2] - wrist))
            mcp_to_wrist = float(np.linalg.norm(pts[mcp][:2] - wrist))
            flags.append(tip_to_wrist < self.finger_fold_threshold * mcp_to_wrist)
        return np.asarray(flags, dtype=bool)


def _saturate(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
