"""Unit tests for the geometric gesture classifier using synthetic landmarks.

Image convention: y grows downward, so "up" == smaller y.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_gesture_control.gesture_classifier import Gesture, GestureClassifier
from vision_gesture_control.landmarks import NUM_LANDMARKS, HandLandmarks
from vision_gesture_control.landmarks import LandmarkIndex as LM

_FINGERS = {
    "index": (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
    "middle": (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    "ring": (LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP),
    "pinky": (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP),
}


def _open_palm() -> np.ndarray:
    """Hand centred at the wrist with all fingers extended upward."""
    pts = np.zeros((NUM_LANDMARKS, 3), dtype=float)
    pts[LM.WRIST] = (0.50, 0.90, 0.0)
    pts[LM.THUMB_CMC] = (0.44, 0.84, 0.0)
    pts[LM.THUMB_MCP] = (0.40, 0.78, 0.0)
    pts[LM.THUMB_IP] = (0.37, 0.72, 0.0)
    pts[LM.THUMB_TIP] = (0.35, 0.66, 0.0)
    pts[LM.INDEX_MCP] = (0.46, 0.68, 0.0)
    pts[LM.INDEX_PIP] = (0.46, 0.60, 0.0)
    pts[LM.INDEX_DIP] = (0.46, 0.55, 0.0)
    pts[LM.INDEX_TIP] = (0.46, 0.50, 0.0)
    pts[LM.MIDDLE_MCP] = (0.50, 0.67, 0.0)
    pts[LM.MIDDLE_PIP] = (0.50, 0.58, 0.0)
    pts[LM.MIDDLE_DIP] = (0.50, 0.52, 0.0)
    pts[LM.MIDDLE_TIP] = (0.50, 0.47, 0.0)
    pts[LM.RING_MCP] = (0.54, 0.68, 0.0)
    pts[LM.RING_PIP] = (0.54, 0.60, 0.0)
    pts[LM.RING_DIP] = (0.54, 0.55, 0.0)
    pts[LM.RING_TIP] = (0.54, 0.50, 0.0)
    pts[LM.PINKY_MCP] = (0.58, 0.70, 0.0)
    pts[LM.PINKY_PIP] = (0.58, 0.63, 0.0)
    pts[LM.PINKY_DIP] = (0.58, 0.59, 0.0)
    pts[LM.PINKY_TIP] = (0.58, 0.55, 0.0)
    return pts


def _fold(pts: np.ndarray, finger: str) -> None:
    mcp, pip, dip, tip = _FINGERS[finger]
    x, y = pts[mcp, 0], pts[mcp, 1]
    pts[pip] = (x, y + 0.03, 0.0)
    pts[dip] = (x, y + 0.05, 0.0)
    pts[tip] = (x, y + 0.04, 0.0)


def _thumbs_up() -> np.ndarray:
    pts = _open_palm()
    for finger in _FINGERS:
        _fold(pts, finger)
    pts[LM.THUMB_CMC] = (0.46, 0.84, 0.0)
    pts[LM.THUMB_MCP] = (0.45, 0.75, 0.0)
    pts[LM.THUMB_IP] = (0.44, 0.66, 0.0)
    pts[LM.THUMB_TIP] = (0.43, 0.56, 0.0)
    return pts


def _hand(pts: np.ndarray) -> HandLandmarks:
    return HandLandmarks(points=pts, handedness="Right", score=1.0)


@pytest.fixture
def classifier() -> GestureClassifier:
    return GestureClassifier()


def test_thumbs_up_is_recognized(classifier: GestureClassifier) -> None:
    result = classifier.classify(_hand(_thumbs_up()))
    assert result.gesture is Gesture.THUMBS_UP
    assert result.confidence > 0.0


def test_thumbs_down_is_recognized(classifier: GestureClassifier) -> None:
    pts = _thumbs_up()
    pts[:, 1] = 1.8 - pts[:, 1]  # mirror vertically about the wrist row (y=0.9)
    result = classifier.classify(_hand(pts))
    assert result.gesture is Gesture.THUMBS_DOWN
    assert result.confidence > 0.0


def test_thumbs_up_is_resolution_independent(classifier: GestureClassifier) -> None:
    pts = _thumbs_up()
    pts[:, 0] *= 0.25  # simulate a very different aspect / scale
    pts[:, 1] *= 0.25
    assert classifier.classify(_hand(pts)).gesture is Gesture.THUMBS_UP


def test_open_palm_does_not_trigger(classifier: GestureClassifier) -> None:
    assert classifier.classify(_hand(_open_palm())).gesture is Gesture.UNKNOWN


def test_fist_does_not_trigger(classifier: GestureClassifier) -> None:
    pts = _open_palm()
    for finger in _FINGERS:
        _fold(pts, finger)
    # thumb tucked across the palm (not extended)
    pts[LM.THUMB_MCP] = (0.48, 0.80, 0.0)
    pts[LM.THUMB_IP] = (0.50, 0.78, 0.0)
    pts[LM.THUMB_TIP] = (0.52, 0.77, 0.0)
    assert classifier.classify(_hand(pts)).gesture is Gesture.UNKNOWN


def test_degenerate_hand_is_unknown(classifier: GestureClassifier) -> None:
    pts = np.full((NUM_LANDMARKS, 3), 0.5, dtype=float)
    assert classifier.classify(_hand(pts)).gesture is Gesture.UNKNOWN
