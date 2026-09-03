"""Unit tests for single-hand geometric gesture classification.

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
    """Hand centred at the wrist with all five fingers extended upward."""
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


def _tuck_thumb(pts: np.ndarray) -> None:
    pts[LM.THUMB_MCP] = (0.48, 0.80, 0.0)
    pts[LM.THUMB_IP] = (0.50, 0.78, 0.0)
    pts[LM.THUMB_TIP] = (0.52, 0.77, 0.0)


def _loose_fold(pts: np.ndarray, finger: str) -> None:
    """A gentle curl: the tip drops just inside its own PIP joint, the way a
    real hand curls the spare fingers for a one-finger pose."""
    mcp, pip, dip, tip = _FINGERS[finger]
    px, py = pts[pip, 0], pts[pip, 1]
    pts[dip] = (px, py + 0.015, 0.0)
    pts[tip] = (px, py + 0.03, 0.0)


def _rotate(pts: np.ndarray, radians: float) -> np.ndarray:
    """Rotate the hand about the wrist (distance-to-wrist preserving)."""
    out = pts.copy()
    wrist = out[LM.WRIST, :2].copy()
    c, s = np.cos(radians), np.sin(radians)
    rot = np.array([[c, -s], [s, c]])
    out[:, :2] = (out[:, :2] - wrist) @ rot.T + wrist
    return out


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
    assert classifier.classify(_hand(pts)).gesture is Gesture.THUMBS_DOWN


def test_thumbs_up_is_resolution_independent(classifier: GestureClassifier) -> None:
    pts = _thumbs_up()
    pts[:, :2] *= 0.25
    assert classifier.classify(_hand(pts)).gesture is Gesture.THUMBS_UP


def test_open_palm_is_recognized(classifier: GestureClassifier) -> None:
    assert classifier.classify(_hand(_open_palm())).gesture is Gesture.OPEN_PALM


def test_fist_is_recognized(classifier: GestureClassifier) -> None:
    pts = _open_palm()
    for finger in _FINGERS:
        _fold(pts, finger)
    _tuck_thumb(pts)
    assert classifier.classify(_hand(pts)).gesture is Gesture.FIST


def test_peace_is_recognized(classifier: GestureClassifier) -> None:
    pts = _open_palm()
    _fold(pts, "ring")
    _fold(pts, "pinky")
    assert classifier.classify(_hand(pts)).gesture is Gesture.PEACE


def test_index_up_is_recognized(classifier: GestureClassifier) -> None:
    pts = _open_palm()
    _fold(pts, "middle")
    _fold(pts, "ring")
    _fold(pts, "pinky")
    _tuck_thumb(pts)
    assert classifier.classify(_hand(pts)).gesture is Gesture.INDEX_UP


def test_middle_finger_is_recognized(classifier: GestureClassifier) -> None:
    pts = _open_palm()
    _fold(pts, "index")
    _fold(pts, "ring")
    _fold(pts, "pinky")
    _tuck_thumb(pts)
    assert classifier.classify(_hand(pts)).gesture is Gesture.MIDDLE_FINGER


def test_middle_finger_with_a_loose_curl(classifier: GestureClassifier) -> None:
    # The spare fingers are only gently curled, not clenched to the palm.
    pts = _open_palm()
    _loose_fold(pts, "index")
    _loose_fold(pts, "ring")
    _loose_fold(pts, "pinky")
    _tuck_thumb(pts)
    assert classifier.classify(_hand(pts)).gesture is Gesture.MIDDLE_FINGER


def test_middle_finger_when_the_hand_is_tilted(classifier: GestureClassifier) -> None:
    pts = _open_palm()
    _fold(pts, "index")
    _fold(pts, "ring")
    _fold(pts, "pinky")
    _tuck_thumb(pts)
    assert classifier.classify(_hand(_rotate(pts, 0.30))).gesture is Gesture.MIDDLE_FINGER
    assert classifier.classify(_hand(_rotate(pts, -0.30))).gesture is Gesture.MIDDLE_FINGER


def test_degenerate_hand_is_unknown(classifier: GestureClassifier) -> None:
    pts = np.full((NUM_LANDMARKS, 3), 0.5, dtype=float)
    assert classifier.classify(_hand(pts)).gesture is Gesture.UNKNOWN
