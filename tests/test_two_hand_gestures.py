"""Unit tests for two-hand gesture classification (heart)."""

from __future__ import annotations

import numpy as np

from vision_gesture_control.gesture_classifier import Gesture, GestureClassifier
from vision_gesture_control.landmarks import NUM_LANDMARKS, HandLandmarks
from vision_gesture_control.landmarks import LandmarkIndex as LM

from test_gesture_classifier import _open_palm  # reuse the open-palm builder


def _hand(pts: np.ndarray) -> HandLandmarks:
    return HandLandmarks(points=pts, handedness="Right", score=1.0)


def _heart_side(x_shift: float) -> np.ndarray:
    """An open palm shifted sideways, with the index tip and thumb tip
    reaching toward the centre so the two hands outline a heart."""
    pts = _open_palm()
    pts[:, 0] += x_shift
    pts[:, 1] -= 0.05
    pts[LM.INDEX_TIP] = (0.50 + np.sign(x_shift) * -0.01, 0.45, 0.0)
    pts[LM.THUMB_TIP] = (0.50, 0.60, 0.0)
    return pts


def test_two_hand_heart_is_recognized() -> None:
    classifier = GestureClassifier()
    left = _hand(_heart_side(-0.15))
    right = _hand(_heart_side(+0.15))
    assert classifier.classify_scene([left, right]).gesture is Gesture.TWO_HAND_HEART


def test_two_open_palms_apart_are_not_a_heart() -> None:
    classifier = GestureClassifier()
    left = _open_palm()
    left[:, 0] -= 0.25
    right = _open_palm()
    right[:, 0] += 0.25
    result = classifier.classify_scene([_hand(left), _hand(right)])
    assert result.gesture is Gesture.OPEN_PALM  # falls back to the first hand


def test_empty_scene_is_unknown() -> None:
    assert GestureClassifier().classify_scene([]).gesture is Gesture.UNKNOWN


def test_single_hand_scene_delegates_to_single_classifier() -> None:
    classifier = GestureClassifier()
    assert classifier.classify_scene([_hand(_open_palm())]).gesture is Gesture.OPEN_PALM
