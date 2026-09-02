"""Unit tests for temporal gesture stabilization."""

from __future__ import annotations

import pytest

from vision_gesture_control.gesture_classifier import Gesture
from vision_gesture_control.gesture_stabilizer import GestureStabilizer


def test_single_frame_does_not_confirm() -> None:
    stabilizer = GestureStabilizer(confirmation_frames=5, window_size=10)
    assert stabilizer.update(Gesture.THUMBS_UP) is Gesture.UNKNOWN


def test_stable_sequence_becomes_confirmed() -> None:
    stabilizer = GestureStabilizer(confirmation_frames=5, window_size=10)
    result = Gesture.UNKNOWN
    for _ in range(5):
        result = stabilizer.update(Gesture.THUMBS_UP)
    assert result is Gesture.THUMBS_UP
    assert stabilizer.confirmed is Gesture.THUMBS_UP


def test_unstable_sequence_is_rejected() -> None:
    stabilizer = GestureStabilizer(confirmation_frames=5, window_size=12)
    result = Gesture.UNKNOWN
    for gesture in [Gesture.THUMBS_UP, Gesture.UNKNOWN] * 6:
        result = stabilizer.update(gesture)
    assert result is Gesture.UNKNOWN


def test_confirmed_gesture_survives_single_noise_frame() -> None:
    stabilizer = GestureStabilizer(confirmation_frames=3, window_size=10)
    for _ in range(3):
        stabilizer.update(Gesture.THUMBS_DOWN)
    assert stabilizer.update(Gesture.UNKNOWN) is Gesture.THUMBS_DOWN


def test_reset_clears_state() -> None:
    stabilizer = GestureStabilizer(confirmation_frames=2, window_size=6)
    stabilizer.update(Gesture.THUMBS_UP)
    stabilizer.update(Gesture.THUMBS_UP)
    stabilizer.reset()
    assert stabilizer.confirmed is Gesture.UNKNOWN
    assert stabilizer.update(Gesture.THUMBS_UP) is Gesture.UNKNOWN


def test_invalid_confirmation_frames_rejected() -> None:
    with pytest.raises(ValueError):
        GestureStabilizer(confirmation_frames=0)
