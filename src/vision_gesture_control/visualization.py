"""On-frame drawing helpers (landmarks + status overlay)."""

from __future__ import annotations

import cv2
import numpy as np

from .gesture_classifier import Gesture
from .landmarks import HAND_CONNECTIONS, HandLandmarks

_SKELETON = (0, 255, 0)
_JOINT = (255, 255, 255)
_TEXT = (0, 255, 255)
_TEXT_OUTLINE = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_landmarks(frame: np.ndarray, hand: HandLandmarks) -> None:
    height, width = frame.shape[:2]
    pixels = [(int(x * width), int(y * height)) for x, y in hand.xy]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixels[start], pixels[end], _SKELETON, 2, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(frame, pixel, 4, _JOINT, -1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    *,
    raw_gesture: Gesture,
    confirmed_gesture: Gesture,
    fps: float,
    action_status: str,
    on_cooldown: bool,
) -> None:
    lines = [
        f"Gesture:   {raw_gesture.value}",
        f"Confirmed: {confirmed_gesture.value}",
        f"Action:    {action_status}",
        f"FPS:       {fps:5.1f}",
        f"Cooldown:  {'YES' if on_cooldown else 'no'}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), _FONT, 0.6, _TEXT_OUTLINE, 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), _FONT, 0.6, _TEXT, 1, cv2.LINE_AA)
        y += 26
