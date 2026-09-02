"""Hand landmark extraction backed by the MediaPipe Tasks ``HandLandmarker``.

The raw MediaPipe result is converted immediately into the neutral
:class:`~vision_gesture_control.landmarks.HandLandmarks` so no third-party type
leaks into the rest of the application. Swapping in another backend only means
writing another class with a ``process(frame_bgr) -> Optional[HandLandmarks]``
method.

The ``.task`` model file is downloaded once to a local cache on first use.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .landmarks import NUM_LANDMARKS, HandLandmarks

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class HandTrackerError(RuntimeError):
    pass


@dataclass
class HandTrackerConfig:
    max_hands: int = 1
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: Optional[str] = None


def _ensure_model(model_path: Optional[str]) -> str:
    if model_path:
        path = Path(model_path)
        if not path.is_file():
            raise HandTrackerError(f"hand model not found: {path}")
        return str(path)

    env_override = os.environ.get("VGC_HAND_MODEL")
    if env_override:
        return _ensure_model(env_override)

    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached = _DEFAULT_MODEL_DIR / "hand_landmarker.task"
    if not cached.is_file():
        logger.info("downloading hand landmark model to %s", cached)
        try:
            urllib.request.urlretrieve(_MODEL_URL, cached)  # noqa: S310 (trusted URL)
        except Exception as exc:  # pragma: no cover - network dependent
            raise HandTrackerError(
                f"failed to download hand model from {_MODEL_URL}"
            ) from exc
    return str(cached)


class HandTracker:
    def __init__(self, config: Optional[HandTrackerConfig] = None) -> None:
        self._config = config or HandTrackerConfig()
        self._landmarker = None
        self._mp = None

    def open(self) -> "HandTracker":
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise HandTrackerError("mediapipe is required for hand tracking") from exc

        model_path = _ensure_model(self._config.model_path)
        try:
            self._mp = mp
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=self._config.max_hands,
                min_hand_detection_confidence=self._config.min_detection_confidence,
                min_hand_presence_confidence=self._config.min_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise HandTrackerError("failed to initialize MediaPipe HandLandmarker") from exc

        logger.info("hand model loaded")
        return self

    def process(self, frame_bgr: np.ndarray) -> Optional[HandLandmarks]:
        if self._landmarker is None or self._mp is None:
            raise HandTrackerError("hand tracker is not open")

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._landmarker.detect(mp_image)
        except Exception as exc:  # pragma: no cover - runtime inference failure
            raise HandTrackerError("hand inference failed") from exc

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
        if points.shape != (NUM_LANDMARKS, 3):
            return None

        handedness = "Unknown"
        score = 0.0
        if result.handedness:
            category = result.handedness[0][0]
            handedness = category.category_name or "Unknown"
            score = float(category.score)

        return HandLandmarks(points=points, handedness=handedness, score=score)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("hand model released")

    def __enter__(self) -> "HandTracker":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
