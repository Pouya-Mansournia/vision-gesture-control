"""Webcam abstraction over OpenCV ``VideoCapture``.

Keeping capture behind this class means the rest of the pipeline depends on
"something that yields BGR frames", which a ROS 2 image subscriber could later
provide instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30


class Camera:
    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self._config = config or CameraConfig()
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> "Camera":
        capture = cv2.VideoCapture(self._config.index, cv2.CAP_ANY)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"could not open camera at index {self._config.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)
        self._capture = capture
        logger.info("camera opened (index=%d)", self._config.index)
        return self

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("failed to read a frame from the camera")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("camera released")

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
