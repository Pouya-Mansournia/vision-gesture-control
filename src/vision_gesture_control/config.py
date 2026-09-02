"""Central configuration. No magic numbers scattered across modules."""

from __future__ import annotations

from dataclasses import dataclass, field

from .camera import CameraConfig
from .hand_tracker import HandTrackerConfig


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)

    # --- Temporal validation ------------------------------------------------
    gesture_confirmation_frames: int = 5
    gesture_window_size: int = 12

    # --- Action behaviour -------------------------------------------------
    action_cooldown_seconds: float = 1.0

    # --- Gesture geometry thresholds -----------------------------------
    thumb_extension_threshold: float = 0.6
    finger_fold_threshold: float = 1.1
    thumb_vertical_ratio: float = 1.2

    # --- Beep parameters -----------------------------------------------
    thumbs_up_frequency_hz: int = 1400
    thumbs_up_duration_ms: int = 120
    thumbs_down_frequency_hz: int = 400
    thumbs_down_duration_ms: int = 350

    # --- Visualization ----------------------------------------------------
    draw_landmarks: bool = True
    show_fps: bool = True
    mirror_preview: bool = True
    window_name: str = "vision-gesture-control"


DEFAULT_CONFIG = AppConfig()
