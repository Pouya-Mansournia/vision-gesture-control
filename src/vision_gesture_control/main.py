"""Application entry point: wire the pipeline and run the real-time loop.

Pipeline per frame::

    Camera -> (mirror) -> HandTracker -> GestureClassifier
           -> GestureStabilizer -> ActionDispatcher -> visualization
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from typing import Deque, Optional

import cv2

from .action_dispatcher import ActionDispatcher, build_gesture_dispatcher
from .camera import Camera, CameraError
from .config import DEFAULT_CONFIG, AppConfig
from .gesture_classifier import GestureClassifier
from .gesture_stabilizer import GestureStabilizer
from .hand_tracker import HandTracker, HandTrackerError
from .visualization import draw_landmarks, draw_overlay

logger = logging.getLogger(__name__)

_EXIT_KEYS = {ord("q"), ord("Q"), 27}  # Q / q / ESC


class FpsMeter:
    """Lightweight rolling FPS estimate based on ``time.perf_counter``."""

    def __init__(self, window: int = 30) -> None:
        self._stamps: Deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._stamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


def build_dispatcher(config: AppConfig) -> ActionDispatcher:
    return build_gesture_dispatcher(cooldown_seconds=config.action_cooldown_seconds)


def run(config: Optional[AppConfig] = None) -> None:
    config = config or DEFAULT_CONFIG
    logger.info("application started")

    classifier = GestureClassifier(
        thumb_extension_threshold=config.thumb_extension_threshold,
        finger_fold_threshold=config.finger_fold_threshold,
        thumb_vertical_ratio=config.thumb_vertical_ratio,
    )
    stabilizer = GestureStabilizer(
        confirmation_frames=config.gesture_confirmation_frames,
        window_size=config.gesture_window_size,
    )
    dispatcher = build_dispatcher(config)
    fps_meter = FpsMeter()

    try:
        with Camera(config.camera) as camera, HandTracker(config.hand_tracker) as tracker:
            while True:
                frame = camera.read()
                if config.mirror_preview:
                    frame = cv2.flip(frame, 1)

                hands = tracker.process(frame)
                raw_gesture = classifier.classify_scene(hands).gesture

                confirmed = stabilizer.update(raw_gesture)
                dispatcher.handle(confirmed)

                if config.draw_landmarks:
                    for hand in hands:
                        draw_landmarks(frame, hand)

                fps_meter.tick()
                if config.show_fps:
                    draw_overlay(
                        frame,
                        raw_gesture=raw_gesture,
                        confirmed_gesture=confirmed,
                        fps=fps_meter.fps,
                        action_status=dispatcher.last_action_name or "-",
                        on_cooldown=dispatcher.on_cooldown(),
                    )

                cv2.imshow(config.window_name, frame)
                if (cv2.waitKey(1) & 0xFF) in _EXIT_KEYS:
                    logger.info("exit requested by user")
                    break
    except CameraError as exc:
        logger.error("camera error: %s", exc)
        raise SystemExit(1) from exc
    except HandTrackerError as exc:
        logger.error("hand tracker error: %s", exc)
        raise SystemExit(1) from exc
    finally:
        cv2.destroyAllWindows()
        logger.info("application stopped")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam hand-gesture control (thumbs up / thumbs down)."
    )
    parser.add_argument("--camera-index", type=int, default=None,
                        help="override the webcam index (default: 0)")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="do not draw the hand skeleton")
    parser.add_argument("--log-level", default="INFO",
                        help="Python logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    if args.camera_index is not None:
        config.camera.index = args.camera_index
    if args.no_landmarks:
        config.draw_landmarks = False

    run(config)


if __name__ == "__main__":
    main()
