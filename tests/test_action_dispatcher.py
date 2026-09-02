"""Unit tests for gesture -> action dispatch, hold and cooldown semantics."""

from __future__ import annotations

from vision_gesture_control.action_dispatcher import Action, ActionDispatcher
from vision_gesture_control.gesture_classifier import Gesture


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _counter_action(box: list[int], name: str = "beep") -> Action:
    def run() -> None:
        box[0] += 1

    return Action(name=name, run=run)


def test_action_fires_once_while_gesture_is_held() -> None:
    box = [0]
    dispatcher = ActionDispatcher(
        {Gesture.THUMBS_UP: _counter_action(box)},
        cooldown_seconds=1.0,
        time_source=FakeClock(),
    )
    for _ in range(10):
        dispatcher.handle(Gesture.THUMBS_UP)
    assert box[0] == 1


def test_cooldown_prevents_repeated_actions() -> None:
    box = [0]
    clock = FakeClock()
    dispatcher = ActionDispatcher(
        {Gesture.THUMBS_UP: _counter_action(box)},
        cooldown_seconds=1.0,
        time_source=clock,
    )
    dispatcher.handle(Gesture.THUMBS_UP)   # fires
    dispatcher.handle(Gesture.UNKNOWN)     # gesture released -> re-armed
    dispatcher.handle(Gesture.THUMBS_UP)   # blocked: still within cooldown
    assert box[0] == 1

    clock.now = 1.5
    dispatcher.handle(Gesture.UNKNOWN)
    dispatcher.handle(Gesture.THUMBS_UP)   # cooldown elapsed -> fires again
    assert box[0] == 2


def test_switching_gesture_triggers_the_other_action() -> None:
    up, down = [0], [0]
    dispatcher = ActionDispatcher(
        {
            Gesture.THUMBS_UP: _counter_action(up),
            Gesture.THUMBS_DOWN: _counter_action(down),
        },
        cooldown_seconds=0.0,
        time_source=FakeClock(),
    )
    dispatcher.handle(Gesture.THUMBS_UP)
    dispatcher.handle(Gesture.THUMBS_DOWN)
    assert up[0] == 1
    assert down[0] == 1


def test_unmapped_and_unknown_gestures_do_nothing() -> None:
    dispatcher = ActionDispatcher({}, cooldown_seconds=0.0, time_source=FakeClock())
    assert dispatcher.handle(Gesture.OPEN_PALM) is None
    assert dispatcher.handle(Gesture.UNKNOWN) is None
    assert dispatcher.last_action_name is None
