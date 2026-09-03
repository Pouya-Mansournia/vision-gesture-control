"""Geometric (landmark-based) gesture classification.

The classifier is a pure function of hand landmarks. It performs **no** I/O and
triggers **no** actions, which keeps it deterministic and unit-testable with
synthetic coordinates.

Operating assumptions (documented on purpose):

* One or two hands, roughly facing the camera, fingers pointing broadly up.
* Hands are not heavily rotated in-plane (no full rotation invariance).
* All distances are normalized by the wrist -> middle-finger-MCP length of the
  hand, so the logic is independent of frame resolution and camera distance.

Single-hand gestures: ``THUMBS_UP``, ``THUMBS_DOWN``, ``OPEN_PALM``, ``FIST``,
``INDEX_UP``, ``PEACE``, ``MIDDLE_FINGER``.
Two-hand gestures: ``TWO_HAND_HEART`` (thumbs and index fingers of both hands
meeting to outline a heart).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np

from .landmarks import HandLandmarks
from .landmarks import LandmarkIndex as LM

_EPS = 1e-9

_FINGERS: Dict[str, tuple[LM, LM, LM]] = {
    # name: (tip, pip, mcp)
    "index": (LM.INDEX_TIP, LM.INDEX_PIP, LM.INDEX_MCP),
    "middle": (LM.MIDDLE_TIP, LM.MIDDLE_PIP, LM.MIDDLE_MCP),
    "ring": (LM.RING_TIP, LM.RING_PIP, LM.RING_MCP),
    "pinky": (LM.PINKY_TIP, LM.PINKY_PIP, LM.PINKY_MCP),
}


class Gesture(str, Enum):
    """Every gesture the architecture supports."""

    UNKNOWN = "UNKNOWN"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    INDEX_UP = "INDEX_UP"
    INDEX_DOWN = "INDEX_DOWN"
    PEACE = "PEACE"
    OK = "OK"
    MIDDLE_FINGER = "MIDDLE_FINGER"
    TWO_HAND_HEART = "TWO_HAND_HEART"


@dataclass(frozen=True)
class ClassificationResult:
    gesture: Gesture
    confidence: float


_UNKNOWN = ClassificationResult(Gesture.UNKNOWN, 0.0)


@dataclass(frozen=True)
class _HandGeometry:
    """Derived, resolution-independent facts about one hand."""

    scale: float
    wrist: np.ndarray
    thumb_tip: np.ndarray
    index_tip: np.ndarray
    thumb_extended: bool
    thumb_dy: float  # +y points down in image space
    thumb_dx: float
    fingers_extended: Dict[str, bool]
    fingers_folded: Dict[str, bool]
    fingers_raised: Dict[str, bool]  # straight (tip past pip past mcp), any length

    @property
    def extended_count(self) -> int:
        return sum(self.fingers_extended.values())

    @property
    def all_extended(self) -> bool:
        return all(self.fingers_extended.values())

    @property
    def all_folded(self) -> bool:
        return all(self.fingers_folded.values())


@dataclass(frozen=True)
class GestureClassifier:
    """Classify one hand, or a scene of hands, into a :class:`Gesture`.

    Parameters
    ----------
    thumb_extension_threshold:
        Minimum thumb length (tip -> CMC) over hand scale to count as extended.
    finger_fold_threshold:
        A finger is "folded" when its tip has curled back so
        ``dist(tip, wrist) < dist(pip, wrist)``, or, as a fallback for a tight
        curl, ``dist(tip, wrist) < k * dist(mcp, wrist)``.
    finger_extension_threshold:
        A finger is "extended" when the tip reaches past its pip and
        ``dist(tip, wrist) > k * dist(mcp, wrist)``. Partly bent fingers are
        neither folded nor extended.
    thumb_vertical_ratio:
        Vertical thumb component must exceed this multiple of the horizontal one
        for the thumb to be considered up or down.
    heart_thumb_join_ratio / heart_index_join_ratio:
        Max distance (over average hand scale) between the two thumbs / two index
        tips for a two-hand heart.
    heart_vertical_gap_ratio:
        The thumbs' midpoint must sit below the index tips' midpoint by at least
        this fraction of the average hand scale.
    """

    thumb_extension_threshold: float = 0.6
    finger_fold_threshold: float = 1.1
    finger_extension_threshold: float = 1.5
    thumb_vertical_ratio: float = 1.2
    heart_thumb_join_ratio: float = 1.6
    heart_index_join_ratio: float = 2.4
    heart_vertical_gap_ratio: float = 0.2

    # ----------------------------------------------------------------- public

    def classify(self, hand: HandLandmarks) -> ClassificationResult:
        """Classify a single hand."""
        geo = self._geometry(hand)
        if geo is None:
            return _UNKNOWN
        return self._classify_single(geo)

    def classify_scene(
        self, hands: Sequence[Optional[HandLandmarks]]
    ) -> ClassificationResult:
        """Classify a frame that may contain zero, one, or more hands.

        Two-hand gestures win when present; otherwise the first hand is used.
        """
        present = [h for h in hands if h is not None]
        if not present:
            return _UNKNOWN

        if len(present) >= 2:
            heart = self._classify_heart(present[0], present[1])
            if heart.gesture is not Gesture.UNKNOWN:
                return heart

        return self.classify(present[0])

    # ---------------------------------------------------------------- geometry

    def _geometry(self, hand: HandLandmarks) -> Optional[_HandGeometry]:
        pts = hand.points
        wrist = pts[LM.WRIST][:2]
        scale = float(np.linalg.norm(pts[LM.MIDDLE_MCP][:2] - wrist))
        if scale < 1e-4:
            return None

        thumb_tip = pts[LM.THUMB_TIP][:2]
        thumb_len_ratio = float(np.linalg.norm(thumb_tip - pts[LM.THUMB_CMC][:2])) / scale
        direction = thumb_tip - pts[LM.THUMB_MCP][:2]

        extended: Dict[str, bool] = {}
        folded: Dict[str, bool] = {}
        raised: Dict[str, bool] = {}
        for name, (tip, pip, mcp) in _FINGERS.items():
            tip_w = float(np.linalg.norm(pts[tip][:2] - wrist))
            pip_w = float(np.linalg.norm(pts[pip][:2] - wrist))
            mcp_w = float(np.linalg.norm(pts[mcp][:2] - wrist))
            # "curled back past its own middle joint" is a scale-free fold cue;
            # the mcp ratio is a fallback for a very tight curl.
            folded[name] = (
                tip_w < pip_w or tip_w < self.finger_fold_threshold * mcp_w
            )
            # "raised" = the joints step outward from the wrist, so the finger
            # is straight regardless of how long it measures. "extended" adds a
            # length check and is what the strict all-fingers poses use.
            raised[name] = tip_w > pip_w > mcp_w
            extended[name] = (
                raised[name] and tip_w > self.finger_extension_threshold * mcp_w
            )

        return _HandGeometry(
            scale=scale,
            wrist=wrist,
            thumb_tip=thumb_tip,
            index_tip=pts[LM.INDEX_TIP][:2],
            thumb_extended=thumb_len_ratio >= self.thumb_extension_threshold,
            thumb_dy=float(direction[1]),
            thumb_dx=abs(float(direction[0])),
            fingers_extended=extended,
            fingers_folded=folded,
            fingers_raised=raised,
        )

    # ------------------------------------------------------------ single hand

    def _classify_single(self, geo: _HandGeometry) -> ClassificationResult:
        raised = geo.fingers_raised
        thumb_vertical = abs(geo.thumb_dy) >= self.thumb_vertical_ratio * geo.thumb_dx

        if geo.thumb_extended and geo.all_folded and thumb_vertical:
            if geo.thumb_dy < 0 and geo.thumb_tip[1] < geo.wrist[1]:
                return ClassificationResult(Gesture.THUMBS_UP, 0.9)
            if geo.thumb_dy > 0 and geo.thumb_tip[1] > geo.wrist[1]:
                return ClassificationResult(Gesture.THUMBS_DOWN, 0.9)

        if geo.all_extended and geo.thumb_extended:
            return ClassificationResult(Gesture.OPEN_PALM, 0.85)

        if geo.all_folded and not geo.thumb_extended:
            return ClassificationResult(Gesture.FIST, 0.85)

        # For the "one or two fingers up" poses the named fingers only need to
        # be straight (raised), and the others only need to be "not raised"
        # (a loose curl counts), which is what a real hand does.
        down = {name: not raised[name] for name in raised}

        if raised["index"] and raised["middle"] and down["ring"] and down["pinky"]:
            return ClassificationResult(Gesture.PEACE, 0.8)

        if (
            raised["index"]
            and down["middle"]
            and down["ring"]
            and down["pinky"]
            and not geo.thumb_extended
        ):
            return ClassificationResult(Gesture.INDEX_UP, 0.8)

        if raised["middle"] and down["index"] and down["ring"] and down["pinky"]:
            return ClassificationResult(Gesture.MIDDLE_FINGER, 0.8)

        return _UNKNOWN

    # -------------------------------------------------------------- two hands

    def _classify_heart(
        self, hand_a: HandLandmarks, hand_b: HandLandmarks
    ) -> ClassificationResult:
        a = self._geometry(hand_a)
        b = self._geometry(hand_b)
        if a is None or b is None:
            return _UNKNOWN

        avg_scale = 0.5 * (a.scale + b.scale)
        thumb_gap = float(np.linalg.norm(a.thumb_tip - b.thumb_tip))
        index_gap = float(np.linalg.norm(a.index_tip - b.index_tip))
        thumb_mid_y = 0.5 * (a.thumb_tip[1] + b.thumb_tip[1])
        index_mid_y = 0.5 * (a.index_tip[1] + b.index_tip[1])

        if not (a.fingers_extended["index"] and b.fingers_extended["index"]):
            return _UNKNOWN
        if not (a.thumb_extended and b.thumb_extended):
            return _UNKNOWN
        if thumb_gap > self.heart_thumb_join_ratio * avg_scale:
            return _UNKNOWN
        if index_gap > self.heart_index_join_ratio * avg_scale:
            return _UNKNOWN
        if (thumb_mid_y - index_mid_y) < self.heart_vertical_gap_ratio * avg_scale:
            return _UNKNOWN

        closeness = _saturate(
            1.0 - thumb_gap / (self.heart_thumb_join_ratio * avg_scale + _EPS)
        )
        return ClassificationResult(Gesture.TWO_HAND_HEART, float(0.6 + 0.4 * closeness))


def _saturate(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
