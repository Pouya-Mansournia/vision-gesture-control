"""vision-gesture-control: real-time webcam hand-gesture recognition.

The public surface intentionally exposes only the pure, dependency-light logic
(gesture classification and temporal stabilization) so that it can be imported
without pulling in OpenCV or MediaPipe.
"""

from .gesture_classifier import ClassificationResult, Gesture, GestureClassifier
from .gesture_stabilizer import GestureStabilizer

__version__ = "0.1.0"

__all__ = [
    "Gesture",
    "GestureClassifier",
    "ClassificationResult",
    "GestureStabilizer",
    "__version__",
]
