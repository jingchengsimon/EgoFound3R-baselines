"""Keep RGB model crops while honoring Ultralytics' BGR ndarray input."""
from functools import wraps

import numpy as np


def configure_rgb_yolo_input(detector):
    original_predict = detector.predict

    @wraps(original_predict)
    def predict(source=None, *args, **kwargs):
        # Both YOLO.__call__ and YOLO.track dispatch through predict.
        bgr = np.ascontiguousarray(source[..., ::-1])
        return original_predict(bgr, *args, **kwargs)

    detector.predict = predict
    return detector
