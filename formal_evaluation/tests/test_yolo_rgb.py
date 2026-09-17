import numpy as np
from formal_evaluation.hand.adapters.yolo_rgb import configure_rgb_yolo_input


def test_detection_and_tracking_convert_once_without_mutating_crop_rgb():
    class Detector:
        def predict(self, source=None, **kwargs):
            assert source.flags.c_contiguous
            return source, kwargs

        def __call__(self, source, **kwargs):
            return self.predict(source, **kwargs)

        def track(self, source, **kwargs):
            return self.predict(source=source, **kwargs)

    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    original = rgb.copy()
    detector = configure_rgb_yolo_input(Detector())
    for call in [detector, detector.track]:
        bgr, kwargs = call(rgb, conf=0.3)
        np.testing.assert_array_equal(bgr, original[..., ::-1])
        np.testing.assert_array_equal(rgb, original)
        assert not np.shares_memory(rgb, bgr)
        assert kwargs == {"conf": 0.3}
