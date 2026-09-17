import numpy as np

from ego_bracketed_fill import fill_camera_space


def main():
    rng = np.random.default_rng(7)
    joints = rng.normal(size=(7, 2, 21, 3))
    markers = rng.normal(size=(7, 2, 195, 3))
    valid = np.ones((7, 2), dtype=bool)
    valid[2:5, 0] = False
    valid[3, 1] = False
    original_joints = joints.copy()
    original_markers = markers.copy()
    out_joints, out_markers, out_valid, info = fill_camera_space(joints, markers, valid)
    assert out_valid.all()
    assert info["filled_counts_left_right"] == [3, 1]
    assert np.array_equal(out_joints[valid], original_joints[valid])
    assert np.array_equal(out_markers[valid], original_markers[valid])
    assert np.isfinite(out_joints).all() and np.isfinite(out_markers).all()
    bad = valid.copy(); bad[0, 0] = False
    try:
        fill_camera_space(joints, markers, bad)
    except ValueError as error:
        assert "first/last" in str(error)
    else:
        raise AssertionError("endpoint extrapolation was not rejected")
    print("ok")


if __name__ == "__main__":
    main()
