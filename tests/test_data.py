from __future__ import annotations

import unittest

import numpy as np


from vdroad.data import rotate_xy


class DiscreteRotationTests(unittest.TestCase):
    def test_four_rotations_match_np_rot90_pixel_coordinates(self) -> None:
        points = np.asarray([[0, 0], [255, 0], [0, 255], [255, 255], [37, 149]], dtype=np.float32)
        expected = points.copy()
        for rotation in range(4):
            actual = rotate_xy(points, rotation)
            if rotation == 0:
                expected = points
            else:
                expected = np.stack([expected[:, 1], 255.0 - expected[:, 0]], axis=1)
            self.assertTrue(np.array_equal(actual, expected), (rotation, actual, expected))

    def test_inverse_rotation_closes_exactly(self) -> None:
        points = np.asarray([[12.25, 90.75], [127.5, 127.5], [240.0, 5.0]], dtype=np.float32)
        for rotation in range(4):
            restored = rotate_xy(rotate_xy(points, rotation), (4 - rotation) % 4)
            self.assertTrue(np.allclose(restored, points, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
