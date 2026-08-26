import unittest

import numpy as np

from evaluation.coordinate_adapter import (
    apply_homogeneous_transform,
    build_crop_ndc_to_image_ndc,
)


class CoordinateAdapterTest(unittest.TestCase):
    def trace(self, *, shape=(1024, 1024, 3), center=(512, 512), scale=5.12):
        return {
            "center": np.asarray(center),
            "scale": scale,
            "ori_shape": shape,
            "box_shape": [1024, 1024],
            "crop_shape": [512, 512],
            "M": np.eye(3),
        }

    def test_full_image_crop_is_identity(self):
        transform = build_crop_ndc_to_image_ndc(self.trace())
        self.assertTrue(np.allclose(transform.matrix, np.eye(4)))
        points = np.asarray([[0.0, 0.0, 0.2], [1.0, -1.0, -0.5]])
        self.assertTrue(
            np.allclose(apply_homogeneous_transform(points, transform.matrix), points)
        )

    def test_half_size_centered_crop_scales_geometry(self):
        transform = build_crop_ndc_to_image_ndc(self.trace(scale=2.56))
        expected = np.diag([0.5, 0.5, 0.5, 1.0])
        self.assertTrue(np.allclose(transform.matrix, expected))

    def test_cape_512_image_resized_to_1024_round_trips_to_identity(self):
        trace = self.trace(
            shape=(512, 512, 3),
            center=(512, 512),
            scale=5.12,
        )
        trace["M"] = np.diag([2.0, 2.0, 1.0])
        transform = build_crop_ndc_to_image_ndc(trace)
        self.assertTrue(np.allclose(transform.matrix, np.eye(4)))

    def test_off_center_crop_keeps_scale_and_restores_translation(self):
        trace = self.trace(
            shape=(512, 512, 3),
            center=(768, 512),
            scale=2.56,
        )
        trace["M"] = np.diag([2.0, 2.0, 1.0])
        transform = build_crop_ndc_to_image_ndc(trace)
        expected = np.asarray(
            [
                [0.5, 0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertTrue(np.allclose(transform.matrix, expected))

    def test_anisotropic_trace_can_be_rejected(self):
        trace = self.trace(shape=(512, 1024, 3))
        with self.assertRaisesRegex(ValueError, "anisotropic"):
            build_crop_ndc_to_image_ndc(trace, max_anisotropy=1.001)


if __name__ == "__main__":
    unittest.main()
