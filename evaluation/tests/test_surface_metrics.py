"""Behavioral checks for ICON's official point-to-surface primitive."""

import unittest


class SurfaceMetricBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from pytorch3d.loss.point_mesh_distance import _PointFaceDistance  # noqa: F401
            from pytorch3d.ops import sample_points_from_meshes
            from pytorch3d.structures import Meshes, Pointclouds
            from lib.dataset.Evaluator import point_mesh_distance
        except ImportError as exc:
            raise unittest.SkipTest(
                "Run in the verified AutoDL icon environment with PyTorch3D installed"
            ) from exc

        cls.torch = torch
        cls.Meshes = Meshes
        cls.Pointclouds = Pointclouds
        cls.sample_points_from_meshes = sample_points_from_meshes
        cls.point_mesh_distance = staticmethod(point_mesh_distance)
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def tetrahedron(self, offset=(0.0, 0.0, 0.0)):
        torch = self.torch
        verts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        verts = verts + torch.tensor(offset, device=self.device)
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=torch.int64,
            device=self.device,
        )
        return self.Meshes(verts=[verts], faces=[faces])

    def disconnected_pair(self, separation=3.0):
        torch = self.torch
        first = self.tetrahedron().verts_list()[0]
        second = self.tetrahedron((separation, 0.0, 0.0)).verts_list()[0]
        verts = torch.cat([first, second], dim=0)
        faces = self.tetrahedron().faces_list()[0]
        faces = torch.cat([faces, faces + 4], dim=0)
        return self.Meshes(verts=[verts], faces=[faces])

    def sampled_cloud(self, mesh, count=2000):
        points = self.sample_points_from_meshes(mesh, count)
        return self.Pointclouds(points)

    def test_identity_has_near_zero_p2s(self):
        self.torch.manual_seed(42)
        mesh = self.tetrahedron()
        distance = self.point_mesh_distance(mesh, self.sampled_cloud(mesh))
        self.assertLess(float(distance), 1e-4)

    def test_translation_increases_p2s(self):
        self.torch.manual_seed(42)
        gt = self.tetrahedron()
        translated_prediction = self.tetrahedron((2.0, 0.0, 0.0))
        distance = self.point_mesh_distance(
            translated_prediction, self.sampled_cloud(gt)
        )
        self.assertGreater(float(distance), 0.5)

    def test_extra_component_exposes_one_directional_blind_spot(self):
        self.torch.manual_seed(42)
        gt = self.tetrahedron()
        prediction_with_extra = self.disconnected_pair()

        gt_to_prediction = self.point_mesh_distance(
            prediction_with_extra, self.sampled_cloud(gt)
        )
        prediction_to_gt = self.point_mesh_distance(
            gt, self.sampled_cloud(prediction_with_extra)
        )

        self.assertLess(float(gt_to_prediction), 1e-4)
        self.assertGreater(float(prediction_to_gt), 0.2)


if __name__ == "__main__":
    unittest.main()
