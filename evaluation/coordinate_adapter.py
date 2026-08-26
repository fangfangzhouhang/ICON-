"""Coordinate bookkeeping for the ``apps.infer`` image-crop route.

``apps.infer`` reconstructs a mesh in the normalized coordinate frame of the
512 x 512 person crop.  CAPE ground-truth meshes used by ICON's evaluator are
expressed in the normalized frame of the complete rendered image.  Comparing
the two without undoing the crop mostly measures preprocessing, not geometry.

This module makes that conversion explicit.  It deliberately performs no ICP,
centroid alignment, or per-case best-fit scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


ADAPTER_SCHEMA_VERSION = 1


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _float(value: Any) -> float:
    array = _array(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one scalar, got shape {array.shape}")
    result = float(array[0])
    if not np.isfinite(result):
        raise ValueError(f"Expected a finite scalar, got {result}")
    return result


def serialise_uncrop_param(uncrop_param: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the preprocessing metadata as portable JSON values."""

    required = {"center", "scale", "ori_shape", "box_shape", "crop_shape", "M"}
    missing = sorted(required - set(uncrop_param))
    if missing:
        raise KeyError(f"uncrop_param is missing keys: {missing}")
    return {
        "center": _array(uncrop_param["center"]).reshape(-1).tolist(),
        "scale": _float(uncrop_param["scale"]),
        "ori_shape": [int(item) for item in _array(uncrop_param["ori_shape"]).reshape(-1)],
        "box_shape": [int(item) for item in _array(uncrop_param["box_shape"]).reshape(-1)],
        "crop_shape": [int(item) for item in _array(uncrop_param["crop_shape"]).reshape(-1)],
        "M": _array(uncrop_param["M"]).tolist(),
    }


@dataclass(frozen=True)
class CoordinateTransform:
    matrix: np.ndarray
    x_scale: float
    y_scale: float
    anisotropy_ratio: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "source_frame": "person_crop_ndc_y_up",
            "target_frame": "full_image_ndc_y_up",
            "matrix": self.matrix.tolist(),
            "x_scale": self.x_scale,
            "y_scale": self.y_scale,
            "anisotropy_ratio": self.anisotropy_ratio,
            "hidden_alignment": False,
        }


def build_crop_ndc_to_image_ndc(
    uncrop_param: Mapping[str, Any],
    *,
    max_anisotropy: Optional[float] = None,
) -> CoordinateTransform:
    """Build the homogeneous transform from person-crop NDC to image NDC.

    The crop implementation uses a square with side ``200 * scale`` in the
    1024-pixel augmented image.  ``M`` maps original image pixels into that
    augmented image.  The returned matrix composes the inverse of both steps.

    For the CAPE experiment the source renders are square, so x/y scale should
    be effectively isotropic.  ``max_anisotropy`` lets the scientific runner
    reject a malformed trace instead of silently inventing a z scale.
    """

    record = serialise_uncrop_param(uncrop_param)
    center = np.asarray(record["center"], dtype=np.float64)
    if center.size != 2:
        raise ValueError(f"center must contain x/y, got {center}")
    original_shape = record["ori_shape"]
    if len(original_shape) < 2:
        raise ValueError(f"ori_shape must contain height/width, got {original_shape}")
    image_height, image_width = float(original_shape[0]), float(original_shape[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"Invalid original image size: {original_shape}")

    augmentation = np.asarray(record["M"], dtype=np.float64)
    if augmentation.shape == (2, 3):
        augmentation = np.vstack([augmentation, [0.0, 0.0, 1.0]])
    if augmentation.shape != (3, 3):
        raise ValueError(f"M must be 2x3 or 3x3, got {augmentation.shape}")
    if not np.isfinite(augmentation).all():
        raise ValueError("M contains non-finite values")

    half_crop = 100.0 * float(record["scale"])
    if half_crop <= 0:
        raise ValueError(f"Crop scale must be positive, got {record['scale']}")

    # Crop NDC (x right, y up) -> augmented-image pixels (u right, v down).
    augmented_from_crop = np.array(
        [
            [half_crop, 0.0, center[0]],
            [0.0, -half_crop, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    original_from_augmented = np.linalg.inv(augmentation)
    image_ndc_from_original = np.array(
        [
            [2.0 / image_width, 0.0, -1.0],
            [0.0, -2.0 / image_height, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    xy = image_ndc_from_original @ original_from_augmented @ augmented_from_crop

    x_scale = float(np.linalg.norm(xy[:2, 0]))
    y_scale = float(np.linalg.norm(xy[:2, 1]))
    if x_scale <= 0 or y_scale <= 0:
        raise ValueError(f"Degenerate crop transform: x_scale={x_scale}, y_scale={y_scale}")
    anisotropy = max(x_scale, y_scale) / min(x_scale, y_scale)
    if max_anisotropy is not None and anisotropy > max_anisotropy:
        raise ValueError(
            "Crop-to-image transform is too anisotropic for metric z scaling: "
            f"ratio={anisotropy:.8f}, allowed={max_anisotropy:.8f}"
        )

    # Orthographic ICON geometry uses the same scale for depth and image-plane
    # coordinates.  CAPE's square renders make these two estimates agree.
    z_scale = (x_scale + y_scale) * 0.5
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = xy[0, 0]
    matrix[0, 1] = xy[0, 1]
    matrix[0, 3] = xy[0, 2]
    matrix[1, 0] = xy[1, 0]
    matrix[1, 1] = xy[1, 1]
    matrix[1, 3] = xy[1, 2]
    matrix[2, 2] = z_scale
    return CoordinateTransform(matrix, x_scale, y_scale, anisotropy)


def apply_homogeneous_transform(vertices: Sequence[Sequence[float]], matrix: Any) -> np.ndarray:
    """Apply a 4x4 transform to an N x 3 vertex array."""

    points = np.asarray(vertices, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {transform.shape}")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    mapped = homogeneous @ transform.T
    if np.any(np.isclose(mapped[:, 3], 0.0)):
        raise ValueError("Transform produced a zero homogeneous coordinate")
    result = mapped[:, :3] / mapped[:, 3:4]
    if not np.isfinite(result).all():
        raise ValueError("Transform produced non-finite vertices")
    return result
