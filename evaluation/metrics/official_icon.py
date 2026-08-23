"""Use ICON's official metric code on an already-aligned mesh pair.

This module deliberately does not perform ICP, translation, rotation, or scale
normalization. Such operations change the scientific question and must be part
of a dataset-specific, explicitly documented protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def evaluate_aligned_meshes(
    pred_path: Path,
    gt_path: Path,
    normal_output_path: Path,
    *,
    device: str = "cuda:0",
    num_samples: int = 1000,
    seed: int = 42,
    skip_normal: bool = False,
) -> Dict[str, Optional[float]]:
    """Evaluate meshes already expressed in the same ICON coordinate frame.

    The function reuses ``lib.dataset.Evaluator`` instead of introducing a new
    Chamfer/P2S convention. Bypassing ``Evaluator.set_mesh`` is intentional:
    that method performs CAPE-specific projection and reconstruction-grid
    normalization. This generic entry point accepts meshes for which those
    dataset transforms have already been applied.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    normal_output_path = Path(normal_output_path)

    for label, path in (("prediction", pred_path), ("ground truth", gt_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} mesh does not exist: {path}")

    try:
        import torch
        from pytorch3d.io import IO
        from lib.dataset.Evaluator import Evaluator
    except ImportError as exc:
        raise RuntimeError(
            "ICON evaluation dependencies are unavailable. Run this command in "
            "the verified AutoDL 'icon' environment with PyTorch3D installed."
        ) from exc

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    io = IO()
    pred_mesh = io.load_mesh(str(pred_path), device=torch_device)
    gt_mesh = io.load_mesh(str(gt_path), device=torch_device)

    evaluator = Evaluator(torch_device)
    evaluator.src_mesh = pred_mesh
    evaluator.tgt_mesh = gt_mesh

    chamfer, p2s = evaluator.calculate_chamfer_p2s(num_samples=num_samples)

    normal_error = None
    if not skip_normal:
        normal_output_path.parent.mkdir(parents=True, exist_ok=True)
        normal_error = evaluator.calculate_normal_consist(str(normal_output_path))

    return {
        "chamfer_cm": _as_float(chamfer),
        "p2s_cm": _as_float(p2s),
        "normal_error": None if normal_error is None else _as_float(normal_error),
    }
