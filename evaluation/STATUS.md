# Evaluation status

Updated: 2026-08-24

## Confirmed

- Fork remote: `https://github.com/fangfangzhouhang/ICON-.git`
- Upstream remote: `https://github.com/YuliangXiu/ICON.git`
- Experiment branch: `evaluation-exp`
- Baseline commit: `ba2ee5681284c3d627305b1c919f4414009f753b`
- The evaluation module does not modify ICON networks, checkpoints, or inference.
- Python 3.8 syntax check passes for all evaluation Python files.
- The command-line interface can be parsed locally.

## Not yet confirmed

- The first AutoDL run reached all three PyTorch3D tests, but the test wrapper
  incorrectly bound `sample_points_from_meshes` as an instance method. Commit
  after `84e46fc` fixes that harness bug; all three tests still require a rerun.
- CAPE terms have not been accepted and CAPE ground-truth data are absent.
- No CAPE Chamfer, P2S, or normal-error value has been produced.
- No valid trained-feature ablation has been run.

## Current blocker and next gate

First run `python -m unittest evaluation.tests.test_surface_metrics -v` in the
verified AutoDL `icon` environment. Only after all three tests pass should the
project accept metric output as trustworthy. CAPE download is a later,
separate user-license gate.
