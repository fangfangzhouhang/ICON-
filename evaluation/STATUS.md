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
- The AutoDL `icon` environment ran all three PyTorch3D metric behavior tests
  successfully: identity, translation, and disconnected-extra-component.
- AutoDL evidence log:
  `/root/autodl-tmp/icon-repro/run-record/evaluation-metric-sanity-rerun2.log`.

## Not yet confirmed

- CAPE terms have not been accepted and CAPE ground-truth data are absent.
- No CAPE Chamfer, P2S, or normal-error value has been produced.
- No valid trained-feature ablation has been run.

## Current blocker and next gate

The metric-behavior gate has passed. The next blocker is the separate CAPE
user-license gate. After the user accepts the official terms, audit the
extracted CAPE directory before downloading anything, then run a two-sample
easy/hard pilot before a full benchmark.
