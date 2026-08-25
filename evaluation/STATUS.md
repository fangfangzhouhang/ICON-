# Evaluation status

Updated: 2026-08-25

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
- The licensed CAPE evaluation subset is present on AutoDL: 150 subjects, three
  rotations per subject, with scans, prepared SMPL parameters, renders, masks,
  calibration, normals, and visibility artifacts.
- The resumable ICON-filter CAPE run completed 450/450 cases with zero recorded
  inference failures. Its report is descriptive for this checkout and must not
  yet be labelled a paper-table reproduction.
- The next experiment is a paired prior-source comparison: CAPE-prepared SMPL
  versus PIXIE-estimated SMPL-X from the identical rendered image. Its contract
  is `evaluation/prior_source_protocol.yaml`.

## Not yet confirmed

- Why the current 450-case descriptive scores differ substantially from the
  published ICON table. The official `apps.train -test` parity run is the
  one-time runner/aggregation calibration for this question.
- The coordinate mapping from the `apps.infer` PIXIE/SMPL-X frame back to the
  CAPE metric frame. No end-to-end PIXIE-prior metric is valid until this mapping
  passes the two-case coordinate gate.
- No valid trained-feature ablation has been run.

## Current blocker and next gate

Finish the one-time official runner parity audit, then implement and test the
coordinate adapter for the paired prior-source experiment. Run only one easy
and one hard image through `apps.infer` until the mapped prediction, estimated
SMPL-X, image silhouette, and CAPE ground truth visibly share the same frame.
