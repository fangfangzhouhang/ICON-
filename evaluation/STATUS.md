# Evaluation status

Updated: 2026-08-26

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
- The one-time official `apps.train -test` aggregate and the custom 450-case
  evaluator agree within the declared engineering tolerances for easy/hard
  Chamfer, P2S, and normal error. This validates the measurement pipeline; it
  does not by itself reproduce the paper table.
- The paired prior-source experiment is implemented locally: CAPE-prepared SMPL
  versus PIXIE-estimated SMPL-X from the identical rendered image. Its contract
  is `evaluation/prior_source_protocol.yaml`, its runner is
  `evaluation/run_prior_source_ab.py`, and its analysis entry point is
  `evaluation/summarize_prior_source_ab.py`.
- The coordinate adapter follows the real `apps.infer` preprocessing chain
  (original image -> 1024 square -> person crop) and has regression tests for
  centred and off-centre CAPE inputs.

## Not yet confirmed

- Why this checkout's descriptive 450-case scores should not be quoted as the
  published ICON table without matching every remaining paper protocol detail
  and every baseline under the same evaluation contract.
- The coordinate mapping has unit-test coverage but has not yet passed the
  required two-case visual and numerical gate on AutoDL. Therefore no
  end-to-end conclusion about PIXIE versus prepared CAPE priors is valid yet.
- No 30-case paired prior-source pilot or 450-case paired prior-source run has
  been executed.
- No valid trained-feature ablation has been run.

## Current blocker and next gate

Run only one easy and one hard image through the new paired runner. Inspect the
mapped PIXIE-route reconstruction, estimated SMPL-X, image silhouette, and CAPE
ground truth before accepting any number. If both cases share the same frame and
the run records contain no stale artifacts, expand first to a balanced 30-case
pilot. Do not launch the 450-case paired run before that gate passes.
