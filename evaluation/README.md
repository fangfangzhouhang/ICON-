# ICON evaluation: first validate the ruler

This directory adds an auditable evaluation layer without changing ICON's
network, checkpoints, inference entry point, or licensed assets.

## Research question before code

An experiment is only interpretable when these six items are fixed:

1. **Hypothesis**: for example, an SMPL-conditioned method is more robust to
   non-standard poses than an image-only implicit method.
2. **Independent variable**: the reconstruction method or a separately trained
   model variant.
3. **Dependent variables**: Chamfer, P2S, normal error, runtime, memory, and
   failure count.
4. **Controls**: identical test subjects, images, calibration, scale, mesh
   cleaning, sample count, and metric implementation.
5. **Sanity checks**: prove that the metric behaves correctly before trusting
   benchmark numbers.
6. **Validity boundary**: do not report CAPE benchmark results without the
   licensed CAPE test set and ground-truth meshes.

## What is reused from the official repository

`evaluation.metrics.official_icon` calls the existing definitions in
`lib/dataset/Evaluator.py`:

- P2S: ground-truth surface samples to the predicted triangle surface;
- Chamfer: the mean of the two directional point-to-surface distances;
- normal error: squared difference between normal renders from four views.

The wrapper records the sample count, random seed, hashes, and Git commit. It
does not silently align or rescale meshes. Prediction and ground truth must
already be in the same coordinate frame expected by ICON.

## Step 1: sanity-check the distance metrics on AutoDL

```bash
conda activate icon
cd /root/autodl-tmp/icon-repro/ICON
python -m unittest evaluation.tests.test_surface_metrics -v
```

Expected conclusions:

- a mesh compared with itself has near-zero P2S;
- translation increases the distance;
- adding a disconnected extra component can leave one-directional P2S near
  zero while increasing the reverse distance and Chamfer.

These are tests of the *measurement tool*, not model-performance results.

## Step 2: evaluate one already-aligned pair

```bash
python -m evaluation.evaluate_mesh \
  --pred /path/to/prediction.obj \
  --gt /path/to/ground_truth.obj \
  --sample-id subject-001 \
  --method icon-filter \
  --output-dir ./evaluation/outputs/subject-001 \
  --device cuda:0 \
  --num-samples 1000 \
  --seed 42
```

Outputs:

- `metrics.json`: values plus the full evaluation contract;
- `metrics.csv`: one-row table suitable for later aggregation;
- `normal_comparison.png`: official four-view normal-render comparison.

Use `--skip-normal` only while debugging surface-distance code. A benchmark row
without normal error must be marked incomplete.

## Step 3: official CAPE benchmark

The generic pair evaluator above is not a replacement for ICON's dataset-aware
test path. Once the licensed CAPE files and official checkpoints are present,
use the repository's official route:

```bash
python -m apps.train -cfg ./configs/train/icon-filter.yaml -test
```

Run the PIFu, PaMIR, ICON-no-filter, and ICON-filter configurations with the
same CAPE subset and the same effective Marching Cubes resolution. Record
failures, runtime, and peak GPU memory in addition to the three geometry
metrics.

## What does not count as a valid ablation

Zeroing or deleting `sdf`, `cmap`, `norm`, or `vis` only at inference time is a
sensitivity/intervention test. It is not a paper-quality ablation because the
network was trained with those inputs. A valid feature ablation needs a
separately trained checkpoint under controlled training conditions.
