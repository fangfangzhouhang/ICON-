# Dataset manifests

Formal benchmark runs should use a versioned manifest instead of discovering
files ad hoc. Each row should identify at least:

```text
sample_id,subset,input_path,pred_path,gt_path,smpl_path,license_source
```

Do not add licensed CAPE meshes or credentials to Git. Only relative paths and
non-sensitive metadata belong in a manifest.
