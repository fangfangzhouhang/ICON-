# ICON 身体先验来源 A/B 实验

## 1. 这次实验到底在问什么

同一张 CAPE 人体图、同一个 ICON 权重、同一份真实三维网格，比较两条路线：

1. **Prepared 路线**：直接使用 CAPE 已准备并对齐好的 SMPL 身体先验；
2. **PIXIE 路线**：只把图像交给官方 `apps.infer`，由 PIXIE 从图像估计 SMPL-X，
   再完成 ICON 的身体反馈和隐式重建。

三项几何指标都越低越好。每一例计算：

`差值 = PIXIE 路线 - Prepared 路线`

差值为正，表示在这一例上，从图像估计身体先验的端到端路线更差；差值为负，
表示它更好。必须先逐例做差，再汇总，不能只比较两组平均数。

## 2. 它能说明什么，不能说明什么

它能测量“标准化、已对齐身体先验”和“真实单图使用场景”之间的端到端性能差距。

它**不是**纯粹只改变 PIXIE 的因果消融，因为两条路线还同时存在这些差异：

- Prepared 路线使用 SMPL，PIXIE 路线使用 SMPL-X；
- 两条路线的图像裁剪和身体反馈路径不同；
- Prepared 路线已经获得数据集准备好的对齐信息。

因此汇报时应称为“先验来源的端到端 A/B 扩展实验”，不能称为“证明 PIXIE
造成了全部误差”，也不能冒充论文表格复现。

## 3. 公平性怎么保证

- 同一个 subject、rotation 和 CAPE 渲染图；
- 同一个 ICON 与 normal checkpoint；
- 同一个 Marching Cubes 分辨率；
- 同一个真实 CAPE Mesh；
- 同一个指标实现、表面采样数和逐例随机种子；
- 不运行 ICP、不做质心对齐、不做逐例最佳缩放；
- PIXIE 输出必须根据真实裁剪记录映射回完整图像坐标，并保存变换矩阵。

`apps.infer --stop-after-recon` 只跳过与本实验无关的重网格、衣服细化和视频导出。
它不改变网络、权重、SMPL 反馈优化或核心 `_recon.obj`。

## 4. 为什么先跑 2 例

2 例不是为了得出性能结论，而是验证实验尺子没有问题：

- 一例来自 easy，一例来自 hard；
- 两条路线都要生成 Mesh；
- 映射后的 PIXIE Mesh 要与 CAPE 人体在同一位置和尺度；
- 三项指标必须是有限数值；
- 两组 normal comparison 要和肉眼观察一致；
- `paired_summary.csv` 必须保留失败样本，不能只留下成功样本。

只有这个坐标门通过，才扩展到 30 例。30 例用于发现失败模式；完整 450 例才用于
稳定统计。若 2 例就发生尺度或方向错误，扩大到 450 例只会批量制造错误数字。

## 5. AutoDL 分步命令

以下命令假设评测 worktree、原始 ICON 数据和现有 450 例结果仍位于之前的路径。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate icon
cd /root/autodl-tmp/icon-repro/ICON-eval-7763b6c
```

先检查入口和权重，不运行模型：

```bash
python -m evaluation.run_prior_source_ab --help
```

准备两张确定的 CAPE 输入图：

```bash
python -m evaluation.run_prior_source_ab \
  --mode prepare \
  --subject-indices 0 50 \
  --rotations 0 \
  --output-dir evaluation/outputs/prior-source-ab-gate
```

让官方单图入口用 PIXIE 估计身体先验并生成核心 Mesh：

```bash
python -m evaluation.run_prior_source_ab \
  --mode infer \
  --subject-indices 0 50 \
  --rotations 0 \
  --output-dir evaluation/outputs/prior-source-ab-gate
```

这一阶段的详细输出写入文件，当前终端只显示开始和完成提示。若想看实时进度，
请在**另一个终端**执行：

```bash
tail -f evaluation/outputs/prior-source-ab-gate/pixie-inference.log
```

查看结束后按 `Ctrl+C` 只会退出日志查看，不会停止第一个终端里的模型运行。

把两条路线与相同真值比较：

```bash
python -m evaluation.run_prior_source_ab \
  --mode evaluate \
  --subject-indices 0 50 \
  --rotations 0 \
  --output-dir evaluation/outputs/prior-source-ab-gate
```

最后汇总成报告：

```bash
python -m evaluation.summarize_prior_source_ab \
  --input-csv evaluation/outputs/prior-source-ab-gate/paired_summary.csv \
  --output-dir evaluation/outputs/prior-source-ab-gate/analysis
```

## 6. 两例门的人工验收

每例至少检查：

- `pixie-inference/icon-filter/obj/*_recon.obj`：原始官方单图输出；
- `cases/<sample>/pixie_image_prior.obj`：映射到评测坐标后的输出；
- `cases/<sample>/prepared_normal.png`：Prepared 路线与真值的法向对比；
- `cases/<sample>/pixie_normal.png`：PIXIE 路线与真值的法向对比；
- `cases/<sample>/paired_metrics.json`：逐例指标、坐标记录和错误信息；
- `paired_summary.csv`：成对结果总表。

两例全部成功后才能进入 30 例。两例出现失败时，先看 `error_type`、
`error_message` 和对应 trace，不要通过自动对齐把错误藏起来。
