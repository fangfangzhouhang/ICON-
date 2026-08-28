# 三维人体重建统一评测 v1

这套评测不是再造一个 ICON，也不是用一个“总分”决定所有论文谁最好。它的作用是把不同论文输出的 Mesh 放到**同一批考题、同一坐标系、同一把尺子、同一失败处理规则**下比较。

## 1. 从科研问题到结果的完整链路

```text
论文主张
  -> 选择能检验该主张的数据与指标
  -> 模型产生预测 Mesh
  -> 论文适配器只转换文件名和元数据
  -> 统一格式校验（缺例、重复、失败、路径）
  -> 用固定指标计算或汇总
  -> 只在同一案例上做配对比较
  -> 报告结论、失败案例和适用边界
```

这里最重要的思想是：**模型和尺子分开**。模型负责生成 Mesh；评测层负责证明大家确实在做同一道题。以后接入 PIFu、GTA 或 SIFU 时，不复制 ICON 网络，只给该方法写一个很薄的适配器。

## 2. 通用几何核心：所有方法都必须交

机器可读协议位于 `evaluation/protocols/geometry_cape_v1.yaml`。当前 CAPE 完整合同是 150 个条目、每个条目 3 个旋转，共 450 个案例。

| 指标 | 大白话 | 主要盲点 |
|---|---|---|
| Chamfer（cm） | 预测表面和真实表面双向检查，整体离得多远 | 平均数可能掩盖局部严重错误 |
| P2S（cm） | 从真实人体表面出发，看最近的预测表面有多远 | 单向检查可能看不见多出来的漂浮碎片 |
| Normal error | 比较表面朝向，观察褶皱和局部形状是否一致 | 依赖固定渲染视角与同一实现 |
| Success rate | 450 题中真正成功生成并评测的比例 | 不能把失败案例删掉后只报成功者均值 |
| Runtime / memory | 生成与评测的工程成本 | 必须统一硬件、预热和测量边界才可横比 |

三个几何指标都越小越好，但它们回答不同问题，所以 v1 **禁止生成一个主观加权的总分**。

## 3. 论文特有指标：按主张增加，不污染通用核心

- 如果论文声称“人体先验更稳”，增加 `prior-robustness-v1` 配对实验：同一张图分别使用 CAPE 准备好的 SMPL 和从图片估计的 PIXIE 先验。
- 如果论文声称“纹理更真实”，需要统一相机、光照和渲染，再增加 PSNR、SSIM、LPIPS；它们不能混进几何总分。
- 如果论文声称“更快、更省显存”，必须固定 GPU、分辨率、预热和计时范围。

这就是以后寻找创新点的方法：先把论文主张拆成可检验假设，再看它在哪类样本、哪个指标、哪一阶段失败，而不是只追一个平均数字。

## 4. 统一结果格式

每个模型每个案例输出一行 CSV。核心字段包括：

- 身份：协议、数据集、方法、案例 ID、难度组、人物、旋转角度；
- 结果：`ok / failed / skipped`，三项几何指标；
- 证据：预测 Mesh、真实 Mesh、错误信息；
- 可复现信息：随机种子、采样点数、网格分辨率、设备、Git commit、配置与权重哈希；
- 工程数据：耗时和峰值显存。

字段定义在 `evaluation/schema.py`。失败行是合法科研证据，但必须写错误原因；缺失一行、重复一行或成功行没有指标，都会被校验器判为失败。

## 5. 接入一篇新论文

1. 固定同一份 CAPE manifest 和许可证边界；
2. 用该论文自己的官方入口生成预测 Mesh；
3. 复制 `evaluation/adapters/TEMPLATE.py`，只做字段映射，不改数字；
4. 运行 `validate_submission`，确认没有静默丢例；
5. 使用本仓库已经校准过的指标实现，或把官方输出转换成统一格式；
6. 用 `compare_methods` 检查案例集合完全一致后再比较；
7. 查看平均数、中位数、P90、失败率和最好/最差案例；
8. 结合论文机制解释差异，并提出新的可证伪实验。

适配器严禁做 ICP 对齐、偷偷缩放、删除失败案例或修改指标。需要坐标转换时，必须作为公开、可审计的预处理步骤写进协议。

## 6. 先用已有 ICON 450 例验证新框架

以下步骤不重新运行 ICON 网络，只检验统一框架能否正确接住已有证据：

```bash
python -m evaluation.adapters.icon \
  --input evaluation/outputs/cape-full-450/pilot_summary.csv \
  --output evaluation/outputs/unified-v1/icon-filter.csv

python -m evaluation.validate_submission \
  --input evaluation/outputs/unified-v1/icon-filter.csv \
  --expected-count 450 \
  --check-paths \
  --output-dir evaluation/outputs/unified-v1/icon-validation

python -m evaluation.compare_methods \
  --submission icon-filter=evaluation/outputs/unified-v1/icon-filter.csv \
  --output-dir evaluation/outputs/unified-v1/icon-report
```

最后一条只有一个方法，因此它是在演练报告链路，不代表完成了论文横向对比。等 PIFu 等方法也生成同一格式后，再重复传入 `--submission pifu=...`；程序会拒绝案例不一致的“伪公平比较”。

## 7. 当前结论边界

- 450 个 CAPE 案例 + 三项指标可说明 ICON 在这份协议下的几何表现；
- 它不能自动等同于复现论文表格，因为还要核对论文的版本、分辨率、预处理、采样与基线合同；
- 只跑 ICON 不能证明优于 PIFu；必须让 PIFu 在同一案例和指标实现下产生可配对结果；
- 先验来源实验和纹理实验属于附加实验，不应改写通用几何分数。
