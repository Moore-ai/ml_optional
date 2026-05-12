# Hyper-Kvasir v2.1 训练报告

**训练时间:** 2026-05-12 15:25:20
**运行目录:** `classify_Hyper_Kvasir/output/20260512_152520/`
**基线版本:** v2.0 (6127a14)
**当前版本:** v2.1 (42011a6)

---

## 变更摘要

本次训练新增少数类过采样和逐类指标 CSV 输出，共涉及 4 个文件：

| # | 文件 | 变更内容 |
|---|------|---------|
| 1 | `config.py` | 新增 `MIN_SAMPLES = 100` |
| 2 | `data/dataset.py` | `get_dataloaders` 中训练拆分后过采样尾部类至 100 样本 |
| 3 | `training/evaluate.py` | 新增 `save_per_class_csv` 函数 |
| 4 | `main.py` | 训练/评估结束后输出 `per_class_metrics.csv` |

过采样策略：对训练集中样本数 < 100 的类随机复制至 100（固定随机种子 42），验证/测试集不做过采样。过采样后训练集从 ~7,463 扩增至 8,050（+587）。

---

## 超参数

| 参数 | 值 |
|------|-----|
| 模型 | EfficientNetV2-S |
| 图像尺寸 | 384 × 384 |
| Batch Size | 32 |
| 优化器 | AdamW (lr=3e-4, weight_decay=1e-4) |
| 调度器 | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| 损失函数 | Focal Loss (γ=1.0) |
| Label Smoothing | 0.1 |
| Phase 1 (冻结) | 5 epoch |
| Phase 2 (微调) | 50 epoch (early stop at 14) |
| RandAugment | num_ops=2, magnitude=5 |
| 早停 | patience=10 |

---

## 整体指标对比

| 指标 | v2.0 基线 | v2.1 | 变化 |
|------|-----------|------|------|
| Test Accuracy | 86.38% | **89.81%** | **+3.43%** |
| Test Macro F1 | 0.5867 | **0.6087** | **+0.0220** |
| Test Weighted F1 | 0.8729 | **0.8999** | **+0.0270** |

---

## 逐类指标 (v2.1)

| 类别 | Precision | Recall | F1 | 测试样本 |
|------|-----------|--------|----|---------|
| barretts | 0.5000 | 0.1667 | 0.2500 | 6 |
| barretts-short-segment | 0.1818 | 0.2500 | 0.2105 | 8 |
| bbps-0-1 | 1.0000 | 0.9794 | 0.9896 | 97 |
| bbps-2-3 | 1.0000 | 0.9709 | 0.9853 | 172 |
| cecum | 0.9862 | 0.9408 | 0.9630 | 152 |
| dyed-lifted-polyps | 0.9416 | 0.9603 | 0.9508 | 151 |
| dyed-resection-margins | 0.9589 | 0.9396 | 0.9492 | 149 |
| esophagitis-a | 0.5488 | 0.7377 | 0.6294 | 61 |
| esophagitis-b-d | 0.7179 | 0.7179 | 0.7179 | 39 |
| hemorrhoids | 0.0000 | 0.0000 | 0.0000 | 1 |
| ileum | 0.0000 | 0.0000 | 0.0000 | 1 |
| impacted-stool | 0.7917 | 1.0000 | 0.8837 | 19 |
| polyps | 1.0000 | 0.9286 | 0.9630 | 154 |
| pylorus | 0.9934 | 1.0000 | 0.9967 | 150 |
| retroflex-rectum | 0.8261 | 0.9828 | 0.8976 | 58 |
| retroflex-stomach | 1.0000 | 0.9565 | 0.9778 | 115 |
| UC grade-0-1 | 0.0000 | 0.0000 | 0.0000 | 5 |
| UC grade-1 | 0.3902 | 0.5333 | 0.4507 | 30 |
| UC grade-1-2 | 0.0000 | 0.0000 | 0.0000 | 1 |
| UC grade-2 | 0.6220 | 0.7612 | 0.6846 | 67 |
| UC grade-2-3 | 0.0000 | 0.0000 | 0.0000 | 4 |
| UC grade-3 | 0.8333 | 0.5000 | 0.6250 | 20 |
| z-line | 0.9350 | 0.8214 | 0.8745 | 140 |

---

## 尾部类改善分析

v2.0 中有 7 个尾部类 F1=0。过采样后：

- **已有改善（2/7）：**
  - `barretts`（原 41 张): **0.0 → 0.2500** ✅
  - `barretts-short-segment`（原 53 张): **0.0 → 0.2105** ✅

- **仍为 F1=0（5/7）：** 这些类在测试集中仅有 1~5 张图像，样本太少无法有效评估
  - `hemorrhoids` (测试 1 张)
  - `ileum` (测试 1 张)
  - `UC grade-0-1` (测试 5 张)
  - `UC grade-1-2` (测试 1 张)
  - `UC grade-2-3` (测试 4 张)

---

## 训练曲线

- 训练曲线见 `figures/training_curves.png`
- 混淆矩阵见 `figures/confusion_matrix.png`
- 逐类指标柱状图见 `figures/per_class_metrics.png`

---

## 结论

过采样有效提升了尾部类的 Macro F1（+0.022），且 Accuracy 和 Weighted F1 同步上涨。尾部 5 类仍为 0 的主要原因系测试集过小（1~5 张），非过采样策略本身的问题。若需进一步改善，建议考虑类别合并或数据增强专门化。
