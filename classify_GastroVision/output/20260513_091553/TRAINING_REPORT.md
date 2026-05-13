# GastroVision v2.1 训练报告 — CE + 过采样 + RandAugment

**运行 ID:** `20260513_091553`
**日期:** 2026-05-13
**分支:** `GastroVision-v2.1`

---

## 配置

| 参数 | 值 |
|------|-----|
| 主干网络 | EfficientNetV2-S (ImageNet 预训练) |
| 输入尺寸 | 384×384 |
| 类别数 | 27 |
| 优化器 | AdamW (lr=3e-4, weight_decay=1e-4) |
| 调度器 | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| 损失函数 | **CrossEntropyLoss**（label_smoothing=0.1, 类别加权） |
| 数据增强 | RandAugment (num_ops=2, magnitude=5) |
| MixUp/CutMix | 禁用 |
| 过采样 | MIN_SAMPLES=100（固定种子 42） |
| 训练策略 | Phase 1 冻结 5 epoch → Phase 2 全量微调 |
| 早停 | patience=10 |
| 混合精度 | bfloat16 + GradScaler |
| 采样策略 | WeightedRandomSampler |

---

## 数据集

| 划分 | 样本数 |
|------|--------|
| 训练集 | 6,438（原 ~5,599 + 839 过采样） |
| 验证集 | 1,200 |
| 测试集 | 1,201 |

---

## 训练过程

### Phase 2 全量微调

| Phase 2 Epoch | Val F1（新最佳） |
|---------------|-----------------|
| 1 | 0.3936 |
| 3 | 0.4887 |
| 5 | 0.5197 |
| 7 | 0.5240 |
| 9 | 0.5780 |
| 12 | 0.5878 |
| 15 | 0.5933 |
| 17 | 0.6258 |
| **21** | **0.6281** |
| 23 | —（早停触发） |

**最佳验证 Macro F1: 0.6281**（Phase 2 epoch 21）
**早停: Phase 2 epoch 23**（总 epoch 28）

---

## 测试集结果

| 指标 | 值 |
|------|-----|
| Test Accuracy | **78.85%** |
| Test Macro F1 | **0.5703** |
| Test Weighted F1 | **0.7895** |

---

## 逐类指标

| 类别 | Precision | Recall | F1 | 测试样本 |
|------|-----------|--------|----|---------|
| Accessory tools | 0.9474 | 0.9474 | 0.9474 | 190 |
| Angiectasia | 1.0000 | 0.5000 | 0.6667 | 2 |
| Barrett's esophagus | 0.7143 | 0.3333 | 0.4545 | 15 |
| Blood in lumen | 0.6774 | 0.8077 | 0.7368 | 26 |
| Cecum | 0.1935 | 0.3529 | 0.2500 | 17 |
| Colon diverticula | 0.3333 | 0.5000 | 0.4000 | 4 |
| Colon polyps | 0.7181 | 0.8699 | 0.7868 | 123 |
| Colorectal cancer | 0.6190 | 0.6190 | 0.6190 | 21 |
| Duodenal bulb | 0.5128 | 0.6452 | 0.5714 | 31 |
| Dyed-lifted-polyps | 1.0000 | 0.9048 | 0.9500 | 21 |
| Dyed-resection-margins | 0.9722 | 0.9459 | 0.9589 | 37 |
| Erythema | 0.0000 | 0.0000 | 0.0000 | 2 |
| Esophageal varices | 0.0000 | 0.0000 | 0.0000 | 1 |
| Esophagitis | 0.2667 | 0.2500 | 0.2581 | 16 |
| Gastric polyps | 0.5714 | 0.8000 | 0.6667 | 10 |
| Gastroesophageal_junction_normal z-line | 0.7368 | 0.8400 | 0.7850 | 50 |
| Ileocecal valve | 0.4909 | 0.9000 | 0.6353 | 30 |
| Mucosal inflammation large bowel | 0.0000 | 0.0000 | 0.0000 | 4 |
| Normal esophagus | 0.7391 | 0.8095 | 0.7727 | 21 |
| Normal mucosa and vascular pattern in the large bowel | 0.8812 | 0.6409 | 0.7421 | 220 |
| Normal stomach | 0.9549 | 0.8699 | 0.9104 | 146 |
| Pylorus | 0.7432 | 0.9322 | 0.8271 | 59 |
| Resected polyps | 0.4444 | 0.2857 | 0.3478 | 14 |
| Resection margins | 1.0000 | 0.3333 | 0.5000 | 3 |
| Retroflex rectum | 0.7273 | 0.8000 | 0.7619 | 10 |
| Small bowel_terminal ileum | 0.8814 | 0.8189 | 0.8490 | 127 |
| Ulcer | 0.0000 | 0.0000 | 0.0000 | 1 |

---

## 全版本总对比

| 版本 | 损失函数 | 过采样 | Aug | MixUp | Acc | Macro F1 | W-F1 | Val Best |
|------|---------|:-----:|:---:|:----:|:---:|:--------:|:----:|:--------:|
| v2.0 Baseline | CE | ✗ | Basic | ✗ | 71.44% | **0.6464** | 0.7688 | 0.6548 |
| v2.0 Tuned | FL γ=1.0 | ✗ | RA=5 | 0.5 | 80.60% | 0.6275 | 0.8056 | 0.5927 |
| v2.1a | FL γ=0.5 | ✓ | RA=5 | 0.5 | 79.18% | 0.5727 | 0.7976 | 0.6318 |
| v2.1b | FL γ=1.0 | ✓ | RA=5 | 0.5 | 80.35% | 0.5944 | 0.8050 | 0.6734 |
| v2.1c | FL γ=1.0 | ✓ | RA=5 | ✗ | **81.85%** | 0.5989 | **0.8142** | 0.6379 |
| **v2.1（本轮）** | **CE** | **✓** | **RA=5** | **✗** | 78.85% | **0.5703** | 0.7895 | 0.6281 |

---

## 分析

**CE + RandAugment 表现不佳。** CrossEntropy 基线在基础增强下 Macro F1 为 0.6464，但加上 RandAugment（mag=5）后反而降至 0.5703。说明 RandAugment 的强度对 CE 损失函数来说过大，可能在缺乏 Focal Loss 的梯度保护下过度破坏了判别特征。

**当前最优组合：FL γ=1.0 + 过采样 + 关 MixUp（v2.1c）**，Accuracy 81.85% 和 Weighted F1 0.8142 均为全版本最高，但 Macro F1（0.5989）未能追上 CE 基线（0.6464）。

**Focal Loss 的双面效应：**
- FL 提升了 Accuracy 和 Weighted F1（多数类更好），但 Macro F1 始终低于 CE
- 过采样改善了部分尾部类（Angiectasia 0→0.67, Colorectal cancer 0.56→0.70）
- MixUp 在 245:1 极不均衡下有副作用，关闭后 Acc 和 W-F1 提升

### 尾部类瓶颈

3 类 F1=0（Erythema 2 张, Esophageal varices 1 张, Ulcer 1 张）系测试样本过少，无法可靠评估。其余尾部类 F1 在 0.25-0.76 之间，改善空间仍大。

---

## 图表

保存在 `figures/` 目录：
- `training_curves.png`
- `confusion_matrix.png`
- `per_class_metrics.png`
