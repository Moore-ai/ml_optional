# GastroVision 分类训练报告

**运行 ID:** `20260512_220442`
**日期:** 2026-05-12
**分支:** `GastroVision-v2.1`

---

## 配置摘要

| 参数 | 值 |
|------|-----|
| 主干网络 | EfficientNetV2-S (ImageNet 预训练) |
| 输入尺寸 | 384×384 |
| 类别数 | 27 |
| 优化器 | AdamW (lr=3e-4, weight_decay=1e-4) |
| 调度器 | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| 损失函数 | FocalLoss (γ=1.0, 类别加权) |
| 数据增强 | RandAugment (num_ops=2, magnitude=5) |
| MixUp/CutMix | 禁用 |
| 过采样 | MIN_SAMPLES=100（9 个尾部类复制至 100） |
| 训练策略 | Phase 1 冻结主干 5 epoch → Phase 2 全量微调 50 epoch |
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
| **合计** | **8,839** |

过采样覆盖 9 个尾部类（<100 样本），固定随机种子 42。

---

## 训练过程

### Phase 1: 冻结主干 (Epoch 1-5)

仅训练分类头，验证指标逐步上升。

### Phase 2: 全量微调 (Epoch 1-39)

| Epoch | Val F1（首次达到 / 新最佳） |
|-------|---------------------------|
| 1 | 0.3929（新最佳） |
| 3 | 0.4637（新最佳） |
| 5 | 0.5324（新最佳） |
| 8 | 0.5633（新最佳） |
| 12 | 0.5972（新最佳） |
| 21 | 0.6025（新最佳） |
| 29 | 0.6084（新最佳） |
| 33 | 0.6291（新最佳） |
| **37** | **0.6379（新最佳）** |
| 39 | —（早停触发） |

**最佳验证 Macro F1: 0.6379**（epoch 37，Phase 2 内）
**早停: epoch 39**（超过 10 epoch 未再突破）

---

## 测试集结果

| 指标 | 值 | v2.0 Tuned | CE 基线 |
|------|-----|-----------|---------|
| **Accuracy** | **0.8185** | 0.8060 | 0.7144 |
| **Macro F1** | **0.5989** | 0.6275 | 0.6464 |
| **Weighted F1** | **0.8142** | 0.8056 | 0.7688 |

Accuracy 和 Weighted F1 创新高，但 Macro F1 仍低于 CE 基线。

---

## 逐类指标

| 类别 | Precision | Recall | F1 | 测试样本 |
|------|-----------|--------|----|---------|
| Accessory tools | 0.9684 | 0.9684 | 0.9684 | 190 |
| Angiectasia | 1.0000 | 0.5000 | **0.6667** | 2 |
| Barrett's esophagus | 0.7500 | 0.6000 | 0.6667 | 15 |
| Blood in lumen | 0.8333 | 0.7692 | 0.8000 | 26 |
| Cecum | 0.2381 | 0.2941 | 0.2632 | 17 |
| Colon diverticula | 1.0000 | 0.2500 | 0.4000 | 4 |
| Colon polyps | 0.8400 | 0.8537 | 0.8468 | 123 |
| Colorectal cancer | 0.8125 | 0.6190 | **0.7027** | 21 |
| Duodenal bulb | 0.5556 | 0.8065 | 0.6579 | 31 |
| Dyed-lifted-polyps | 1.0000 | 0.8571 | 0.9231 | 21 |
| Dyed-resection-margins | 0.9000 | 0.9730 | 0.9351 | 37 |
| Erythema | 0.0000 | 0.0000 | **0.0000** | 2 |
| Esophageal varices | 0.0000 | 0.0000 | **0.0000** | 1 |
| Esophagitis | 0.5000 | 0.0625 | 0.1111 | 16 |
| Gastric polyps | 0.6154 | 0.8000 | **0.6957** | 10 |
| Gastroesophageal_junction_normal z-line | 0.6765 | 0.9200 | 0.7797 | 50 |
| Ileocecal valve | 0.4898 | 0.8000 | 0.6076 | 30 |
| Mucosal inflammation large bowel | 0.5000 | 0.2500 | 0.3333 | 4 |
| Normal esophagus | 0.7917 | 0.9048 | 0.8444 | 21 |
| Normal mucosa and vascular pattern in the large bowel | 0.8333 | 0.7727 | 0.8019 | 220 |
| Normal stomach | 0.9078 | 0.8767 | 0.8920 | 146 |
| Pylorus | 0.8571 | 0.9153 | 0.8852 | 59 |
| Resected polyps | 0.6667 | 0.2857 | 0.4000 | 14 |
| Resection margins | 0.5000 | 0.6667 | 0.5714 | 3 |
| Retroflex rectum | 0.7143 | 0.5000 | 0.5882 | 10 |
| Small bowel_terminal ileum | 0.8387 | 0.8189 | 0.8287 | 127 |
| Ulcer | 0.0000 | 0.0000 | **0.0000** | 1 |

F1=0 的 3 个类（Erythema, Esophageal varices, Ulcer）测试样本仅 1-2 张，无法有效评估。

---

## 改进历程对比

| 轮次 | 变更 | Acc | Macro F1 | W-F1 | Val Best |
|------|------|:---:|:--------:|:----:|:--------:|
| v2.0 Baseline | CE + 基础增强 | 71.44% | 0.6464 | 0.7688 | 0.6548 |
| v2.0 Tuned | +Focal γ=1 +RA=5 +MixUp | 80.60% | 0.6275 | 0.8056 | 0.5927 |
| v2.1 (a) | +Oversample +γ=0.5 | 79.18% | 0.5727 | 0.7976 | 0.6318 |
| v2.1 (b) | +Oversample +γ=1.0 | 80.35% | 0.5944 | 0.8050 | 0.6734 |
| **v2.1 (c)** | **+Oversample +γ=1.0 -MixUp** | **81.85%** | **0.5989** | **0.8142** | **0.6379** |

---

## 分析

### 过采样效果

- **正面：** Angiectasia 从 F1=0 提升至 0.6667（测试仅 2 张），Colorectal cancer 0.56→0.70，Gastric polyps 0.63→0.70
- **局限：** 3 类（Erythema/Esophageal varices/Ulcer）因测试样本仅 1-2 张无法有效评估

### MixUp/CutMix 影响

关闭 MixUp 后 Accuracy（+0.50%）和 Weighted F1（+0.92%）均有提升，说明在 245:1 极不均衡场景下 MixUp 的标签混合可能弊大于利。

### Macro F1 瓶颈

所有 Focal Loss 变体的 Macro F1 均低于 CE 基线（0.5989 vs 0.6464）。可能原因：
- Focal Loss 的梯度重新加权与 WeightedRandomSampler + 类别权重叠加后，对少数类产生过度或矛盾的梯度方向
- CE 基线虽无高级技巧，但在加权采样和分层拆分的配合下，Macro F1 表现反而最优

### 未来方向

1. **CrossEntropy + 过采样 + RandAugment** — CE 基线自身 Macro F1 最高，加上过采样和 RA 后可能同时提升所有指标
2. **测试集扩展** — 3 个 F1=0 的类测试样本仅 1-2 张，无法可靠评估，建议收集更多数据或考虑类别合并
3. **尝试更低 γ** — γ=0.5 加上 MixUp 会降低太多，但 CE（γ=0）+ 过采样可能才是最佳组合

---

## 可视化

图表保存在 `figures/` 目录：
- `training_curves.png` — Loss / Accuracy / Macro F1 曲线
- `confusion_matrix.png` — 27×27 混淆矩阵
- `per_class_metrics.png` — 按 F1 排序的 Precision / Recall / F1 柱状图
