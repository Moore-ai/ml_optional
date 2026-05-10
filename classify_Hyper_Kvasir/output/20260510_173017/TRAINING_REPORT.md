# Hyper-Kvasir 训练报告

## 模型配置

| 参数 | 值 |
|------|-----|
| 模型 | EfficientNetV2-S (ImageNet 预训练) |
| 分类头 | Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.2) → Linear(512→23) |
| 训练集 | 7,463 张 (70%) |
| 验证集 | 1,599 张 (15%) |
| 测试集 | 1,600 张 (15%) |

## 训练策略

| 参数 | 值 |
|------|-----|
| Phase 1 | 冻结主干 5 epoch，仅训练分类头 |
| Phase 2 | 全量微调，早停 patience=10 |
| 优化器 | AdamW (lr=3e-4, weight_decay=1e-4) |
| 调度器 | CosineAnnealingLR (T_max=30, eta_min=1e-6) |
| 损失 | 类别加权 CrossEntropyLoss + Label Smoothing 0.1 |
| 增强 | RandomHorizontalFlip, RandomRotation(15°), ColorJitter(0.2) |
| 混合精度 | bfloat16 + GradScaler |
| 图像尺寸 | 384×384 |

## 测试集结果

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.8363** |
| **Macro F1** | **0.5916** |
| **Weighted F1** | **0.8713** |

## 按类别 F1 (升序)

| 类别 | F1 | Precision | Recall | 训练样本 |
|------|----|-----------|--------|---------|
| barretts-short-segment | 0.0000 | 0.0000 | 0.0000 | 53 |
| hemorrhoids | 0.0000 | 0.0000 | 0.0000 | 6 |
| ileum | 0.0000 | 0.0000 | 0.0000 | 9 |
| ulcerative-colitis-grade-0-1 | 0.0000 | 0.0000 | 0.0000 | 35 |
| ulcerative-colitis-grade-1-2 | 0.0000 | 0.0000 | 0.0000 | 11 |
| ulcerative-colitis-grade-2-3 | 0.0000 | 0.0000 | 0.0000 | 28 |
| barretts | 0.1818 | 0.2000 | 0.1667 | 41 |
| ulcerative-colitis-grade-1 | 0.4706 | 0.5714 | 0.4000 | 201 |
| esophagitis-a | 0.6761 | 0.5926 | 0.7869 | 403 |
| esophagitis-b-d | 0.7097 | 0.9565 | 0.5641 | 260 |
| ulcerative-colitis-grade-2 | 0.7260 | 0.6709 | 0.7910 | 443 |
| ulcerative-colitis-grade-3 | 0.7273 | 0.6667 | 0.8000 | 133 |
| z-line | 0.7791 | 0.8899 | 0.6929 | 932 |
| impacted-stool | 0.8636 | 0.7600 | 1.0000 | 131 |
| bbps-2-3 | 0.8647 | 1.0000 | 0.7616 | 1,148 |
| dyed-resection-margins | 0.8993 | 0.9690 | 0.8389 | 989 |
| dyed-lifted-polyps | 0.9301 | 0.9852 | 0.8808 | 1,002 |
| polyps | 0.9379 | 1.0000 | 0.8831 | 1,028 |
| cecum | 0.9488 | 0.9858 | 0.9145 | 1,009 |
| retroflex-rectum | 0.9508 | 0.9062 | 1.0000 | 391 |
| pylorus | 0.9691 | 1.0000 | 0.9400 | 999 |
| bbps-0-1 | 0.9845 | 0.9896 | 0.9794 | 646 |
| retroflex-stomach | 0.9868 | 1.0000 | 0.9739 | 764 |

## 分析

**优势**: 17/23 类别 F1 > 0.67，头部类别（bbps-0-1, retroflex-stomach, pylorus）接近完美分类。数据量较大的类别性能优异。

**问题**: 6 个极端少数类（样本量 ≤53）F1 为 0。这 6 类合计仅 142 张（占总数 1.3%），模型从未学会识别它们。

**与 GastroVision 对比**:
- Hyper-Kvasir Accuracy **+12%** (83.63% vs 71.44%) — 多 50% 训练数据优势明显
- Hyper-Kvasir Macro F1 **-8.5%** (0.5916 vs 0.6464) — 6 个零 F1 类别严重拖低 Macro 均值
- Hyper-Kvasir Weighted F1 **+13%** (0.8713 vs 0.7688) — 大类别表现更好

## 改进方向

1. **少数类过采样**: 对 ≤53 样本的 6 个类别做复制过采样，或使用 SMOTE
2. **类别合并**: 将 UC 中间等级（grade-0-1, grade-1-2, grade-2-3）合并，减少类别数
3. **Focal Loss**: 替换 CrossEntropyLoss，让模型关注难分类样本
4. **数据增强强化**: 对少数类额外使用 CutMix / MixUp
