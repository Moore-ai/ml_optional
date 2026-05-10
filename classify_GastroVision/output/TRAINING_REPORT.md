# GastroVision 分类模型训练报告

**训练时间:** 2026-05-06

## 模型配置

| 参数 | 值 |
|------|-----|
| 骨干网络 | EfficientNetV2-S（ImageNet 预训练） |
| 输入尺寸 | 384×384 |
| 类别数 | 27 |
| 优化器 | AdamW（lr=3e-4, weight_decay=1e-4） |
| 调度器 | CosineAnnealingLR（T_max=30, eta_min=1e-6） |
| 损失函数 | CrossEntropyLoss（label_smoothing=0.1, 类别加权） |
| 混合精度 | bfloat16 + GradScaler |
| 采样策略 | WeightedRandomSampler |

### 训练策略

- **Phase 1:** 冻结主干，仅训练分类头（5 epoch）
- **Phase 2:** 全量微调（45 epoch，含早停 patience=10）

## 数据集划分

| 划分 | 样本数 |
|------|--------|
| 训练集 | 5,599 |
| 验证集 | 1,200 |
| 测试集 | 1,201 |
| **合计** | **8,000** |

类别严重不均衡（最多达 1,467 张，最少仅 6 张，比例 245:1），使用 WeightedRandomSampler + 类别加权损失处理。

## 训练过程

| Epoch | 阶段 | Val Macro F1 | 备注 |
|-------|------|-------------|------|
| 1-5 | Phase 1（冻结） | 0.2353（最佳） | 分类头热身 |
| 6 | Phase 2（微调） | 0.2857 | 开始全量微调 |
| 7 | Phase 2 | 0.3584 | |
| 8 | Phase 2 | 0.4530 | |
| 9 | Phase 2 | 0.4879 | |
| 10 | Phase 2 | 0.5150 | |
| 11 | Phase 2 | 0.5580 | |
| 12 | Phase 2 | 0.5798 | |
| 13 | Phase 2 | 0.5973 | |
| 14 | Phase 2 | 0.6195 | |
| 15 | Phase 2 | 0.6497 | |
| 16 | Phase 2 | **0.6548** | **最佳** |
| 17-29 | Phase 2 | ≤0.6548 | 未再提升 |
| — | **早停触发** | | patience=10，epoch 29 停止 |

## 测试集结果

| 指标 | 分数 |
|------|------|
| Accuracy | **0.7144**（71.44%） |
| Macro F1 | **0.6464**（64.64%） |
| Weighted F1 | **0.7688**（76.88%） |

### 结果分析

- **Weighted F1（0.7688）显著高于 Macro F1（0.6464）**，说明模型在大类上表现较好，小类样本不足仍是瓶颈。
- 相比随机基线（Macro F1 ≈ 0.037），模型学到了有意义的分类特征。
- Phase 2 全量微调带来了显著提升（0.2353 → 0.6548），说明 backbone 适应内窥镜图像域的必要性。

## 输出文件

```
output/
├── checkpoints/
│   ├── best.pth          — 最佳模型（251 MB，epoch 16，F1=0.6548）
│   └── latest.pth        — 最新检查点（epoch 29）
├── figures/
│   ├── training_curves.png    — Loss/Accuracy/Macro F1 曲线
│   ├── confusion_matrix.png   — 27×27 归一化混淆矩阵
│   └── per_class_metrics.png  — 各类别 Precision/Recall/F1 柱状图
└── logs/                     — TensorBoard 事件文件
    ├── Loss_train/ / Loss_val/
    ├── Acc_train/ / Acc_val/
    └── F1_train/ / F1_val/
```

## 改进方向

1. **类别不平衡:** 对小类尝试过采样或合成样本（如 MixUp）
2. **模型:** 尝试 EfficientNetV2-M/L 或 Focal Loss
3. **集成:** 多折交叉验证后集成推理
