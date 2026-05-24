# 机器学习应用期末报告选做题

基于 EfficientNetV2-S 迁移学习的胃肠道内窥镜图像多分类项目，包含两个独立子任务。

[必选题仓库](https://github.com/Moore-ai/ml_necessary.git)

## 最佳结果

| 数据集 | Accuracy | Macro F1 | Weighted F1 | 配置 |
|--------|:--------:|:--------:|:-----------:|------|
| **GastroVision** (27 类) | 81.85% | 0.5989 | 0.8142 | FL γ=1.0 + 过采样 + RA + 无 MixUp |
| **Hyper-Kvasir** (23 类) | 89.81% | 0.6087 | 0.8999 | FL γ=1.0 + 过采样 + RA + MixUp |

## 项目结构

```
├── classify_GastroVision/          ← GastroVision 分类
│   ├── config.py                   — 超参数 + 改进开关
│   ├── main.py                     — CLI 入口
│   ├── data/                       — 数据集 + 增强
│   ├── models/                     — EfficientNetV2-S 分类器
│   ├── training/                   — 训练 + 评估 + 损失函数
│   └── visualization.py           — 曲线 + 混淆矩阵 + 逐类柱状图
├── classify_Hyper_Kvasir/          ← Hyper-Kvasir 分类
│   └── (同构)
├── dataset-summary/                — 数据集文档
├── docs/superpowers/               — 设计文档与实验计划
└── pyproject.toml
```

## 环境配置

依赖管理使用 [uv](https://docs.astral.sh/uv/)，CUDA 13.0：

```bash
uv sync --index-strategy unsafe-best-match
```

## 使用方法

```bash
# GastroVision
uv run python -m classify_GastroVision.main --mode train
uv run python -m classify_GastroVision.main --mode eval
uv run python -m classify_GastroVision.main --mode eval --checkpoint path/to/best.pth

# Hyper-Kvasir
uv run python -m classify_Hyper_Kvasir.main --mode train
uv run python -m classify_Hyper_Kvasir.main --mode eval
```

训练输出按时间戳组织到 `output/<YYYYmmDD_HHMMSS>/`，包含 `checkpoints/`、`figures/`、`logs/`。

## 数据集

| 属性 | GastroVision | Hyper-Kvasir |
|------|-------------|--------------|
| 类别数 | 27 | 23 |
| 图像数 | ~7,000 | 10,662 |
| 图像尺寸 | 768×576 | 不定 |
| 目录结构 | 扁平 | 三层层级 |
| 类别不均衡比 | 245:1 | ~43:1 |

## 训练策略

**二阶段微调**：Phase 1 冻结主干 5 epoch → Phase 2 全量微调 50 epoch（early stop patience=10）。

**可配置改进**（`config.py` 开关）：

- **Focal Loss**（γ 可调，类别加权）
- **RandAugment**（num_ops=2, magnitude=5）
- **MixUp / CutMix**（可调概率和 α）
- **少数类过采样**（MIN_SAMPLES=100）
- **Test-Time Augmentation**（原始 + 水平翻转，2 视图平均）
- **Label Smoothing**（0.1）

共享配置：AdamW (lr=3e-4)、CosineAnnealingLR (T_max=50)、混合精度 bfloat16 + GradScaler。

## GastroVision 实验历程

| 版本 | 配置 | Acc | Macro F1 | W-F1 |
|------|------|:---:|:--------:|:----:|
| v2.0 Baseline | CE + 基础增强 | 71.44% | **0.6464** | 0.7688 |
| v2.0 Tuned | +FL γ=1 +RA +MixUp | 80.60% | 0.6275 | 0.8056 |
| v2.1 (a) | +Oversample +γ=0.5 | 79.18% | 0.5727 | 0.7976 |
| v2.1 (b) | +Oversample +γ=1.0 | 80.35% | 0.5944 | 0.8050 |
| **v2.1 (c)** | **+Oversample +γ=1.0 -MixUp** | **81.85%** | 0.5989 | **0.8142** |
| v2.1 (d) | CE + Oversample +RA | 78.85% | 0.5703 | 0.7895 |
| v2.1 (e) | CE + Oversample +基础增强 | 81.77% | 0.5842 | 0.8130 |

**结论**：Focal Loss 提升 Accuracy/Weighted F1 但降低 Macro F1；MixUp 在 245:1 极不均衡下有副作用；过采样改善部分尾部类。

## Hyper-Kvasir 实验历程

| 版本 | 配置 | Acc | Macro F1 | W-F1 |
|------|------|:---:|:--------:|:----:|
| v1.0 Baseline | CE + 基础增强 | 86.38% | 0.5867 | 0.8729 |
| v2.0 | +FL +RA +Label Smoothing | 88.37% | 0.5927 | 0.8837 |
| v2.1a | +Oversample +MixUp | 86.30% | 0.5977 | 0.8630 |
| **v2.1** | **+Oversample +MixUp** | **89.81%** | **0.6087** | **0.8999** |

**结论**：Focal Loss + RandAugment + 过采样 + MixUp 组合效果最佳，Accuracy +3.43%，Macro F1 +0.022。

## 代码质量

```bash
uv run pyright                    # 类型检查
uv run pyright <file>             # 单文件检查
```
