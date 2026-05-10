# Hyper-Kvasir 数据集文档

## 来源

- **论文**: Hyper-Kvasir, A comprehensive multi-class image and video dataset for gastrointestinal endoscopy
- **存储位置**: `Hyper-Kvasir/labeled-images/`
- **总大小**: ~3.7 GB
- **图片格式**: JPEG，尺寸不一（常见 635×547、619×529、1221×1013 等）
- **彩色通道**: 3 (RGB)

## 基本信息

| 属性 | 值 |
|---|---|
| 总图像数 | 10,662 |
| 类别数 | 23 |
| 标注粒度 | 图像级 |
| 数据来源 | 胃镜和结肠镜检查 |
| 附加信息 | `image-labels.csv` 包含视频来源、器官、发现类型、分类层级 |

## 目录结构

```
Hyper-Kvasir/labeled-images/
├── image-labels.csv                     ← 元数据（Video file, Organ, Finding, Classification）
├── lower-gi-tract/                      ← 下消化道
│   ├── anatomical-landmarks/            ← 解剖标志
│   │   ├── cecum/
│   │   ├── ileum/
│   │   └── retroflex-rectum/
│   ├── pathological-findings/           ← 病理发现
│   │   ├── hemorrhoids/
│   │   ├── polyps/
│   │   ├── ulcerative-colitis-grade-0-1/
│   │   ├── ulcerative-colitis-grade-1/
│   │   ├── ulcerative-colitis-grade-1-2/
│   │   ├── ulcerative-colitis-grade-2/
│   │   ├── ulcerative-colitis-grade-2-3/
│   │   └── ulcerative-colitis-grade-3/
│   ├── quality-of-mucosal-views/        ← 黏膜视图质量
│   │   ├── bbps-0-1/
│   │   ├── bbps-2-3/
│   │   └── impacted-stool/
│   └── therapeutic-interventions/       ← 治疗干预
│       ├── dyed-lifted-polyps/
│       └── dyed-resection-margins/
└── upper-gi-tract/                      ← 上消化道
    ├── anatomical-landmarks/            ← 解剖标志
    │   ├── pylorus/
    │   ├── retroflex-stomach/
    │   └── z-line/
    └── pathological-findings/           ← 病理发现
        ├── barretts/
        ├── barretts-short-segment/
        ├── esophagitis-a/
        └── esophagitis-b-d/
```

## 类别分布

### 下消化道（15 类）

| 类别 | 样本数 | 子类别 |
|------|--------|--------|
| cecum (盲肠) | 1,009 | anatomical-landmarks |
| polyps (息肉) | 1,028 | pathological-findings |
| dyed-lifted-polyps (染色抬高息肉) | 1,002 | therapeutic-interventions |
| dyed-resection-margins (染色切缘) | 989 | therapeutic-interventions |
| ulcerative-colitis-grade-2 (溃疡性结肠炎 2 级) | 443 | pathological-findings |
| retroflex-rectum (直肠反转) | 391 | anatomical-landmarks |
| ulcerative-colitis-grade-1 (溃疡性结肠炎 1 级) | 201 | pathological-findings |
| ulcerative-colitis-grade-3 (溃疡性结肠炎 3 级) | 133 | pathological-findings |
| impacted-stool (粪便嵌塞) | 131 | quality-of-mucosal-views |
| ulcerative-colitis-grade-0-1 (溃疡性结肠炎 0-1 级) | 35 | pathological-findings |
| ulcerative-colitis-grade-2-3 (溃疡性结肠炎 2-3 级) | 28 | pathological-findings |
| ulcerative-colitis-grade-1-2 (溃疡性结肠炎 1-2 级) | 11 | pathological-findings |
| ileum (回肠) | 9 | anatomical-landmarks |
| hemorrhoids (痔疮) | 6 | pathological-findings |

### 黏膜视图质量（3 类，计入下消化道）

| 类别 | 样本数 | 说明 |
|------|--------|------|
| bbps-2-3 | 1,148 | 肠道准备良好 |
| bbps-0-1 | 646 | 肠道准备欠佳 |
| impacted-stool | 131 | 粪便影响观察 |

### 上消化道（7 类）

| 类别 | 样本数 | 子类别 |
|------|--------|--------|
| pylorus (幽门) | 999 | anatomical-landmarks |
| z-line (Z 线/食管胃交界) | 932 | anatomical-landmarks |
| retroflex-stomach (胃反转) | 764 | anatomical-landmarks |
| esophagitis-a (食管炎 A 级) | 403 | pathological-findings |
| esophagitis-b-d (食管炎 B-D 级) | 260 | pathological-findings |
| barretts-short-segment (巴雷特食管短段) | 53 | pathological-findings |
| barretts (巴雷特食管) | 41 | pathological-findings |

### 分布统计

- **最多类别**: bbps-2-3 (1,148)
- **最少类别**: hemorrhoids (6)
- **不均衡比**: ~191:1
- **中位数**: ~201
- **均值**: ~463

## 内容概述

数据涵盖上下消化道内窥镜检查中的多种场景：

- **解剖标志**: cecum, pylorus, z-line, retroflex-rectum, retroflex-stomach, ileum
- **炎症性肠病**: ulcerative-colitis（多个严重等级 0-3）
- **息肉和肿瘤**: polyps, dyed-lifted-polyps
- **食管病变**: esophagitis（A/B-D 级）, barretts, barretts-short-segment
- **肛肠**: hemorrhoids
- **肠道准备质量**: bbps-0-1, bbps-2-3, impacted-stool
- **术后**: dyed-resection-margins

## 与 GastroVision 的差异

| 维度 | Hyper-Kvasir | GastroVision |
|------|-------------|-------------|
| 图像数 | 10,662 | ~8,000 |
| 类别数 | 23 | 27 |
| 组织结构 | 按 GI 部位 + 发现类型分层 | 扁平化，27 个独立目录 |
| 图像尺寸 | 不一（典型 500-1200px） | 统一 768×576 |
| 类别不均衡 | ~191:1 | ~245:1 |
| 元数据 | 含 CSV（视频来源、分类层级） | 无元数据文件 |
| 标签粒度 | 包含分级标签（UC 0-3, BBPS 0-3） | 仅二分类标签 |

## 使用建议

### 任务设计

- **解剖标志分类**: 从 23 类中识别 GI 解剖位置
- **溃疡性结肠炎分级**: 利用 6 个分级类别（0-3 及中间级别）
- **食管炎分级**: esophagitis-a vs esophagitis-b-d
- **肠道准备评估**: BBPS 评分分类

### 类别不均衡处理

少数类（hemorrhoids 6 张, ileum 9 张）建议：
1. 加权损失函数
2. 过采样或少类合并
3. 分层拆分确保验证/测试集覆盖

### 评价指标

- Macro F1-score
- 按类别的 Precision / Recall
- 混淆矩阵

## CSV 元数据

```csv
Video file,Organ,Finding,Classification
```

- **Video file**: 源视频文件 UUID
- **Organ**: GI 部位（Lower GI / Upper GI）
- **Finding**: 发现名称（如 cecum, polyps）
- **Classification**: 分类层级（如 anatomical-landmarks, pathological-findings）

## 代码中引用路径

```python
from pathlib import Path

data_root = Path("Hyper-Kvasir/labeled-images")
```
