# GastroVision 数据集文档

## 来源

- **存储位置**: OSF (Open Science Framework)
- **数据集路径**: `GastroVision/osfstorage-archive/Gastrovision/Gastrovision/Gastrovision/`
- **总大小**: ~3.4 GB

## 基本信息

| 属性 | 值 |
|---|---|
| 图像格式 | JPEG |
| 图像尺寸 | 768×576（多数），720×576（少量） |
| 彩色通道 | 3 (RGB) |
| 类别数 | 27 |
| 总图像数 | ~7,000+ |

## 类别分布

数据集存在严重的类别不均衡问题。列表按样本数降序排列：

| 类别 | 样本数 |
|---|---|
| Normal mucosa and vascular pattern in the large bowel | 1467 |
| Accessory tools | 1266 |
| Normal stomach | 969 |
| Small bowel_terminal ileum | 846 |
| Colon polyps | 820 |
| Pylorus | 393 |
| Gastroesophageal_junction_normal z-line | 330 |
| Dyed-resection-margins | 246 |
| Duodenal bulb | 205 |
| Ileocecal valve | 200 |
| Blood in lumen | 171 |
| Dyed-lifted-polyps | 141 |
| Normal esophagus | 140 |
| Colorectal cancer | 139 |
| Cecum | 113 |
| Esophagitis | 107 |
| Barrett's esophagus | 95 |
| Resected polyps | 92 |
| Retroflex rectum | 67 |
| Gastric polyps | 65 |
| Mucosal inflammation large bowel | 29 |
| Colon diverticula | 29 |
| Resection margins | 25 |
| Angiectasia | 17 |
| Erythema | 15 |
| Esophageal varices | 7 |
| Ulcer | 6 |

### 分布统计

- **最多类别**: Normal mucosa and vascular pattern in the large bowel (1467)
- **最少类别**: Ulcer (6)
- **不均衡比**: ~245:1
- **中位数**: ~113
- **均值**: ~260

## 目录结构

```
GastroVision/osfstorage-archive/
├── Gastrovision/
│   └── Gastrovision.zip
│       └── Gastrovision/
│           └── Gastrovision/          ← 实际数据根目录
│               ├── Accessory tools/
│               ├── Angiectasia/
│               ├── Barrett's esophagus/
│               ├── Blood in lumen/
│               ├── Cecum/
│               ├── Colon diverticula/
│               ├── Colon polyps/
│               ├── Colorectal cancer/
│               ├── Duodenal bulb/
│               ├── Dyed-lifted-polyps/
│               ├── Dyed-resection-margins/
│               ├── Erythema/
│               ├── Esophageal varices/
│               ├── Esophagitis/
│               ├── Gastric polyps/
│               ├── Gastroesophageal_junction_normal z-line/
│               ├── Ileocecal valve/
│               ├── Mucosal inflammation large bowel/
│               ├── Normal esophagus/
│               ├── Normal mucosa and vascular pattern in the large bowel/
│               ├── Normal stomach/
│               ├── Pylorus/
│               ├── Resected polyps/
│               ├── Resection margins/
│               ├── Retroflex rectum/
│               ├── Small bowel_terminal ileum/
│               └── Ulcer/
└── Wiki images/                       ← 说明图片
    ├── gastrovision5 (1).png
    ├── resolution.png
    └── ...
```

## 内容概述

数据集涵盖胃肠道内窥镜检查中的多种场景：

- **正常解剖结构**: Normal esophagus, Normal stomach, Normal mucosa, Duodenal bulb, Pylorus, Cecum, Ileocecal valve, etc.
- **病变/异常**: Esophagitis, Barrett's esophagus, Esophageal varices, Colon polyps, Colorectal cancer, Ulcer, Angiectasia, Erythema, etc.
- **术后场景**: Dyed-lifted-polyps, Dyed-resection-margins, Resection margins, Resected polyps
- **器械**: Accessory tools
- **其他**: Blood in lumen, Retroflex rectum, Mucosal inflammation

## 使用建议

### 类别不均衡处理

鉴于严重的不均衡比（245:1），建议：

1. **过采样** — 对小类别（<50 样本）进行复制或 SMOTE
2. **数据增强** — 随机翻转、旋转、色彩抖动等
3. **加权损失函数** — 按类别频率设置权重
4. **分层采样** — 确保训练/验证/测试集保持各类别比例

### 数据集拆分

建议按类别分层拆分，避免少数类在某一折中缺失：

- 训练集: 70%
- 验证集: 15%
- 测试集: 15%

### 评价指标

由于不均衡特性，不应仅使用准确率。建议关注：

- Macro F1-score
- Precision / Recall（按类别）
- Confusion Matrix
- AUC-ROC（对二分类任务）

## 代码中引用路径

```python
from pathlib import Path

data_root = Path("GastroVision/osfstorage-archive/Gastrovision/Gastrovision/Gastrovision")
```
