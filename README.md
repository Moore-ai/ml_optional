# 环境配置

## 依赖管理

本项目使用 `uv` 管理依赖，需先安装 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

## CUDA 后端

PyTorch 安装 CUDA 13.0 版本，来自 PyTorch 官方索引：

```bash
uv sync --index-strategy unsafe-best-match
```

## 可用命令

```bash
uv run python classify_GastroVision/main.py --mode train   # 训练
uv run python classify_GastroVision/main.py --mode eval    # 评估
uv add <package>                                            # 添加依赖
```
