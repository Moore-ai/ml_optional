import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)


def compute_metrics(predictions, targets):
    """计算并返回完整指标字典。"""
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0.0),  # type: ignore[arg-type]
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0.0),  # type: ignore[arg-type]
        "confusion_matrix": confusion_matrix(targets, predictions),
    }


def compute_per_class_metrics(predictions, targets, class_names):
    """按类别计算 precision / recall / f1。"""
    precision = precision_score(targets, predictions, average=None, zero_division=0.0)  # type: ignore[arg-type]
    recall = recall_score(targets, predictions, average=None, zero_division=0.0)  # type: ignore[arg-type]
    f1 = f1_score(targets, predictions, average=None, zero_division=0.0)  # type: ignore[arg-type]

    # 将 numpy 标量转换为 Python float，解决 dict 索引类型问题
    return {
        name: {
            "precision": float(precision[i]),  # type: ignore[index]
            "recall": float(recall[i]),  # type: ignore[index]
            "f1": float(f1[i]),  # type: ignore[index]
        }
        for i, name in enumerate(class_names)
    }
