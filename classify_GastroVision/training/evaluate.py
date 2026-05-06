import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)


def compute_metrics(predictions, targets):
    """计算并返回完整指标字典。"""
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0.0),
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0.0),
        "confusion_matrix": confusion_matrix(targets, predictions),
    }


def compute_per_class_metrics(predictions, targets, class_names):
    """按类别计算 precision / recall / f1。"""
    precision = precision_score(targets, predictions, average=None, zero_division=0.0)
    recall = recall_score(targets, predictions, average=None, zero_division=0.0)
    f1 = f1_score(targets, predictions, average=None, zero_division=0.0)

    # 将 numpy 标量转换为 Python float，解决 dict 索引类型问题
    return {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i, name in enumerate(class_names)
    }
