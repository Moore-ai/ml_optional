import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classify_GastroVision.config import *


def plot_training_curves(history, save_dir=FIGURE_DIR):
    """绘制 Loss / Accuracy / Macro F1 曲线并保存。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc"), ("train_f1", "val_f1")],
        ["Loss", "Accuracy", "Macro F1"],
    ):
        ax.plot(epochs, history[key[0]], "b-", label="Train", linewidth=1.5)
        ax.plot(epochs, history[key[1]], "r-", label="Val", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "training_curves.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrix(cm, class_names, save_dir=FIGURE_DIR):
    """绘制 27×27 混淆矩阵热力图并保存。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(22, 22))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()
    path = save_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class_metrics(per_class_metrics, class_names, save_dir=FIGURE_DIR):
    """按 F1 排序绘制 Precision / Recall / F1 横向柱状图。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    names = list(per_class_metrics.keys())
    precision = [per_class_metrics[n]["precision"] for n in names]
    recall = [per_class_metrics[n]["recall"] for n in names]
    f1 = [per_class_metrics[n]["f1"] for n in names]

    # 按 F1 升序排列
    sorted_idx = np.argsort(f1)
    names_sorted = [names[i] for i in sorted_idx]
    data = {k: [v[i] for i in sorted_idx] for k, v in
            [("Precision", precision), ("Recall", recall), ("F1", f1)]}

    fig, ax = plt.subplots(figsize=(12, 14))
    y = np.arange(len(names))
    height = 0.25

    for i, (label, color) in enumerate([("Precision", "#4ECDC4"), ("Recall", "#45B7D1"), ("F1", "#96CEB4")]):
        ax.barh(y + (i - 1) * height, data[label], height, label=label, alpha=0.8, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Per-Class Metrics (sorted by F1)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    path = save_dir / "per_class_metrics.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_all_results(history, per_class_metrics, cm, class_names, save_dir=FIGURE_DIR):
    """生成全部可视化图表。"""
    plot_training_curves(history, save_dir)
    plot_confusion_matrix(cm, class_names, save_dir)
    plot_per_class_metrics(per_class_metrics, class_names, save_dir)
