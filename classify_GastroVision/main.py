# pyright: reportPrivateImportUsage=false, reportArgumentType=false
import argparse
import torch
from torch import cat, device as torch_device
from pathlib import Path

from classify_GastroVision.config import *
from classify_GastroVision.data.dataset import get_dataloaders
from classify_GastroVision.models.classifier import GastroClassifier
from classify_GastroVision.training.train import train_model
from classify_GastroVision.training.evaluate import compute_metrics, compute_per_class_metrics
from classify_GastroVision.visualization import plot_all_results, plot_confusion_matrix, plot_per_class_metrics


def _find_latest_run():
    """扫描 OUTPUT_BASE 找最新时间戳子目录，返回其 Path 或 None。"""
    if not OUTPUT_BASE.exists():
        return None
    runs = [d for d in OUTPUT_BASE.iterdir() if (d / "checkpoints" / "best.pth").exists()]
    return max(runs, key=lambda d: d.name) if runs else None


def run_train():
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Classes ({len(class_names)}): {[c[:20] for c in class_names]}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = GastroClassifier()
    history, best_f1, run_dir = train_model(model, train_loader, val_loader, class_names)
    print(f"\nBest validation Macro F1: {best_f1:.4f}")

    print("\nEvaluating on test set...")
    model.load_checkpoint(run_dir / "checkpoints" / "best.pth")
    model.eval()
    device = next(model.parameters()).device

    all_preds, all_targets = [], []
    for images, labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_targets.append(labels)

    preds = cat(all_preds).numpy()
    targets = cat(all_targets).numpy()

    metrics = compute_metrics(preds, targets)
    per_class = compute_per_class_metrics(preds, targets, class_names)

    print(f"Test Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Test Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1:  {metrics['weighted_f1']:.4f}")

    print("\nGenerating visualizations...")
    plot_all_results(history, per_class, metrics["confusion_matrix"], class_names,
                     save_dir=run_dir / "figures")
    print(f"All figures saved to {run_dir / 'figures'}/")

    return metrics


def run_eval(checkpoint_path, run_dir=None):
    print("Loading data...")
    _, _, test_loader, class_names = get_dataloaders()
    print(f"Classes: {len(class_names)}, Test samples: {len(test_loader.dataset)}")

    model = GastroClassifier()
    model.load_checkpoint(Path(checkpoint_path))
    model.eval()
    device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds, all_targets = [], []
    for images, labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_targets.append(labels)

    preds = cat(all_preds).numpy()
    targets = cat(all_targets).numpy()
    metrics = compute_metrics(preds, targets)
    per_class = compute_per_class_metrics(preds, targets, class_names)

    print(f"Test Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Test Macro F1:     {metrics['macro_f1']:.4f}")

    fig_dir = (run_dir / "figures") if run_dir else FIGURE_DIR
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_dir=fig_dir)
    plot_per_class_metrics(per_class, class_names, save_dir=fig_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GastroVision Classification")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval mode")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory name (eval mode resolves to OUTPUT_BASE/run-dir/best.pth)")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "eval":
        if args.checkpoint:
            run_eval(args.checkpoint)
        elif args.run_dir:
            run_eval(OUTPUT_BASE / args.run_dir / "checkpoints" / "best.pth",
                     run_dir=OUTPUT_BASE / args.run_dir)
        else:
            latest = _find_latest_run()
            if latest is not None:
                print(f"Using latest run: {latest.name}")
                run_eval(latest / "checkpoints" / "best.pth", run_dir=latest)
            else:
                run_eval(CHECKPOINT_DIR / "best.pth")


if __name__ == "__main__":
    main()
