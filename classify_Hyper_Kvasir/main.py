# pyright: reportPrivateImportUsage=false, reportArgumentType=false
import argparse
import torch
from torch import cat, device as torch_device
from pathlib import Path

from classify_Hyper_Kvasir.config import *
from classify_Hyper_Kvasir.data.dataset import get_dataloaders
from classify_Hyper_Kvasir.models.classifier import GastroClassifier
from classify_Hyper_Kvasir.training.train import train_model
from classify_Hyper_Kvasir.training.evaluate import compute_metrics, compute_per_class_metrics
from classify_Hyper_Kvasir.visualization import plot_all_results, plot_confusion_matrix, plot_per_class_metrics


def run_train():
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Classes ({len(class_names)}): {[c[:20] for c in class_names]}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = GastroClassifier()
    history, best_f1 = train_model(model, train_loader, val_loader, class_names)
    print(f"\nBest validation Macro F1: {best_f1:.4f}")

    print("\nEvaluating on test set...")
    model.load_checkpoint(CHECKPOINT_DIR / "best.pth")
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
    plot_all_results(history, per_class, metrics["confusion_matrix"], class_names)
    print(f"All figures saved to {FIGURE_DIR}/")

    return metrics


def run_eval(checkpoint_path):
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

    plot_confusion_matrix(metrics["confusion_matrix"], class_names)
    plot_per_class_metrics(per_class, class_names)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Hyper-Kvasir Classification")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval mode")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "eval":
        run_eval(args.checkpoint or CHECKPOINT_DIR / "best.pth")


if __name__ == "__main__":
    main()
