# pyright: reportPrivateImportUsage=false
from datetime import datetime
import torch
import torch.nn as nn
from torch import bincount, cat, device as torch_device, tensor, bfloat16
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classify_Hyper_Kvasir.config import *
from classify_Hyper_Kvasir.training.evaluate import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = cat(all_preds).numpy()
    targets = cat(all_labels).numpy()
    metrics = compute_metrics(preds, targets)
    return running_loss / len(loader.dataset), metrics["accuracy"], metrics["macro_f1"]


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type="cuda", dtype=bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = cat(all_preds).numpy()
    targets = cat(all_labels).numpy()
    metrics = compute_metrics(preds, targets)
    return running_loss / len(loader.dataset), metrics["accuracy"], metrics["macro_f1"]


def train_model(model, train_loader, val_loader, class_names):
    device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    # 类别加权损失
    class_counts = bincount(
        tensor([label for _, label in train_loader.dataset])
    )
    weights = (1.0 / class_counts.float()).to(device)
    weights = weights / weights.sum()

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler("cuda")

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / timestamp
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    log_dir = run_dir / "logs"
    for d in [run_dir, ckpt_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc",
                                "train_f1", "val_f1")}
    best_f1 = 0.0
    patience = 0
    global_step = 0

    # === Phase 1: 冻结主干，只训练分类头 ===
    print("Phase 1: Training head (frozen backbone)")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FREEZE_EPOCHS, eta_min=ETA_MIN,
    )

    for epoch in range(FREEZE_EPOCHS):
        loss, acc, f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(f1)
        history["val_f1"].append(val_f1)

        writer.add_scalars("Loss", {"train": loss, "val": val_loss}, global_step)
        writer.add_scalars("Acc", {"train": acc, "val": val_acc}, global_step)
        writer.add_scalars("F1", {"train": f1, "val": val_f1}, global_step)
        global_step += 1

        print(f"[P1 {epoch+1}/{FREEZE_EPOCHS}] Loss: {loss:.4f} Acc: {acc:.4f} F1: {f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_checkpoint(
                ckpt_dir / "best.pth", epoch, optimizer, scheduler,
                {"val_f1": val_f1, "val_acc": val_acc},
            )

    # === Phase 2: 全量微调 ===
    print("Phase 2: Full fine-tuning")
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN,
    )

    for epoch in range(EPOCHS):
        loss, acc, f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(f1)
        history["val_f1"].append(val_f1)

        writer.add_scalars("Loss", {"train": loss, "val": val_loss}, global_step)
        writer.add_scalars("Acc", {"train": acc, "val": val_acc}, global_step)
        writer.add_scalars("F1", {"train": f1, "val": val_f1}, global_step)
        global_step += 1

        print(f"[P2 {epoch+1}/{EPOCHS}] Loss: {loss:.4f} Acc: {acc:.4f} F1: {f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            model.save_checkpoint(
                ckpt_dir / "best.pth", epoch, optimizer, scheduler,
                {"val_f1": val_f1, "val_acc": val_acc},
            )
            print(f"  => New best (F1: {val_f1:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.save_checkpoint(
            ckpt_dir / "latest.pth", epoch, optimizer, scheduler,
            {"val_f1": val_f1, "val_acc": val_acc},
        )

    writer.close()
    return history, best_f1, run_dir
