import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from classify_GastroVision.config import NUM_CLASSES


class GastroClassifier(nn.Module):
    """EfficientNetV2-S + 自定义 1280→512→27 分类头。"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)
        in_features: int = self.backbone.classifier[1].in_features  # type: ignore[assignment]

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def save_checkpoint(self, path, epoch, optimizer, scheduler, metrics):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
