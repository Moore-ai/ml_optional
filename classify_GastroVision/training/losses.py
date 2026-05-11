# pyright: reportPrivateImportUsage=false
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weights (tensor of shape [num_classes] or None).
        gamma: Focusing parameter. gamma=0 recovers standard CE loss.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_raw = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_raw)
        loss = ((1 - pt) ** self.gamma) * ce_raw
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()
