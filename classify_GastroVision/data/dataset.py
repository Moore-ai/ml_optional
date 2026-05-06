import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from config import *
from data.transforms import train_transform, val_transform, test_transform


class _SubsetWithTransform(Subset):
    """支持独立 transform 的 Subset。"""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def _stratified_split(labels, train_ratio, val_ratio, test_ratio):
    """三层分层拆分，极小类别时回退到随机拆分。"""
    n = len(labels)
    indices = np.arange(n)

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_ratio),
        stratify=labels, random_state=42,
    )
    temp_labels = labels[temp_idx]
    has_small = np.any(np.bincount(temp_labels) < 2)

    if has_small:
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
        )
    else:
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5,
            stratify=temp_labels, random_state=42,
        )
    return train_idx, val_idx, test_idx


def get_dataloaders():
    """返回 (train_loader, val_loader, test_loader, class_names)。"""
    dataset = ImageFolder(str(DATA_ROOT))
    targets = np.array(dataset.targets)

    train_idx, val_idx, test_idx = _stratified_split(
        targets, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    )

    train_set = _SubsetWithTransform(dataset, train_idx, train_transform)
    val_set = _SubsetWithTransform(dataset, val_idx, val_transform)
    test_set = _SubsetWithTransform(dataset, test_idx, test_transform)

    # WeightedRandomSampler: 样本权重与类别频率成反比
    class_counts = np.bincount(targets[train_idx])
    sample_weights = 1.0 / class_counts[targets[train_idx]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_set, BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, dataset.classes
