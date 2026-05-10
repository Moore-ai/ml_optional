import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from classify_Hyper_Kvasir.config import *
from classify_Hyper_Kvasir.data.transforms import train_transform, val_transform, test_transform


class LeafDataset(Dataset):
    """递归扫描根目录，将每个叶子目录作为一个类别。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        leaf_dirs = self._find_leaf_dirs(self.root)
        leaf_dirs = sorted(leaf_dirs, key=lambda d: d.name)

        for idx, leaf_dir in enumerate(leaf_dirs):
            class_name = leaf_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            for img_path in sorted(leaf_dir.glob("*.jpg")):
                self.samples.append((img_path, idx))

    @staticmethod
    def _find_leaf_dirs(root: Path) -> list[Path]:
        leaves: list[Path] = []
        for dirpath, dirnames, _filenames in os.walk(root):
            if not dirnames:
                leaves.append(Path(dirpath))
        return leaves

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img, label


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

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


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
    dataset = LeafDataset(DATA_ROOT)
    targets = np.array(dataset.targets)

    train_idx, val_idx, test_idx = _stratified_split(
        targets, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    )

    train_set = _SubsetWithTransform(dataset, train_idx, train_transform)
    val_set = _SubsetWithTransform(dataset, val_idx, val_transform)
    test_set = _SubsetWithTransform(dataset, test_idx, test_transform)

    class_counts = np.bincount(targets[train_idx])
    sample_weights = (1.0 / class_counts[targets[train_idx]]).tolist()
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
