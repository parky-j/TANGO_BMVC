"""CIFAR-100 (ID) and OOD loaders, matching the paper's preprocessing.

All OOD images are resized to 32x32 and normalized with the **ID** dataset's
statistics -- this is what the paper does and what the released checkpoints
were trained/evaluated under. Changing the constants below changes the
numbers.

Auto-downloading OOD splits (torchvision): svhn, cifar10, mnist, dtd.
Manual splits (point --data-root at a directory holding them):
    tin        -> <root>/tiny-imagenet-200/val/images/*.JPEG
    places365  -> <root>/places365/**/*.jpg
    lsun_c     -> <root>/LSUN/**/*.{jpg,png}   (ODIN-curated subset)
"""
from __future__ import annotations

import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_STATS = {
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
}

AUTO_OOD = ("svhn", "cifar10", "cifar100", "mnist", "dtd")
MANUAL_OOD = ("tin", "places365", "lsun_c")


def _norm(id_dataset: str):
    return transforms.Normalize(*_STATS[id_dataset])


def train_transform(id_dataset: str = "cifar100"):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _norm(id_dataset),
    ])


def test_transform(id_dataset: str = "cifar100"):
    return transforms.Compose([
        transforms.ToTensor(),
        _norm(id_dataset),
    ])


def ood_transform(id_dataset: str = "cifar100"):
    """OOD is resized to 32x32 and normalized with the ID statistics."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        _norm(id_dataset),
    ])


class _ImageFolderFlat(Dataset):
    """Every image under `root` (recursive), label 0. For TIN/Places365/LSUN."""

    EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG")

    def __init__(self, root, transform=None, limit=None):
        self.files = sorted(
            f for f in glob(os.path.join(root, "**", "*"), recursive=True)
            if f.endswith(self.EXT))
        if not self.files:
            raise FileNotFoundError(
                f"No images under {root!r}. See tango/data.py docstring for the "
                f"expected layout of the manual OOD splits.")
        if limit:
            self.files = self.files[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_id_loaders(id_dataset="cifar100", data_root="./data",
                   batch_size=64, num_workers=2):
    """(train_loader, test_loader, num_classes) for the ID dataset."""
    cls = {"cifar100": datasets.CIFAR100, "cifar10": datasets.CIFAR10}[id_dataset]
    nc = 100 if id_dataset == "cifar100" else 10
    tr = cls(data_root, train=True, download=True,
             transform=train_transform(id_dataset))
    te = cls(data_root, train=False, download=True,
             transform=test_transform(id_dataset))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=False),
        DataLoader(te, batch_size=max(batch_size, 256), shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        nc,
    )


def get_ood_loader(name, id_dataset="cifar100", data_root="./data",
                   batch_size=256, num_workers=2, limit=None):
    """Test-split loader for one OOD dataset, preprocessed like the paper."""
    name = name.lower()
    t = ood_transform(id_dataset)

    if name == "svhn":
        ds = datasets.SVHN(data_root, split="test", download=True, transform=t)
    elif name == "cifar10":
        ds = datasets.CIFAR10(data_root, train=False, download=True, transform=t)
    elif name == "cifar100":
        ds = datasets.CIFAR100(data_root, train=False, download=True, transform=t)
    elif name == "mnist":
        # MNIST is 1-channel; expand to 3 before the shared transform.
        t3 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            _norm(id_dataset),
        ])
        ds = datasets.MNIST(data_root, train=False, download=True, transform=t3)
    elif name == "dtd":
        ds = datasets.DTD(data_root, split="test", download=True, transform=t)
    elif name in MANUAL_OOD:
        sub = {"tin": "tiny-imagenet-200/val",
               "places365": "places365",
               "lsun_c": "LSUN"}[name]
        ds = _ImageFolderFlat(os.path.join(data_root, sub), transform=t, limit=limit)
    else:
        raise ValueError(
            f"Unknown OOD split {name!r}. "
            f"auto-download: {AUTO_OOD}; manual: {MANUAL_OOD}")

    if limit and not isinstance(ds, _ImageFolderFlat):
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
