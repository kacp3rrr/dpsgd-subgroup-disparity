"""Imbalanced MNIST (Esipova et al., 2023, Sec. 6.1): keep each training 8 with probability 0.09."""

import random

import torch
from torchvision import datasets

from ..config import DATA_DIR


def load_mnist(seed: int, keep_eight: float = 0.09, data_dir=DATA_DIR) -> dict:
    """Returns tensors {X_train, y_train, g_train, X_test, y_test, g_test, meta}; group = digit.

    Images are scaled to [0, 1], with no normalization. The full test set is kept.
    """
    train = datasets.MNIST(str(data_dir), train=True, download=True)
    test = datasets.MNIST(str(data_dir), train=False, download=True)

    # Same draw sequence as dp-testing / the authors: one random.random() per 8, in index order.
    rng = random.Random(seed)
    keep = [i for i, y in enumerate(train.targets.tolist()) if y != 8 or rng.random() <= keep_eight]

    X_train = train.data[keep].unsqueeze(1).float() / 255.0
    y_train = train.targets[keep]
    X_test = test.data.unsqueeze(1).float() / 255.0
    y_test = test.targets
    return {
        "X_train": X_train, "y_train": y_train, "g_train": y_train,
        "X_test": X_test, "y_test": y_test, "g_test": y_test,
        "group_names": {d: str(d) for d in range(10)},
        "meta": {"n_train": len(keep), "train_counts": torch.bincount(y_train, minlength=10).tolist()},
    }
