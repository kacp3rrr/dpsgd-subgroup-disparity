"""Paper architectures (Esipova et al., 2023, App. B.3 / B.6)."""

import torch
from torch import nn


def paper_cnn() -> nn.Module:
    """MNIST CNN: two 3x3 convs (32, 16 channels), tanh, no pooling. 97,114 params.

    Matches the authors' released script (net=cnn, hidden_channels=32,16); App. B.3 states
    80,522 params, which conflicts with that configuration.
    """
    return nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.Tanh(),
        nn.Conv2d(32, 16, 3), nn.Tanh(),
        nn.Flatten(), nn.Linear(16 * 24 * 24, 10),
    )


def paper_mlp(input_dim: int, hidden: int = 256) -> nn.Module:
    """Adult MLP: two hidden layers of 256, tanh. 91,650 params at input_dim=98 (matches App. B.3)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 2),
    )


def build_model(dataset: str, seed: int, input_dim: int | None = None) -> nn.Module:
    """Seeded init on CPU, so a seed gives the same initial weights for every method."""
    torch.manual_seed(seed)
    return paper_cnn() if dataset == "mnist" else paper_mlp(input_dim)
