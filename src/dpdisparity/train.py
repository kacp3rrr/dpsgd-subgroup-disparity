"""One training loop for every dataset and method: non-private SGD, DP-SGD, DPSGD-Global-Adapt."""

import platform
import subprocess
import time
from datetime import datetime, timezone

import opacus
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader, TensorDataset

from . import data, metrics
from .config import ROOT, RunCfg
from .global_adapt import GlobalAdaptOptimizer
from .models import build_model

# Training-RNG offsets per method (as dp-testing's Adult runs). The data subset and initial
# weights depend only on the seed, so methods are paired within a seed.
SEED_OFFSET = {"nonprivate": 1000, "dpsgd": 2000, "global_adapt": 4000}


def resolve_device(device: str = "auto") -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def noise_multiplier(cfg: RunCfg, q: float) -> float:
    """cfg.sigma, or the σ that reaches cfg.target_epsilon over the steps Opacus will take (int(1/q) per epoch)."""
    if cfg.target_epsilon is None:
        return cfg.sigma
    return get_noise_multiplier(target_epsilon=cfg.target_epsilon, target_delta=cfg.delta, sample_rate=q,
                                steps=cfg.epochs * int(1 / q), accountant="rdp")


def train(cfg: RunCfg, seed: int, device: str = "auto") -> dict:
    device = resolve_device(device)
    d = data.load(cfg.dataset, seed, cfg.balance, cfg.n_train)
    model = build_model(cfg.dataset, seed, d["meta"].get("input_dim")).to(device)
    train_ds = TensorDataset(d["X_train"], d["y_train"])
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0)

    private = cfg.method != "nonprivate"
    if private:
        model = GradSampleModule(model)
        loader = DPDataLoader.from_data_loader(loader)  # Poisson sampling, q = 1 / len(loader)
        q = loader.sample_rate
        sigma = noise_multiplier(cfg, q)
        kw = dict(noise_multiplier=sigma, max_grad_norm=cfg.C, expected_batch_size=int(len(train_ds) * q))
        if cfg.method == "global_adapt":
            optimizer = GlobalAdaptOptimizer(optimizer, Z0=cfg.Z0, tau=cfg.tau, sigma2=cfg.sigma2, eta_Z=cfg.eta_Z, **kw)
        else:
            optimizer = DPOptimizer(optimizer, **kw)
        accountant = RDPAccountant()
        optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=q))

    torch.manual_seed(seed + SEED_OFFSET[cfg.method])
    history, steps, start = [], 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum, n, steps = loss_sum + loss.item() * len(y), n + len(y), steps + 1
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n, 1)}
                       | ({"Z": optimizer.Z} if cfg.method == "global_adapt" else {}))

    result = {
        "dataset": cfg.dataset, "method": cfg.method, "seed": seed, "source": "rerun",
        "config": cfg.asdict(), "steps": steps, "train_counts": d["meta"]["train_counts"],
        "history": history, "privacy": None, "final_Z": None,
    }
    if private:
        if cfg.method == "global_adapt":
            for _ in range(optimizer.count_releases):  # compose the private count releases
                accountant.step(noise_multiplier=cfg.sigma2, sample_rate=q)
            result["final_Z"] = optimizer.Z
        result["privacy"] = {"epsilon": accountant.get_epsilon(cfg.delta), "delta": cfg.delta,
                             "accountant": "rdp", "q": q, "poisson": True,
                             "sigma": sigma}
        model = model._module
    result["evaluation"] = metrics.evaluate(model, d["X_test"], d["y_test"], d["g_test"], d["group_names"], device)
    result["provenance"] = provenance(device) | {"elapsed_s": round(time.time() - start, 1)}
    return result


def provenance(device: str) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        sha = None
    return {"python": platform.python_version(), "torch": torch.__version__, "opacus": opacus.__version__,
            "cuda": torch.version.cuda, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "git_sha": sha, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
