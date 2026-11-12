"""Load results: frozen dp-testing runs (results/legacy/) or this repo's reruns (results/<dataset>/).

Legacy files are unmodified copies of dp-testing/results/{mnist,adult}/phase{1,2,4}_seed_{0-4}.json
(dp-testing commit 8ae750c). They are translated on read into the same shape `train.train` writes.
"""

import json
from dataclasses import replace
from pathlib import Path

from .config import FOLKTABLES_EPSILON, METHODS, RESULTS_DIR, RunCfg, preset
from .train import train

PHASE_METHOD = {1: "nonprivate", 2: "dpsgd", 4: "global_adapt"}


def result_path(cfg: RunCfg, seed: int) -> Path:
    """results/<dataset>/<method>_seed_<s>.json. Folktables nests by group and data variant,
    e.g. results/folktables/sex/, results/folktables/sex/balanced/, results/folktables/sex/n25000_eps1/.
    A learning rate other than the preset's adds `lr<lr>` (e.g. .../n5000_lr0.2/), so it never overwrites the preset run."""
    eps = cfg.target_epsilon not in (None, FOLKTABLES_EPSILON)
    lr = cfg.lr != preset(cfg.dataset, cfg.method).lr
    variant = "_".join(v for v in ("balanced" if cfg.balance else "", f"n{cfg.n_train}" if cfg.n_train else "",
                                   f"eps{cfg.target_epsilon:g}" if eps else "", f"lr{cfg.lr:g}" if lr else "") if v)
    return RESULTS_DIR / cfg.dataset.replace("folktables_", "folktables/") / variant / f"{cfg.method}_seed_{seed}.json"


def run_or_load(cfg: RunCfg, seeds=range(5), device: str = "auto", verbose: bool = True) -> list[dict]:
    """Reuse results/<dataset>/<method>_seed_<s>.json when its config matches; otherwise train and save."""
    out = []
    for seed in seeds:
        path = result_path(cfg, seed)
        if path.exists() and json.loads(path.read_text())["config"] == cfg.asdict():
            out.append(json.loads(path.read_text()))
            continue
        if verbose:
            print(f"training {path.relative_to(RESULTS_DIR).parent}/{cfg.method} seed {seed} ...", flush=True)
        result = train(cfg, seed, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1))
        out.append(result)
    return out


def load(dataset: str, source: str = "legacy", seeds=range(5), device: str = "auto", baseline_lr: float | None = None,
         **options) -> list[dict]:
    """All three methods x seeds. source="rerun" trains whatever is missing. `options` override preset
    fields, e.g. balance=True, n_train=5000 or target_epsilon=1 on Folktables (non-private runs ignore ε).
    `baseline_lr` swaps in a non-private baseline trained at that learning rate (e.g. Global-Adapt's 0.2)."""
    if source == "legacy":
        return load_legacy(dataset)
    out = []
    for m in METHODS:
        cfg = replace(preset(dataset, m), **{k: v for k, v in options.items() if m != "nonprivate" or k != "target_epsilon"})
        if m == "nonprivate" and baseline_lr is not None:
            cfg = replace(cfg, lr=baseline_lr)
        out += run_or_load(cfg, seeds, device)
    return out


def load_legacy(dataset: str) -> list[dict]:
    translate = {"mnist": _legacy_mnist, "adult": _legacy_adult}[dataset]
    files = sorted((RESULTS_DIR / "legacy" / dataset).glob("phase*_seed_*.json"))
    return [translate(json.loads(f.read_text())) for f in files]


def _legacy_common(r: dict, dataset: str) -> dict:
    method = PHASE_METHOD[r["phase"]]
    eps = r.get("achieved_epsilon")
    return {
        "dataset": dataset, "method": method, "seed": r["seed"], "source": "dp-testing",
        "config": preset(dataset, method).asdict(), "final_Z": r.get("final_Z"),
        "privacy": None if eps is None else {"epsilon": eps, "delta": r.get("target_delta", r.get("delta")),
                                             "accountant": "rdp", "q": r.get("sample_rate_q"), "poisson": True},
        "provenance": {**r.get("software_versions", {}), "device": r.get("device")},
    }


def _legacy_mnist(r: dict) -> dict:
    groups = {str(k): {"n": n, "accuracy": a, "loss": l} for k, (n, a, l) in
              enumerate(zip(r["test_counts_by_digit"], r["accuracy_by_digit"], r["loss_by_digit"]))}
    return _legacy_common(r, "mnist") | {
        "train_counts": r["training_counts_by_digit"],
        "evaluation": {"overall": {"accuracy": r["overall_accuracy"], "loss": r["overall_loss"]}, "groups": groups},
    }


def _legacy_adult(r: dict) -> dict:
    ev = r["evaluation"]
    keys = ("tn", "fp", "fn", "tp", "fnr", "fpr", "tpr")
    groups = {name: {"n": m["total_samples"], "accuracy": m["accuracy"], "loss": ev["group_losses"][name]}
                    | {k: m[k] for k in keys} for name, m in ev["groups"].items()}
    return _legacy_common(r, "adult") | {
        "train_counts": None,
        "evaluation": {"overall": {"accuracy": ev["overall"]["accuracy"], "loss": ev["overall_loss"]}, "groups": groups},
    }
