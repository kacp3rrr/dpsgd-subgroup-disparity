"""Load results: frozen dp-testing runs (results/legacy/) or this repo's reruns (results/<dataset>/).

Legacy files are unmodified copies of dp-testing/results/{mnist,adult}/phase{1,2,4}_seed_{0-4}.json
(dp-testing commit 8ae750c). They are translated on read into the same shape `train.train` writes.
"""

import json

from .config import METHODS, RESULTS_DIR, RunCfg, preset
from .train import train

PHASE_METHOD = {1: "nonprivate", 2: "dpsgd", 4: "global_adapt"}


def run_or_load(cfg: RunCfg, seeds=range(5), device: str = "auto", verbose: bool = True) -> list[dict]:
    """Reuse results/<dataset>/<method>_seed_<s>.json when its config matches; otherwise train and save."""
    out = []
    for seed in seeds:
        path = RESULTS_DIR / cfg.dataset / f"{cfg.method}_seed_{seed}.json"
        if path.exists() and json.loads(path.read_text())["config"] == cfg.asdict():
            out.append(json.loads(path.read_text()))
            continue
        if verbose:
            print(f"training {cfg.dataset}/{cfg.method} seed {seed} ...", flush=True)
        result = train(cfg, seed, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1))
        out.append(result)
    return out


def load(dataset: str, source: str = "legacy", seeds=range(5), device: str = "auto") -> list[dict]:
    """All three methods x seeds. source="rerun" trains whatever is missing."""
    if source == "legacy":
        return load_legacy(dataset)
    return [r for m in METHODS for r in run_or_load(preset(dataset, m), seeds, device)]


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
