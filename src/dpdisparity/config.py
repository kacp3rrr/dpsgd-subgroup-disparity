"""Paper hyperparameters (Esipova et al., 2023, App. B) and published reference numbers."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

METHODS = ("nonprivate", "dpsgd", "global_adapt")
METHOD_LABELS = {"nonprivate": "Non-private", "dpsgd": "DP-SGD", "global_adapt": "Global-Adapt"}


@dataclass(frozen=True)
class RunCfg:
    dataset: str
    method: str
    epochs: int
    lr: float
    batch_size: int = 256
    sigma: float | None = None   # gradient noise multiplier
    C: float | None = None       # clipping norm
    Z0: float | None = None      # Global-Adapt: initial upper bound on gradient norms
    tau: float | None = None     # Global-Adapt: count threshold (fraction of Z)
    sigma2: float | None = None  # Global-Adapt: noise multiplier of the private count
    eta_Z: float | None = None   # Global-Adapt: learning rate of Z
    delta: float = 1e-6
    target_epsilon: float | None = None  # if set, sigma is solved per run to reach this ε (SPEC §10)
    balance: bool | None = None          # Folktables: subsample the groups to equal size, as on Adult
    n_train: int | None = None           # Folktables: stratified training subsample; the test set is unchanged

    def asdict(self) -> dict:
        # The Folktables-only fields are left out when unset, so existing MNIST/Adult result configs still match.
        return {k: v for k, v in asdict(self).items() if v is not None or k not in OPTIONAL_FIELDS}


OPTIONAL_FIELDS = ("target_epsilon", "balance", "n_train")


PRESETS = {
    ("mnist", "nonprivate"): RunCfg("mnist", "nonprivate", epochs=60, lr=0.01),
    ("mnist", "dpsgd"): RunCfg("mnist", "dpsgd", epochs=60, lr=0.01, sigma=0.8, C=1.0),
    ("mnist", "global_adapt"): RunCfg("mnist", "global_adapt", epochs=60, lr=0.1, sigma=0.8, C=1.0,
                                      Z0=50.0, tau=0.7, sigma2=10.0, eta_Z=0.1),
    ("adult", "nonprivate"): RunCfg("adult", "nonprivate", epochs=20, lr=0.01),
    ("adult", "dpsgd"): RunCfg("adult", "dpsgd", epochs=20, lr=0.01, sigma=1.0, C=0.5),
    ("adult", "global_adapt"): RunCfg("adult", "global_adapt", epochs=20, lr=0.2, sigma=1.0, C=0.5,
                                      Z0=50.0, tau=1.0, sigma2=10.0, eta_Z=0.1),
}

# Folktables ACSIncome: the Adult presets, but at a fixed ε = 3.41 (Adult's) with σ solved for each run's
# sampling rate, so runs of different size are comparable (SPEC §10).
FOLKTABLES_EPSILON = 3.41
PRESETS |= {(ds, m): replace(PRESETS[("adult", m)], dataset=ds,
                             **({} if m == "nonprivate" else {"sigma": None, "target_epsilon": FOLKTABLES_EPSILON}))
            for ds in ("folktables_sex", "folktables_race") for m in METHODS}


def preset(dataset: str, method: str) -> RunCfg:
    return PRESETS[(dataset, method)]


# The two groups compared on each dataset, in the paper's column order.
GROUPS = {"mnist": ("2", "8"), "adult": ("Male", "Female"),
          "folktables_sex": ("Male", "Female"), "folktables_race": ("White", "Black")}

# Published numbers, mean ± standard error over 5 seeds, in percentage points.
PAPER = {
    "mnist": {
        "source": "Esipova et al. (2023), Table 2",
        "nonprivate":   {"acc": {"2": (98.0, 0.1), "8": (84.3, 1.1)}},
        "dpsgd":        {"acc": {"2": (89.0, 0.1), "8": (26.3, 0.4)},
                         "pi": {"2": (8.9, 0.1), "8": (57.9, 1.3)}, "gap": (48.9, 1.3)},
        "global_adapt": {"acc": {"2": (92.0, 0.2), "8": (65.5, 1.2)},
                         "pi": {"2": (6.0, 0.2), "8": (18.8, 0.9)}, "gap": (12.8, 0.8)},
    },
    "adult": {
        "source": "Esipova et al. (2023), Table 4",
        "nonprivate":   {"acc": {"Male": (80.5, 0.4), "Female": (92.2, 0.1)}},
        "dpsgd":        {"acc": {"Male": (69.9, 0.4), "Female": (88.5, 0.1)},
                         "pi": {"Male": (10.6, 0.3), "Female": (3.6, 0.1)}, "gap": (6.9, 0.3)},
        "global_adapt": {"acc": {"Male": (80.7, 0.4), "Female": (92.3, 0.1)},
                         "pi": {"Male": (-0.1, 0.1), "Female": (-0.1, 0.1)}, "gap": (0.0, 0.1)},
    },
}
