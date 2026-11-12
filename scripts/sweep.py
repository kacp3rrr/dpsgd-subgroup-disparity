"""Train (or reuse) multi-seed runs into results/<dataset>/.

    uv run python scripts/sweep.py --dataset adult                       # all methods, seeds 0-4
    uv run python scripts/sweep.py --dataset mnist --methods dpsgd --seeds 0 1 --device cuda
    uv run python scripts/sweep.py --dataset folktables_sex --balance    # F1, balanced sexes
    uv run python scripts/sweep.py --dataset folktables_sex --n-train 5000 10000 25000   # F2 size sweep
    uv run python scripts/sweep.py --dataset folktables_sex --n-train 25000 --target-epsilon 0.5 1 2 8   # F4
    uv run python scripts/sweep.py --dataset folktables_sex --methods nonprivate --lr 0.2   # matched-lr baseline
"""

import argparse
from dataclasses import replace
from itertools import product

from dpdisparity.config import METHODS, preset
from dpdisparity.runner import run_or_load

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["mnist", "adult", "folktables_sex", "folktables_race"])
parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--device", default="auto")
parser.add_argument("--balance", action="store_const", const=True, help="Folktables: equal-size groups")
parser.add_argument("--n-train", nargs="+", type=int, default=[None], help="Folktables: training-set sizes")
parser.add_argument("--lr", type=float, help="override the preset learning rate (stored as a separate variant)")
parser.add_argument("--target-epsilon", nargs="+", type=float, default=[None], help="Folktables: ε values (private methods)")
args = parser.parse_args()

for target, n, method in product(args.target_epsilon, args.n_train, args.methods):
    cfg = replace(preset(args.dataset, method), balance=args.balance, n_train=n)
    if target is not None and method != "nonprivate":  # non-private runs have no ε
        cfg = replace(cfg, target_epsilon=target)
    if args.lr is not None:
        cfg = replace(cfg, lr=args.lr)
    for r in run_or_load(cfg, args.seeds, args.device):
        eps = r["privacy"]["epsilon"] if r["privacy"] else None
        accs = {g: round(100 * m["accuracy"], 2) for g, m in r["evaluation"]["groups"].items()}
        print(f"n={n} {method} seed {r['seed']}: eps={eps and round(eps, 3)} acc={accs}", flush=True)
