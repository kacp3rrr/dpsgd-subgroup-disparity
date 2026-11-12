"""Train (or reuse) multi-seed runs into results/<dataset>/.

    uv run python scripts/sweep.py --dataset adult                       # all methods, seeds 0-4
    uv run python scripts/sweep.py --dataset mnist --methods dpsgd --seeds 0 1 --device cuda
"""

import argparse

from dpdisparity.config import METHODS, preset
from dpdisparity.runner import run_or_load

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["mnist", "adult"])
parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--device", default="auto")
args = parser.parse_args()

for method in args.methods:
    for r in run_or_load(preset(args.dataset, method), args.seeds, args.device):
        eps = r["privacy"]["epsilon"] if r["privacy"] else None
        accs = {g: round(100 * m["accuracy"], 2) for g, m in r["evaluation"]["groups"].items()}
        print(f"{method} seed {r['seed']}: eps={eps and round(eps, 3)} acc={accs}", flush=True)
