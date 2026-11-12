"""Seed aggregation, privacy costs and comparison with the paper. All values in percentage points.

Definitions (CONTEXT.md §4):
  privacy cost      pi_k = A_k(non-private) - A_k(private), paired by seed
  privacy-cost gap  |pi_a - pi_b|: reported as |mean over seeds of (pi_a - pi_b)|, with the SE of that
                    per-seed difference. dp-testing's mean of per-seed |.| is kept as `gap_abs_mean`.
SE = SD / sqrt(n_seeds), matching the paper's ± convention.
"""

import numpy as np
import pandas as pd

from .config import GROUPS, METHOD_LABELS, METHODS, PAPER

RATES = ("accuracy", "fnr", "fpr", "tpr")


def to_frame(results: list[dict]) -> pd.DataFrame:
    """Long table: one row per (method, seed, group)."""
    rows = [{"method": r["method"], "seed": r["seed"], "group": name, **m}
            for r in results for name, m in r["evaluation"]["groups"].items()]
    df = pd.DataFrame(rows)
    for col in RATES:
        if col in df:
            df[col] = pd.to_numeric(df[col]) * 100
    return df


def run_info(results: list[dict]) -> pd.DataFrame:
    """One row per method: ε (δ), σ, final Z, number of seeds, source."""
    df = pd.DataFrame([{"method": r["method"], "epsilon": (r["privacy"] or {}).get("epsilon"),
                        "delta": (r["privacy"] or {}).get("delta"), "sigma": (r["privacy"] or {}).get("sigma"),
                        "final_Z": r["final_Z"], "source": r["source"]} for r in results])
    out = df.groupby("method", sort=False).agg(seeds=("method", "size"), epsilon=("epsilon", "mean"),
                                               delta=("delta", "first"), sigma=("sigma", "mean"),
                                               final_Z=("final_Z", "mean"), source=("source", "first"))
    out = out.reindex(METHODS).rename(index=METHOD_LABELS)
    out = out.assign(epsilon=out.epsilon.round(3), sigma=out.sigma.round(3), final_Z=out.final_Z.round(2),
                     delta=out.delta.map(lambda d: f"{d:.0e}" if pd.notna(d) else d))
    out = out.dropna(axis=1, how="all")  # σ is recorded only by this repo's runs (solved per run on Folktables)
    return out.astype(object).where(out.notna(), "–")


def privacy_costs(df: pd.DataFrame) -> pd.DataFrame:
    """[method, seed, group, pi] for each private method."""
    base = df[df.method == "nonprivate"][["seed", "group", "accuracy"]]
    priv = df[df.method != "nonprivate"].merge(base, on=["seed", "group"], suffixes=("", "_np"))
    return priv.assign(pi=priv.accuracy_np - priv.accuracy)[["method", "seed", "group", "pi"]]


def summarize(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """[method, group, mean, sd, se] of `value` across seeds."""
    g = df.groupby(["method", "group"], sort=False)[value]
    out = g.agg(mean="mean", sd="std", n="size").reset_index()
    return out.assign(se=out.sd / np.sqrt(out.n))


def gaps(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Privacy-cost gap between the dataset's two focus groups, per private method."""
    a, b = GROUPS[dataset]
    pi = privacy_costs(df).pivot_table(index=["method", "seed"], columns="group", values="pi")
    diff = (pi[a] - pi[b]).groupby("method")
    out = pd.DataFrame({"gap": diff.mean().abs(), "se": diff.std() / np.sqrt(diff.size()),
                        "gap_abs_mean": diff.apply(lambda d: d.abs().mean())})
    return out.reindex([m for m in METHODS if m in out.index])


def _fmt(mean, se=None, digits=1):
    if mean is None or pd.isna(mean):
        return "–"
    return f"{mean:.{digits}f}" if se is None or pd.isna(se) else f"{mean:.{digits}f} ± {se:.{digits}f}"


def comparison_table(results: list[dict], dataset: str) -> pd.DataFrame:
    """Paper-style table: ours vs Esipova et al., mean ± SE over seeds, in percentage points."""
    a, b = GROUPS[dataset]
    df = to_frame(results)
    acc = summarize(df, "accuracy").set_index(["method", "group"])
    pi = summarize(privacy_costs(df), "pi").set_index(["method", "group"])
    gap, info = gaps(df, dataset), run_info(results)
    paper = PAPER[dataset]
    cols = [f"Acc {a}", f"Acc {b}", f"π {a}", f"π {b}", f"Gap π {a},{b}", "ε"]
    rows = {}
    for m in METHODS:
        ours = [_fmt(acc.loc[(m, g), "mean"], acc.loc[(m, g), "se"]) for g in (a, b)]
        ours += [_fmt(pi.loc[(m, g), "mean"], pi.loc[(m, g), "se"]) if m != "nonprivate" else "–" for g in (a, b)]
        ours += [_fmt(gap.loc[m, "gap"], gap.loc[m, "se"]) if m in gap.index else "0 (by definition)",
                 _fmt(pd.to_numeric(info.loc[METHOD_LABELS[m], "epsilon"], errors="coerce"), digits=2)]
        p = paper[m]
        theirs = [_fmt(*p["acc"][g]) for g in (a, b)]
        theirs += [_fmt(*p["pi"][g]) if "pi" in p else "–" for g in (a, b)]
        theirs += [_fmt(*p["gap"]) if "gap" in p else "0 (by definition)", "–"]
        rows[(METHOD_LABELS[m], "Ours")] = ours
        rows[(METHOD_LABELS[m], paper["source"])] = theirs
    out = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    out.index = pd.MultiIndex.from_tuples(out.index, names=["Method", "Source"])
    return out


def settings_table(dataset: str) -> pd.DataFrame:
    """Hyperparameters of each method's preset (blank = not used)."""
    from .config import preset
    rows = {METHOD_LABELS[m]: preset(dataset, m).asdict() for m in METHODS}
    out = pd.DataFrame(rows).T.drop(columns=["dataset", "method"])
    out["delta"] = out["delta"].map(lambda d: f"{d:.0e}")
    if "target_epsilon" in out:  # Folktables: σ is solved per run for the target ε
        out.loc[out.target_epsilon.notna(), "sigma"] = "solved"
    return out.rename(columns={"sigma": "σ", "sigma2": "σ₂", "tau": "τ", "eta_Z": "η_Z", "Z0": "Z₀", "delta": "δ",
                               "target_epsilon": "target ε"}).fillna("")


def headline(results: list[dict], dataset: str) -> pd.Series:
    """Key numbers (means over seeds) for comparing two sets of runs, e.g. dp-testing vs a rerun."""
    a, b = GROUPS[dataset]
    df = to_frame(results)
    acc = df.groupby(["method", "group"]).accuracy.mean()
    gap, info = gaps(df, dataset), run_info(results)
    out = {f"{METHOD_LABELS[m]}: Acc {g}": acc[(m, g)] for m in METHODS for g in (a, b)}
    out |= {f"{METHOD_LABELS[m]}: gap": gap.loc[m, "gap"] for m in gap.index}
    out |= {f"{METHOD_LABELS[m]}: ε": float(info.loc[METHOD_LABELS[m], "epsilon"]) for m in METHODS[1:]}
    return pd.Series(out).round(2)


def gaps_by_setting(settings: dict) -> pd.DataFrame:
    """{name: (results, dataset)} -> [method, group=name, mean, se] of the privacy-cost gap, for plots.group_bars."""
    out = pd.concat({name: gaps(to_frame(r), ds) for name, (r, ds) in settings.items()}, names=["group", "method"])
    return out.reset_index().rename(columns={"gap": "mean"})


def size_sweep(runs_by_n: dict, dataset: str, key: str = "n_train") -> dict[str, pd.DataFrame]:
    """{n: results} -> per training-set size, mean ± SE over seeds of the privacy-cost gap ("gap"),
    and per method and group of π ("pi"), accuracy and FNR. Each table has a `key` column
    (pass key="epsilon" for {ε: results})."""
    def cat(f):
        return pd.concat({n: f(to_frame(r)) for n, r in runs_by_n.items()}, names=[key, None]).reset_index(level=0)
    return {"gap": cat(lambda df: gaps(df, dataset).reset_index().rename(columns={"gap": "mean"})),
            "pi": cat(lambda df: summarize(privacy_costs(df), "pi")),
            "accuracy": cat(lambda df: summarize(df, "accuracy")),
            "fnr": cat(lambda df: summarize(df, "fnr"))}


def base_rates(results: list[dict], dataset: str) -> pd.DataFrame:
    """Per group (means over seeds, %): the base rate, the accuracy of always predicting the negative class,
    the non-private model's headroom above that constant, and DP-SGD's accuracy, FNR and π.
    Tests our Adult observation that DP-SGD's π roughly equals that headroom (it falls back to the constant)."""
    df = to_frame(results)
    df = df.assign(base_rate=100 * (df.fn + df.tp) / df.n)
    mean = df.groupby(["method", "group"])[["accuracy", "fnr", "base_rate"]].mean()
    pi = privacy_costs(df).groupby(["method", "group"]).pi.mean()
    rows = {}
    for g in GROUPS[dataset]:
        npriv, dp = mean.loc[("nonprivate", g)], mean.loc[("dpsgd", g)]
        const = 100 - npriv.base_rate
        rows[g] = {"share positive": npriv.base_rate, "constant accuracy": const,
                   "non-private accuracy": npriv.accuracy, "headroom": npriv.accuracy - const,
                   "DP-SGD accuracy": dp.accuracy, "DP-SGD FNR": dp.fnr, "DP-SGD π": pi[("dpsgd", g)]}
    return pd.DataFrame(rows).T.round(1)


def setting_summary(settings: dict) -> pd.DataFrame:
    """{name: (results, dataset)} -> one row per setting: non-private accuracy per group, the privacy-cost gap
    (mean ± SE) per private method and Global-Adapt's π per group, in pp. Used to compare non-private baselines."""
    rows = {}
    for name, (results, dataset) in settings.items():
        df, (a, b) = to_frame(results), GROUPS[dataset]
        acc = df[df.method == "nonprivate"].groupby("group").accuracy.mean()
        pi = privacy_costs(df).groupby(["method", "group"]).pi.mean()
        g = gaps(df, dataset)
        rows[name] = {"groups": f"{a} / {b}", "non-private acc": f"{acc[a]:.1f} / {acc[b]:.1f}"} \
            | {f"{METHOD_LABELS[m]} gap": _fmt(g.loc[m, "gap"], g.loc[m, "se"], 2) for m in g.index} \
            | {"Global-Adapt π": " / ".join(f"{round(pi[('global_adapt', k)], 1) + 0.0:.1f}" for k in (a, b))}  # + 0.0: no "-0.0"
    return pd.DataFrame(rows).T
