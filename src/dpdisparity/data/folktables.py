"""Folktables ACSIncome (Ding et al., 2021): ACS 1-Year person survey, label income > $50K.

Pipeline, mirroring data/adult.py where it can:
ACS PUMS person file -> folktables' ACSIncome filter (age > 16, income > $100, hours > 0, weight >= 1;
CA 2018: 195,665 people) -> [race only: keep White (RAC1P 1) and Black (RAC1P 2)] -> standardize AGEP, WKHP
over the full pool (ddof=1, as Adult) -> SEX to 0/1 (stays an input, as on Adult) -> one-hot the other 7
categoricals (OCCP, POBP, ... give ~815 inputs) -> [balance: keep every member of the smaller group and each
of the larger with p = n_small / n_large, as Adult] -> seeded shuffle, first 20% test -> [n_train: stratified
by group, seeded, from this seed's training split only, so the test set is the same for every n].
One-hot encoding runs before balancing so input_dim is fixed per group variant (Adult encodes after).
"""

import functools
import hashlib

import numpy as np
import pandas as pd
import torch
from folktables import ACSDataSource, ACSIncome
from folktables.load_acs import _STATE_CODES

from ..config import DATA_DIR

NUMERIC = ["AGEP", "WKHP"]
CATEGORICAL = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "RAC1P"]
GROUP_NAMES = {"sex": {0: "Male", 1: "Female"}, "race": {0: "White", 1: "Black"}}
SHA256 = {("CA", 2018): "dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0"}
# (rows, group 0, group 1) after the ACSIncome filter, for the pinned files.
COUNTS = {("CA", 2018, "sex"): (195_665, 103_311, 92_354), ("CA", 2018, "race"): (129_563, 121_006, 8_557)}


@functools.lru_cache(maxsize=2)
def _pool(state: str, year: int, group: str, data_dir: str):
    """Encoded full pool (X float32, y, g) for one state/year/group variant; cached across runs in a sweep."""
    root = f"{data_dir}/folktables"
    raw = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=root).get_data(
        states=[state], download=True)
    if (state, year) in SHA256:
        csv = f"{root}/{year}/1-Year/psam_p{_STATE_CODES[state]}.csv"
        with open(csv, "rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != SHA256[(state, year)]:
            raise ValueError(f"{csv}: SHA-256 {actual} does not match the pinned {SHA256[(state, year)]}")
    df, label, _ = ACSIncome.df_to_pandas(raw)
    y = label["PINCP"].to_numpy(np.int64)
    if group == "race":
        keep = df["RAC1P"].isin([1, 2]).to_numpy()
        df, y = df[keep], y[keep]
    g = ((df["SEX"] == 2) if group == "sex" else (df["RAC1P"] == 2)).to_numpy(np.int64)
    if (state, year, group) in COUNTS:
        assert (len(g), int((g == 0).sum()), int((g == 1).sum())) == COUNTS[(state, year, group)]

    df = df.copy()
    df[NUMERIC] = (df[NUMERIC] - df[NUMERIC].mean()) / df[NUMERIC].std(ddof=1)
    df["SEX"] = (df["SEX"] == 2).astype(np.float32)
    X = pd.get_dummies(df, columns=CATEGORICAL, dtype=np.float32).to_numpy(np.float32)
    return X, y, g


def load_folktables(seed: int, state: str = "CA", year: int = 2018, group: str = "sex", balance: bool = False,
                    n_train: int | None = None, data_dir=DATA_DIR) -> dict:
    """Same dict as load_adult; group 0/1 = Male/Female (sex) or White/Black (race)."""
    X, y, g = _pool(state, year, group, str(data_dir))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if balance:
        counts = np.bincount(g)
        small = counts.argmin()
        idx = idx[(g == small) | (rng.random(len(g)) < counts[small] / counts.max())]
    idx = rng.permutation(idx)
    n_test = len(idx) // 5
    test, train = idx[:n_test], idx[n_test:]
    if n_train is not None:
        # `train` is already in seeded random order, so a prefix per group is a seeded draw, and the draws
        # for different n are nested. Group counts follow the training split's shares.
        share = np.bincount(g[train], minlength=2) / len(train)
        take = np.round(share * n_train).astype(int)
        take[-1] = n_train - take[:-1].sum()
        train = np.concatenate([train[g[train] == k][:m] for k, m in enumerate(take)])

    t = lambda a, i: torch.from_numpy(a[i])
    return {
        "X_train": t(X, train), "y_train": t(y, train), "g_train": t(g, train),
        "X_test": t(X, test), "y_test": t(y, test), "g_test": t(g, test),
        "group_names": GROUP_NAMES[group],
        "meta": {"n_train": len(train), "n_test": n_test, "input_dim": X.shape[1],
                 "train_counts": np.bincount(g[train], minlength=2).tolist(),
                 "test_counts": np.bincount(g[test], minlength=2).tolist(),
                 "test_positive_rate": [float(y[test][g[test] == k].mean()) for k in (0, 1)]},
    }
