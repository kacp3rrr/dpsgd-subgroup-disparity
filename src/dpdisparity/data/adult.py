"""UCI Adult, sex-balanced (Esipova et al., 2023, App. B.1, following Le Quy et al. 2022 / Xu et al. 2021).

Pipeline (ported unchanged from dp-testing):
48,842 raw rows -> drop rows with '?' -> 45,222 (30,527 male, 14,695 female) -> drop fnlwgt ->
standardize 5 numeric columns over the full complete-case pool (ddof=1) -> race to white/non-white ->
keep every woman and each man with p = 14695/30527 (~14.7k per sex) -> one-hot 6 categoricals ->
seeded shuffle, first 20% test. Sex stays an input feature (input_dim = 98, matching the paper's 91,650 params).
"""

import hashlib
import io
import math
import random
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

from ..config import DATA_DIR

URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
SHA256 = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
}
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
           "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
           "native_country", "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "education", "marital_status", "occupation", "relationship", "native_country"]
GROUP_NAMES = {0: "Male", 1: "Female"}


def download(data_dir=DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if not all((data_dir / f).exists() for f in SHA256):
        with urllib.request.urlopen(URL) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                for name in SHA256:
                    (data_dir / name).write_bytes(z.read(name))
    for name, expected in SHA256.items():
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: SHA-256 {actual} does not match the pinned {expected}")


def load_adult(seed: int, data_dir=DATA_DIR) -> dict:
    """Returns tensors {X_train, y_train, g_train, X_test, y_test, g_test, meta}; group 0 = Male, 1 = Female."""
    download(data_dir)
    read = dict(header=None, names=COLUMNS, skipinitialspace=True)
    df = pd.concat([pd.read_csv(data_dir / "adult.data", **read),
                    pd.read_csv(data_dir / "adult.test", skiprows=1, **read)], ignore_index=True)
    assert len(df) == 48_842
    df["income"] = df["income"].astype(str).str.rstrip(".")
    df = df.drop(columns="fnlwgt").replace("?", np.nan).dropna().reset_index(drop=True)
    n_male, n_female = int((df.sex == "Male").sum()), int((df.sex == "Female").sum())
    assert (len(df), n_male, n_female) == (45_222, 30_527, 14_695)

    df[NUMERIC] = (df[NUMERIC] - df[NUMERIC].mean()) / df[NUMERIC].std(ddof=1)
    df["income"] = (df["income"] == ">50K").astype(np.int64)
    df["sex"] = (df["sex"] == "Female").astype(np.int64)
    df["race"] = (df["race"] != "White").astype(np.int64)

    # Balance by sex. Same draw sequence as dp-testing: one rng.random() per row, in row order.
    rng = random.Random(seed)
    p_keep = np.where(df["sex"] == 1, 1.0, n_female / n_male)
    df = df[np.array([rng.random() for _ in range(len(df))]) <= p_keep].reset_index(drop=True)
    df = pd.get_dummies(df, columns=CATEGORICAL, dtype=np.float32)

    X = torch.tensor(df.drop(columns="income").to_numpy(np.float32))
    y = torch.tensor(df["income"].to_numpy(np.int64))
    g = torch.tensor(df["sex"].to_numpy(np.int64))
    order = torch.tensor(pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).to_numpy())
    n_test = math.floor(0.2 * len(df))
    test, train = order[:n_test], order[n_test:]
    return {
        "X_train": X[train], "y_train": y[train], "g_train": g[train],
        "X_test": X[test], "y_test": y[test], "g_test": g[test],
        "group_names": GROUP_NAMES,
        "meta": {"n_train": len(train), "n_test": n_test, "input_dim": X.shape[1],
                 "train_counts": torch.bincount(g[train], minlength=2).tolist(),
                 "test_counts": torch.bincount(g[test], minlength=2).tolist(),
                 "test_positive_rate": [float(y[test][g[test] == k].float().mean()) for k in (0, 1)]},
    }
