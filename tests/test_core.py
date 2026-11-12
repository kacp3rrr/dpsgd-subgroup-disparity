import pytest
import torch

from dpdisparity import analysis
from dpdisparity.config import DATA_DIR
from dpdisparity.global_adapt import scale_factors
from dpdisparity.metrics import binary_rates


def test_global_adapt_scaling():
    f = scale_factors(torch.tensor([0.5, 10.0, 50.0, 200.0]), C=1.0, Z=50.0)
    assert torch.allclose(f[:3], torch.tensor(1 / 50))  # at or below Z: one shared factor C/Z
    assert torch.isclose(f[3] * 200.0, torch.tensor(1.0), atol=1e-6)  # above Z: clipped to norm C


def test_binary_rates():
    r = binary_rates(torch.tensor([0, 0, 1, 1, 1]), torch.tensor([0, 1, 0, 1, 1]))
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 1, 1, 2)
    assert r["fnr"] == pytest.approx(1 / 3)
    assert binary_rates(torch.tensor([0]), torch.tensor([0]))["fnr"] is None


def _result(method, seed, acc2, acc8):
    groups = {"2": {"accuracy": acc2}, "8": {"accuracy": acc8}}
    return {"method": method, "seed": seed, "evaluation": {"groups": groups}}


def test_privacy_cost_gap_is_not_raw_gap():
    # CONTEXT.md worked example: pi_2 = 8.70, pi_8 = 61.43 -> gap 52.73, while the raw accuracy gap is 64.1.
    results = [_result("nonprivate", 0, 0.9773, 0.8632), _result("dpsgd", 0, 0.8903, 0.2489)]
    df = analysis.to_frame(results)
    pi = analysis.privacy_costs(df).set_index("group").pi
    assert pi["2"] == pytest.approx(8.70) and pi["8"] == pytest.approx(61.43)
    assert analysis.gaps(df, "mnist").loc["dpsgd", "gap"] == pytest.approx(52.73)


@pytest.mark.skipif(not (DATA_DIR / "adult.data").exists(), reason="Adult not downloaded")
def test_adult_split_matches_dp_testing():
    from dpdisparity.data import load_adult
    d = load_adult(seed=0)
    assert d["meta"]["input_dim"] == 98
    assert d["meta"]["train_counts"] == [11719, 11765]  # dp-testing pairing_seed_0.json
    assert len(d["y_test"]) == 5871


@pytest.mark.parametrize("q", [1 / 20, 1 / 92, 1 / 612])  # Folktables n = 5k, Adult, full CA
def test_sigma_solve_reaches_target_epsilon(q):
    from dataclasses import replace

    from opacus.accountants import RDPAccountant

    from dpdisparity.config import preset
    from dpdisparity.train import noise_multiplier
    cfg = replace(preset("folktables_sex", "dpsgd"), n_train=None)
    sigma = noise_multiplier(cfg, q)
    acc = RDPAccountant()
    acc.history = [(sigma, q, cfg.epochs * int(1 / q))]  # the steps the Poisson sampler takes
    assert abs(acc.get_epsilon(cfg.delta) - cfg.target_epsilon) < 0.05
    assert noise_multiplier(preset("adult", "dpsgd"), q) == 1.0  # fixed-σ presets are untouched


@pytest.mark.skipif(not (DATA_DIR / "folktables/2018/1-Year/psam_p06.csv").exists(), reason="ACS CA 2018 not downloaded")
def test_folktables_loader():
    from dpdisparity.data import load
    full = load("folktables_sex", seed=0)
    assert full["meta"]["n_train"] + full["meta"]["n_test"] == 195_665
    assert full["meta"]["n_test"] == 195_665 // 5
    assert torch.equal(full["X_train"], load("folktables_sex", seed=0)["X_train"])  # deterministic
    sub = load("folktables_sex", seed=0, n_train=5000)
    assert sub["meta"]["n_train"] == 5000 and sum(sub["meta"]["train_counts"]) == 5000
    assert torch.equal(sub["X_test"], full["X_test"]) and torch.equal(sub["y_test"], full["y_test"])  # test set fixed
    share = lambda d: d["meta"]["train_counts"][1] / d["meta"]["n_train"]
    assert abs(share(sub) - share(full)) < 1e-3  # stratified by group
    bal = load("folktables_sex", seed=0, balance=True)
    c = bal["meta"]["train_counts"]
    assert abs(c[0] - c[1]) / sum(c) < 0.02 and bal["meta"]["input_dim"] == full["meta"]["input_dim"]
    race = load("folktables_race", seed=0)
    assert race["meta"]["n_train"] + race["meta"]["n_test"] == 129_563 and race["group_names"] == {0: "White", 1: "Black"}


def test_result_paths_keep_variants_apart():
    from dataclasses import replace

    from dpdisparity.config import preset
    from dpdisparity.runner import result_path
    base = preset("folktables_sex", "nonprivate")
    paths = {result_path(c, 0) for c in (base, replace(base, lr=0.2), replace(base, n_train=5000),
                                        replace(base, n_train=5000, lr=0.2), replace(base, balance=True))}
    assert len(paths) == 5  # an override never overwrites the preset run's file
    assert result_path(preset("adult", "dpsgd"), 0).parent.name == "adult"  # MNIST/Adult paths unchanged
