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
