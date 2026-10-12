import numpy as np
import pytest

from hotelrank import abtest


def test_two_proportion_n_matches_textbook():
    # 10% -> 12% at alpha 0.05, power 0.8: about 3,841 per arm (standard tables)
    assert abs(abtest.n_two_proportions(0.10, 0.20) - 3841) <= 5


def test_closed_form_power_matches_simulation():
    n = abtest.n_two_proportions(0.13, 0.05, alpha=0.05 / 3)
    sim = abtest.simulate_power(0.13, 0.05, n, alpha=0.05 / 3, n_sim=4000, seed=1)
    assert abs(sim - 0.8) < 0.03


def test_mde_inverts_n():
    n = abtest.n_two_proportions(0.13, 0.04)
    assert abtest.mde_two_proportions(0.13, n) == pytest.approx(0.04, rel=0.01)


def test_noninferiority_n_and_guard():
    n = abtest.n_noninferiority_mean(mean=100, sd=300, margin_rel=0.02)
    assert n == pytest.approx(2 * (300 * (1.959964 + 0.841621) / 2) ** 2, rel=1e-3)
    with pytest.raises(ValueError):
        abtest.n_noninferiority_mean(100, 300, 0.02, true_rel_diff=-0.03)


def test_capping_lowers_variance():
    rng = np.random.default_rng(0)
    v = np.where(rng.random(20000) < 0.13, rng.lognormal(5.5, 1.0, 20000), 0.0)
    raw, capped = abtest.per_search_value_stats(v), abtest.per_search_value_stats(v, 0.99)
    assert capped["sd"] < raw["sd"] and capped["cap"] > 0
