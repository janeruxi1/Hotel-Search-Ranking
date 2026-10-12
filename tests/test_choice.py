import numpy as np
import pandas as pd
import pytest

from hotelrank import choice

TRUTH = {"a": 1.0, "b": -0.5, "c": 0.25}


def _with_positions(df):
    """Give each search's rows the displayed slots 1, 2, 3, 4, 6, ... (reserved slots skipped)."""
    from hotelrank.synthetic import display_slots

    df = df.copy()
    rank = df.groupby("srch_id").cumcount().to_numpy()
    df["position"] = display_slots(int(rank.max()) + 1)[rank]
    return df


@pytest.fixture(scope="module")
def sim():
    return choice.simulate(3000, TRUTH, seed=1, size=(5, 20))


def test_fit_recovers_planted_coefficients(sim):
    f = choice.fit(choice.build(sim, "chosen", attributes=list(TRUTH), position_dummies=False))
    assert f.converged
    for nm, b in TRUTH.items():
        assert abs(f.coef(nm) - b) < 3 * f.se[f.names.index(nm)], nm


def test_matches_statsmodels_with_position_levels(sim):
    sm = pytest.importorskip("statsmodels.discrete.conditional_models")
    # short lists so every position level is chosen at least sometimes (no separation)
    df = _with_positions(choice.simulate(1500, {"a": 1.0, "b": -0.5}, seed=2, size=(5, 9)))
    f = choice.fit(choice.build(df, "chosen", attributes=["a", "b"]))
    dd = df.sort_values(["srch_id", "position"])
    dummies = pd.get_dummies(dd["position"], prefix="pos", dtype=float).drop(columns="pos_1")
    m = sm.ConditionalLogit(dd["chosen"].astype(float), pd.concat([dd[["a", "b"]], dummies], axis=1),
                            groups=dd["srch_id"]).fit(disp=0)
    assert np.allclose(f.beta, m.params.reindex(f.names).to_numpy(), atol=5e-3)
    assert np.allclose(f.se, m.bse.reindex(f.names).to_numpy(), rtol=1e-2)
    assert f.loglik >= m.llf - 1e-6  # Newton reaches at least statsmodels' optimum


def test_build_keeps_only_single_choice_searches():
    df = pd.DataFrame({"srch_id": [1, 1, 2, 2, 3, 3], "position": [1, 2, 1, 2, 1, 2],
                       "y": [1, 0, 1, 1, 0, 0], "a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    d = choice.build(df, "y", attributes=["a"], position_dummies=False)
    assert d.n_searches == 1 and d.search_ids.tolist() == [1]


def test_reserved_slots_are_not_position_levels(sim):
    d = choice.build(_with_positions(sim), "chosen", attributes=["a"])
    assert 5 not in d.levels and 1 not in d.levels


def test_ratio_with_se_delta_method(sim):
    f = choice.fit(choice.build(sim, "chosen", attributes=list(TRUTH), position_dummies=False))
    v, se = choice.ratio_with_se(f, "a", "b")
    assert v == pytest.approx(-f.coef("a") / f.coef("b"))
    draws = np.random.default_rng(0).multivariate_normal(f.beta, f.cov, 20000)
    sim_ratio = -draws[:, 0] / draws[:, 1]
    assert se == pytest.approx(sim_ratio.std(), rel=0.1)


def test_price_equivalent_arithmetic():
    eq = choice.price_equivalent(np.log(2), 100.0)
    assert eq["pct_price_cut"] == pytest.approx(50.0) and eq["usd"] == pytest.approx(50.0)


def test_subset_and_holdout(sim):
    d = choice.build(sim, "chosen", attributes=list(TRUTH), position_dummies=False)
    mask = (d.search_ids % 5) == 0
    test = choice.subset(d, mask)
    assert test.n_searches == mask.sum() and test.chosen.sum() == mask.sum()
    f = choice.fit(choice.subset(d, ~mask))
    h = choice.holdout_metrics(test, f.beta)
    assert h["top1_hit_rate"] > h["uniform_top1"] and 0 < h["mcfadden_r2"] < 1
    cal = choice.calibration(test, f.beta)
    assert np.allclose(cal["pred"], cal["obs"], atol=0.05)


def test_hotel_mean_price_is_per_hotel():
    df = pd.DataFrame({"prop_id": [1, 1, 2], "price_usd": [100.0, 400.0, 50.0]})
    out = choice.add_hotel_mean_price(df)
    assert out.loc[0, "hotel_mean_log_price"] == pytest.approx(np.log(200.0), rel=1e-6)
    assert out.loc[2, "hotel_mean_log_price"] == pytest.approx(np.log(50.0), rel=1e-6)
