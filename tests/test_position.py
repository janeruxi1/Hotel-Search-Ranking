import numpy as np
import pandas as pd
import pytest

from hotelrank import audit, position
from hotelrank.data import RESERVED_SLOTS
from hotelrank.synthetic import display_slots


def _linear_panel(n_searches=3000, seed=0, slope=-0.004, beta_x=0.05):
    """y = search effect + slope * position + beta_x * x + noise, every search shows slots 1..30."""
    rng = np.random.default_rng(seed)
    slots = display_slots(26)
    rows = []
    for s in range(n_searches):
        fe = rng.normal(0.2, 0.05)
        x = rng.normal(0, 1, len(slots))
        y = fe + slope * slots + beta_x * x + rng.normal(0, 0.1, len(slots))
        rows.append(pd.DataFrame({"srch_id": s, "position": slots, "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_group_sum_matches_bincount_contiguous_and_not():
    codes = np.array([0, 0, 1, 1, 1, 2, 0])  # last row breaks contiguity
    v = np.arange(7, dtype=float)
    expected = np.bincount(codes, weights=v)
    assert np.allclose(position._group_sum(v, codes, 3), expected)
    m = np.column_stack([v, 2 * v])
    assert np.allclose(position._group_sum(m, codes, 3)[:, 1], 2 * expected)


def test_lpm_recovers_linear_slope_and_control():
    d = _linear_panel()
    fit = position.within_search_lpm(d, "y", ["x"], position="linear", max_position=30)
    b = dict(zip(fit["names"], fit["beta"]))
    se = dict(zip(fit["names"], fit["se"]))
    assert abs(b["position"] - (-0.004)) < 3 * se["position"]
    assert abs(b["x"] - 0.05) < 3 * se["x"]


def test_lpm_dummies_trace_the_line():
    d = _linear_panel(n_searches=2000, seed=1)
    curve = position.lpm_curve(position.within_search_lpm(d, "y", None, max_position=30)).set_index("position")
    for k in (10, 20, 30):
        assert curve.loc[k, "diff"] == pytest.approx(-0.004 * (k - 1), abs=4 * curve.loc[k, "se"] + 1e-4)


def test_lpm_drops_constant_control():
    d = _linear_panel(n_searches=200)
    d["const"] = 1.0
    fit = position.within_search_lpm(d, "y", ["x", "const"], position="linear", max_position=30)
    assert fit["dropped"] == ["const"] and "const" not in fit["names"]


def test_balanced_panel_keeps_complete_searches_only():
    d = _linear_panel(n_searches=10)
    d = d[~((d["srch_id"] == 3) & (d["position"] == 12))]  # search 3 misses a slot
    p = position.balanced_panel(d, 30)
    assert 3 not in set(p["srch_id"]) and p["srch_id"].nunique() == 9
    assert not set(RESERVED_SLOTS) & set(p["position"])


def test_rate_by_position_balanced_bootstrap(synth):
    flags = audit.price_flags(synth, audit.per_search(synth)).reset_index().rename(columns={"index": "srch_id"})
    x = position.analysis_sample(synth, flags)
    panel = position.balanced_panel(x, 30)
    c = position.rate_by_position(panel, "click_bool", n_boot=200, seed=0, max_position=30).set_index("position")
    direct = panel.groupby("position")["click_bool"].mean()
    assert np.allclose(c["rate"], direct.loc[c.index])
    assert ((c["lo"] <= c["rate"]) & (c["rate"] <= c["hi"])).all()
    assert c.loc[1, "rel_to_1"] == 1.0


def test_rate_by_position_unbalanced_path(synth):
    flags = audit.price_flags(synth, audit.per_search(synth)).reset_index().rename(columns={"index": "srch_id"})
    x = position.analysis_sample(synth, flags)
    c = position.rate_by_position(x[x["random_bool"].astype(bool)], "click_bool", n_boot=50, seed=0)
    assert ((c["lo"] <= c["rate"]) & (c["rate"] <= c["hi"])).all()


def test_ratio_summary_inverts_relative_rate():
    c = pd.DataFrame({"position": [1, 10], "rel_to_1": [1.0, 0.25], "rel_lo": [1.0, 0.2], "rel_hi": [1.0, 0.5]})
    r = position.ratio_summary(c, ks=(10,))
    assert r["pos1_over_pos10"] == {"ratio": 4.0, "lo": 2.0, "hi": 5.0}


def test_analysis_sample_drops_flagged_searches_and_reserved_slots(synth):
    flags = audit.price_flags(synth, audit.per_search(synth)).reset_index().rename(columns={"index": "srch_id"})
    x = position.analysis_sample(synth, flags)
    sorted_ids = set(flags.loc[flags["price_sorted"].fillna(False).astype(bool), "srch_id"])
    assert not sorted_ids & set(x["srch_id"])
    assert {"rel_log_price", "no_reviews"} <= set(x.columns)
    assert x.groupby("srch_id")["rel_log_price"].mean().abs().max() < 1e-5


def test_bias_factor_point_and_interval(synth):
    flags = audit.price_flags(synth, audit.per_search(synth)).reset_index().rename(columns={"index": "srch_id"})
    x = position.analysis_sample(synth, flags)
    arms = {a: position.balanced_panel(x[x["random_bool"].astype(bool) == r], 30)
            for a, r in (("algorithm", False), ("random", True))}
    c = {a: position.rate_by_position(d, "click_bool", n_boot=200, seed=3, max_position=30) for a, d in arms.items()}
    b = position.bias_factor(c["algorithm"], c["random"], 10)
    naive = 1 / c["algorithm"].set_index("position").loc[10, "rel_to_1"]
    rand = 1 / c["random"].set_index("position").loc[10, "rel_to_1"]
    assert b["factor"] == pytest.approx(naive / rand, rel=1e-3)
    assert b["lo"] <= b["factor"] <= b["hi"]
