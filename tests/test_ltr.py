"""Learning to rank (notebook 06): metrics, the position correction, features and the ranker."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hotelrank import choice, ltr, policy, position


def _blocks(sizes):
    search = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return search, starts


# ------------------------------------------------------------------ metrics
def test_ndcg_perfect_reversed_and_empty():
    search, starts = _blocks([3, 3])
    rel = np.array([5.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    perfect = ltr.ndcg_per_search(search, starts, rel, np.array([3, 2, 1, 3, 2, 1.0]))
    assert perfect[0] == pytest.approx(1.0) and np.isnan(perfect[1])
    worst = ltr.ndcg_per_search(search, starts, rel, np.array([1, 2, 3, 3, 2, 1.0]))
    expected = (5 / np.log2(4) + 1 / np.log2(3)) / (5 + 1 / np.log2(3))
    assert worst[0] == pytest.approx(expected)


def test_ndcg_truncation():
    search, starts = _blocks([4])
    rel = np.array([0, 0, 0, 1.0])
    s = np.array([4, 3, 2, 1.0])  # the relevant hotel is ranked 4th
    assert ltr.ndcg_per_search(search, starts, rel, s, k=3)[0] == 0.0
    assert ltr.ndcg_per_search(search, starts, rel, s, k=4)[0] == pytest.approx(1 / np.log2(5))


def test_single_relevant_hotel_ignores_its_weight():
    """Why the correction weights searches: one relevant hotel makes NDCG weight-free."""
    search, starts = _blocks([4])
    s = np.array([1, 4, 3, 2.0])
    a = ltr.ndcg_per_search(search, starts, np.array([0, 0, 1.0, 0]), s)
    b = ltr.ndcg_per_search(search, starts, np.array([0, 0, 7.3, 0]), s)
    assert a == pytest.approx(b)


def test_search_weights_use_the_chosen_hotels_slot():
    search, starts = _blocks([3, 3, 2])
    rel = np.array([1.0, 5.0, 0, 1.0, 0, 1.0, 0, 0])
    slots = np.array([1, 2, 3, 1, 2, 3, 1, 2])
    theta = np.array([np.nan, 1.0, 0.5, 0.25])
    w = ltr.search_weights(search, starts, rel, slots, theta)
    assert w[0] == pytest.approx(2.0)      # booked hotel in slot 2 beats the click in slot 1
    assert w[1] == pytest.approx(1.0)      # first of two clicks
    assert np.isnan(w[2])


def test_examination_from_position_coefficients():
    fit = SimpleNamespace(names=["log_price", "pos_2", "pos_3", "pos_6"],
                          beta=np.array([-1.0, np.log(0.8), np.log(0.6), np.log(0.3)]))
    th = ltr.examination(fit, max_slot=6)
    assert th[1] == 1.0 and th[2] == pytest.approx(0.8) and th[6] == pytest.approx(0.3)
    assert th[4] == pytest.approx(0.6) and th[5] == pytest.approx(0.6)  # reserved slots carry forward


def test_shown_order_agreement():
    search, _ = _blocks([3, 3])
    slots = np.array([1, 2, 3, 1, 2, 3])
    assert ltr.shown_order_agreement(search, slots, -slots.astype(float)) == pytest.approx(1.0)
    assert ltr.shown_order_agreement(search, slots, slots.astype(float)) == pytest.approx(-1.0)


def test_booked_rank():
    search, starts = _blocks([3, 2])
    booked = np.array([0, 0, 1, 0, 0])
    r = ltr.booked_rank(search, starts, booked, np.array([1, 2, 3, 1, 2.0]))
    assert r[0] == 1 and np.isnan(r[1])


def test_paired_bootstrap_ignores_missing():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.0, 1.0, 1.0, 3.0])
    out = ltr.paired_bootstrap(a, b, n_boot=200)
    assert out["diff"] == pytest.approx(1.0) and out["n"] == 3 and out["ci"] == [1.0, 1.0]


# ------------------------------------------------------------------ features
def test_labels():
    x = pd.DataFrame({"click_bool": [0, 1, 1], "booking_bool": [0, 0, 1]})
    assert ltr.labels(x).tolist() == [0, 1, 2]


def test_split_ranked_is_disjoint_and_complete():
    x = pd.DataFrame({"srch_id": [1, 1, 2, 3, 4, 4]})
    a, b = ltr.split_ranked(x)
    assert set(a.srch_id).isdisjoint(b.srch_id) and len(a) + len(b) == len(x)


def test_hotel_history_shrinks_rare_hotels():
    x = pd.DataFrame({"prop_id": [1] * 100 + [2] * 2, "click_bool": [1] * 50 + [0] * 50 + [1, 1],
                      "booking_bool": [0.0] * 102, "position": [1] * 102})
    h = ltr.hotel_history(x, prior_n=20)
    overall = x.click_bool.mean()
    assert h.loc[1, "hist_ctr"] == pytest.approx((50 + 20 * overall) / 120)
    assert overall < h.loc[2, "hist_ctr"] < 1.0      # 2 of 2 clicks, pulled towards the overall rate


@pytest.fixture(scope="module")
def prepared(synth):
    from hotelrank import audit

    flags = audit.price_flags(synth, audit.per_search(synth)).reset_index().rename(columns={"index": "srch_id"})
    x = position.analysis_sample(synth, flags)
    x = policy.prepare(choice.add_hotel_mean_price(x, full=x))
    rnd = x["random_bool"].astype(bool).to_numpy()
    hist_src, _ = ltr.split_ranked(x[~rnd])
    history = ltr.hotel_history(hist_src)
    return ltr.add_features(x[rnd], history), history


def test_add_features(prepared):
    xr, history = prepared
    assert set(ltr.EXTENDED) <= set(xr.columns)
    assert xr["price_pct_in_search"].between(0, 1).all()
    med = xr.groupby("srch_id")["log_price_vs_search_median"].median()
    assert np.allclose(med, 0, atol=1e-5)
    unseen = ~xr["prop_id"].isin(history.index)
    if unseen.any():
        assert np.allclose(xr.loc[unseen, "hist_ctr"], history.attrs["prior"]["hist_ctr"])


def test_ranker_beats_a_shuffle(prepared):
    xr, _ = prepared
    test = (xr["srch_id"] % 5 == 0).to_numpy()
    tr, te = xr[~test], xr[test]
    val = ((tr["srch_id"] // 5) % 5 == 0).to_numpy()
    r = ltr.fit_ranker("L1", tr[~val], tr[val], ltr.CORE, min_data_in_leaf=20)
    e = policy.make_eval_set(te)
    rel = ltr.relevance(ltr.labels(te))
    model = np.nanmean(ltr.ndcg_per_search(e.search, e.starts, rel, r.score(te)))
    shuffled = np.nanmean(ltr.ndcg_per_search(e.search, e.starts, rel, np.random.default_rng(0).random(len(te))))
    assert r.best_iteration >= 1 and model > shuffled
