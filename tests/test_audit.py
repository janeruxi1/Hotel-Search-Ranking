import numpy as np
import pandas as pd
import pytest

from hotelrank import audit
from hotelrank.data import RESERVED_SLOTS
from hotelrank.synthetic import TRUTH, display_slots


def test_display_slots_skip_reserved():
    s = display_slots(30)
    assert len(s) == 30 and not set(RESERVED_SLOTS) & set(s.tolist())
    assert s[:5].tolist() == [1, 2, 3, 4, 6]


def test_per_search_and_arm_summary(synth):
    s = audit.per_search(synth)
    assert len(s) == synth["srch_id"].nunique()
    arms = audit.arm_summary(s)
    assert arms["algorithm"]["searches"] + arms["random"]["searches"] == len(s)
    assert arms["random"]["searches_with_click"] == 1.0
    # the generator books far more often in the algorithm arm, as in the real release
    assert arms["algorithm"]["searches_with_booking"] > arms["random"]["searches_with_booking"]


def test_reserved_slots_are_detected(synth):
    occ = audit.slot_occupancy(synth)
    shares = audit.reserved_slot_share(occ)
    assert all(v == 0 for v in shares["algorithm"].values())


def test_rank_corr_known_cases():
    df = pd.DataFrame({"srch_id": [1] * 6 + [2] * 6 + [3] * 3,
                       "position": list(range(1, 7)) * 2 + [1, 2, 3],
                       "v": [1, 2, 3, 4, 5, 6] + [6, 5, 4, 3, 2, 1] + [1, 2, 3]})
    r = audit.within_search_rank_corr(df, "v")
    assert r[1] == pytest.approx(1.0) and r[2] == pytest.approx(-1.0)
    assert np.isnan(r[3])  # fewer than 5 rows


def test_price_sorted_share_recovered(synth):
    flags = audit.price_flags(synth, audit.per_search(synth))
    rnd = flags[flags["random"]]
    assert rnd["price_sorted"].mean() == pytest.approx(TRUTH["price_sorted_share"], abs=0.03)
    assert flags.loc[~flags["random"], "price_sorted"].mean() < 0.02


def test_balance_random_arm_flat_algorithm_tilted(synth):
    flags = audit.price_flags(synth, audit.per_search(synth))
    keep = ~flags["bad_price"] & ~flags["price_sorted"].fillna(False)
    b = audit.balance(synth, ["prop_starrating", "promotion_flag"], keep=keep)
    assert abs(b["random"]["prop_starrating"]) < 0.04
    assert b["algorithm"]["prop_starrating"] < -0.1


def test_price_summary_flags_errors():
    df = pd.DataFrame({"srch_id": [1, 1, 1, 2, 2], "price_usd": [100, 0, 9e6, 50, 60],
                       "srch_length_of_stay": [1, 1, 1, 2, 2]})
    p = audit.price_summary(df)
    assert p["impressions_price_le_0"] == 1 and p["share_above_cap"] == pytest.approx(0.2)
