"""A synthetic impressions table with the real table's columns and a known truth.

Used by CI (the real data cannot be redistributed) and by tests that check the estimators
recover what was planted. The generator mimics the real data's structure:

* searches of 5-38 results, with slots 5, 11, 17 and 23 left empty;
* two arms: the "algorithm" arm orders hotels by attractiveness plus noise, the "random" arm
  shuffles them, except a share of random searches that the user sorted by price;
* clicks follow a position-based model, P(click) = examination(rank) x attractiveness;
* only searches with at least one click are kept, and at most one booking per search, with the
  booking probability differing by arm (as in the real release).

Nothing here is calibrated to produce the real results; it only has to exercise the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RESERVED_SLOTS

TRUTH = {
    "examination_exponent": 0.6,          # examination(rank) = rank ** -0.6
    "intercept": -1.2,
    "beta": {"prop_starrating": 0.25, "prop_review_score": 0.20, "rel_log_price": -1.5,
             "promotion_flag": 0.35, "prop_location_score1": 0.10},
    "price_sorted_share": 0.06,
    "booking_given_click": {"algorithm": 0.9, "random": 0.15},
    "algorithm_noise_sd": 0.6,
}


def examination(rank: np.ndarray) -> np.ndarray:
    return np.asarray(rank, dtype=float) ** -TRUTH["examination_exponent"]


def display_slots(n: int) -> np.ndarray:
    """The first ``n`` positions, skipping the reserved slots."""
    slots, p = [], 1
    while len(slots) < n:
        if p not in RESERVED_SLOTS:
            slots.append(p)
        p += 1
    return np.array(slots)


def make_impressions(n_searches: int = 3000, seed: int = 0, random_share: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    b = TRUTH["beta"]
    cols: dict[str, list] = {}
    hotel_counter = 0
    t0 = np.datetime64("2012-11-01T00:00:00")

    def add(name, value, n, dtype=None):
        arr = np.asarray(value, dtype=dtype)
        cols.setdefault(name, []).append(np.full(n, arr, dtype=arr.dtype) if arr.ndim == 0 else arr)

    sid = 0
    while sid < n_searches:
        n = int(rng.integers(5, 35))
        is_random = rng.random() < random_share
        stars = rng.integers(1, 6, n)
        review = np.clip(np.round(rng.normal(3.8, 0.7, n) * 2) / 2, 0, 5)
        loc = np.round(rng.uniform(0, 7, n), 2)
        promo = rng.random(n) < 0.22
        base = rng.lognormal(np.log(80), 0.3)
        price = np.round(base * np.exp(0.10 * stars + rng.normal(0, 0.25, n)), 2)
        lp = np.log(price)
        rel = lp - lp.mean()
        util = (TRUTH["intercept"] + b["prop_starrating"] * (stars - 3) + b["prop_review_score"] * (review - 3.8)
                + b["rel_log_price"] * rel + b["promotion_flag"] * promo + b["prop_location_score1"] * (loc - 3.5))
        attract = 1 / (1 + np.exp(-util))
        if is_random:
            order = np.argsort(price) if rng.random() < TRUTH["price_sorted_share"] else rng.permutation(n)
        else:
            order = np.argsort(-(util + rng.normal(0, TRUTH["algorithm_noise_sd"], n)))
        rank = np.empty(n, int)
        rank[order] = np.arange(1, n + 1)
        click = rng.random(n) < examination(rank) * attract
        if not click.any():
            continue  # the real release keeps only searches with at least one click
        sid += 1
        arm = "random" if is_random else "algorithm"
        booking = np.zeros(n, bool)
        if rng.random() < TRUTH["booking_given_click"][arm]:
            idx = np.flatnonzero(click)
            w = attract[idx] / attract[idx].sum()
            booking[rng.choice(idx, p=w)] = True
        los = int(rng.integers(1, 6))
        add('srch_id', np.full(n, sid, np.int32), n)
        add('date_time', t0 + np.timedelta64(int(rng.integers(0, 240 * 24)), "h"), n)
        add('site_id', np.int16(rng.integers(1, 35)), n)
        add('visitor_location_country_id', np.int16(219), n)
        add('visitor_hist_starrating', np.float32(np.nan), n)
        add('visitor_hist_adr_usd', np.float32(np.nan), n)
        add('prop_country_id', np.int16(219), n)
        add('prop_id', np.arange(hotel_counter, hotel_counter + n, dtype=np.int32), n)
        add('prop_starrating', stars.astype(np.int8), n)
        add('prop_review_score', review.astype(np.float32), n)
        add('prop_brand_bool', rng.random(n) < 0.6, n)
        add('prop_location_score1', loc.astype(np.float32), n)
        add('prop_location_score2', np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 1, n)).astype(np.float32), n)
        add('prop_log_historical_price', (lp + rng.normal(0, 0.1, n)).astype(np.float32), n)
        add('position', display_slots(n)[rank - 1].astype(np.int8), n)
        add('price_usd', price, n)
        add('promotion_flag', promo, n)
        add('srch_destination_id', np.int32(rng.integers(1, 500)), n)
        add('srch_length_of_stay', np.int16(los), n)
        add('srch_booking_window', np.int16(rng.integers(0, 90)), n)
        add('srch_adults_count', np.int8(2), n)
        add('srch_children_count', np.int8(0), n)
        add('srch_room_count', np.int8(1), n)
        add('srch_saturday_night_bool', bool(rng.random() < 0.5), n)
        add('srch_query_affinity_score', np.float32(np.nan), n)
        add('orig_destination_distance', np.float32(np.nan), n)
        add('random_bool', is_random, n)
        add('click_bool', click, n)
        add('gross_bookings_usd', np.where(booking, price * los, np.nan).astype(np.float32), n)
        add('booking_bool', booking, n)
        add('comp_n_known', rng.integers(0, 3, n).astype(np.int8), n)
        add('comp_n_expedia_cheaper', np.zeros(n, np.int8), n)
        add('comp_n_expedia_dearer', np.zeros(n, np.int8), n)
        hotel_counter += n
    out = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    out["date_time"] = out["date_time"].astype("datetime64[ns]")
    return out
