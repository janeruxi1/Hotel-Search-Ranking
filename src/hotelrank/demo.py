"""Pure functions behind the Streamlit demo. They read only the committed records in
``reports/metrics`` (aggregates), never the Kaggle data, so the public demo redistributes
nothing and every number it shows is one the notebooks produced."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import abtest
from .data import RESERVED_SLOTS

POLICY_LABELS = {
    "B1_reviews_first": "Reviews first",
    "B2_cheapest_first": "Cheapest first (the proposal)",
    "M1_most_likely_booked": "Model: most likely booked",
    "M2_highest_expected_value": "Model: highest expected value",
    "logged_random_order": "Random order, as shown",
}


def available_positions(fit_record: dict) -> list[int]:
    """Positions with an estimated effect: 1 (the reference) and every pos_k in the fit."""
    return [1] + sorted(int(n[4:]) for n in fit_record["names"] if n.startswith("pos_"))


def position_value(fit_record: dict, from_pos: int, to_pos: int, price: float) -> dict:
    """Price cut (USD at ``price``) worth the same as moving a hotel from ``from_pos`` to ``to_pos``.

    Utility difference in log-price units, -(b_to - b_from) / b_log_price, with a delta-method
    95% interval from the fit's covariance; converted with 1 - exp(-x). Negative values mean the
    move is worth a price *rise* (moving down the list).
    """
    names, beta = fit_record["names"], np.asarray(fit_record["beta"], float)
    cov = np.asarray(fit_record["cov"], float)
    for p in (from_pos, to_pos):
        if p in RESERVED_SLOTS or (p != 1 and f"pos_{p}" not in names):
            raise ValueError(f"no estimate for position {p}")
    w = np.zeros(len(names))
    if to_pos != 1:
        w[names.index(f"pos_{to_pos}")] += 1.0
    if from_pos != 1:
        w[names.index(f"pos_{from_pos}")] -= 1.0
    j = names.index("log_price")
    a, bd = float(w @ beta), float(beta[j])
    g = -w / bd
    g[j] += a / bd ** 2
    x, se = -a / bd, float(np.sqrt(g @ cov @ g))

    def usd(v: float) -> float:
        return price * (1 - math.exp(-v))

    return {"log_units": x, "pct": 1 - math.exp(-x), "usd": usd(x),
            "usd_lo": usd(x - 1.96 * se), "usd_hi": usd(x + 1.96 * se)}


def position_curves(p2: dict, outcome: str) -> pd.DataFrame:
    """Rate relative to position 1, naive (algorithm order) and randomized, with 95% bands."""
    rows = []
    for arm, label in (("algorithm", "Expedia's ranking (naive)"), ("random", "Random order")):
        for pos, r in p2["raw_curves"][f"{arm}/{outcome}"].items():
            rows.append({"position": int(pos), "series": label, "relative to position 1": r["rel_to_1"],
                         "lo": r["rel_lo"], "hi": r["rel_hi"]})
    return pd.DataFrame(rows)


def policy_table(p4: dict) -> pd.DataFrame:
    rows = []
    for k, v in p4["policies"].items():
        rows.append({"ranking": POLICY_LABELS.get(k, k),
                     "bookings per 100 searches": 100 * v["bookings_per_search"],
                     "95% interval": f"{100 * v['bookings_ci'][0]:.2f}–{100 * v['bookings_ci'][1]:.2f}",
                     "booking value per search (USD)": v["value_per_search_usd"]})
    return pd.DataFrame(rows)


def frontier(p4: dict) -> pd.DataFrame:
    f = pd.DataFrame(p4["frontier_exploratory"])
    f["bookings per 100 searches"] = 100 * f["bookings"]
    return f.rename(columns={"value": "booking value per search (USD)"})


def size_test(p5: dict, p4: dict, *, margin: float, discount: float, comparisons: int = 3,
              power: float = 0.8, daily_searches: int = 100_000, cap_value: bool = True) -> dict:
    """Re-run the phase-5 sizing with the knobs a stakeholder would turn."""
    inp = p5["inputs"]
    p0 = inp["baseline_booking_rate"]
    alpha = 0.05 / comparisons
    lift = p5["lifts"]["T2 blend vs T1 reviews first"]["offline"] * discount
    n_booking = abtest.n_two_proportions(p0, lift, alpha, power) if lift > 0 else None
    v = inp["value_capped_p99" if cap_value else "value_raw"]
    n_guard = abtest.n_noninferiority_mean(v["mean"], v["sd"], margin, 0.025, power)
    n_arm = max(n_guard, n_booking or 0)
    total = 3 * n_arm
    days = max(14, math.ceil(total / daily_searches))
    return {"alpha": alpha, "lift": lift, "n_booking": n_booking, "n_guardrail": n_guard, "n_per_arm": n_arm,
            "n_total": total, "days": days, "binding": "guardrail" if n_guard >= (n_booking or 0) else "booking lift",
            "mde": abtest.mde_two_proportions(p0, n_arm, alpha, power)}
