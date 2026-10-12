"""Offline evaluation of ranking policies on randomly ordered searches.

The booking model is a conditional logit with an *outside option*: in each search the user
either books one of the hotels shown or books nothing ("clicked but did not book"). Hotel
utility = attributes + position effect; outside utility = a constant plus search-level
covariates. Because positions were assigned at random in these searches, the model learns
whether putting attractive hotels near the top raises the chance that *any* booking happens,
not only which hotel gets booked.

A policy is a score per hotel; within each search, hotels are re-assigned to the search's own
displayed slots in descending order of score. Expected bookings per search = 1 - P(outside);
expected booking value = sum_i P(book i) * nightly price * nights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import choice
from .data import RESERVED_SLOTS

OUTSIDE_COVARIATES = ["log_n_results", "log_length_of_stay", "log1p_booking_window", "srch_adults_count",
                      "srch_children_count", "srch_room_count", "srch_saturday_night_bool"]
HOTEL_ATTRIBUTES = [a for a in choice.ATTRIBUTES]
OUTSIDE_COLUMNS = ["outside"] + [f"out_{c}" for c in OUTSIDE_COVARIATES]


def prepare(x: pd.DataFrame, max_position: int = 38) -> pd.DataFrame:
    """Hotel rows used by the booking model: displayed slots only, with search covariates."""
    x = x[(x["position"] <= max_position) & ~x["position"].isin(RESERVED_SLOTS)].copy()
    n = x.groupby("srch_id")["position"].transform("size")
    x["log_n_results"] = np.log(n).astype(np.float32)
    x["log_length_of_stay"] = np.log(x["srch_length_of_stay"].astype(float)).astype(np.float32)
    x["log1p_booking_window"] = np.log1p(x["srch_booking_window"].clip(lower=0).astype(float)).astype(np.float32)
    for c in ("srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool"):
        x[c] = x[c].astype(np.float32)
    x["value_if_booked"] = (x["price_usd"].astype(float) * x["srch_length_of_stay"].astype(float)).astype(np.float32)
    return x.sort_values(["srch_id", "position"], kind="stable")


def with_outside_rows(x: pd.DataFrame, outcome: str = "booking_bool") -> pd.DataFrame:
    """Append one outside-option row per search; it is chosen when nothing was booked."""
    hotels = x.copy()
    hotels["chosen"] = hotels[outcome].astype(bool)
    for c in OUTSIDE_COLUMNS:
        hotels[c] = np.float32(0)
    first = hotels.groupby("srch_id", sort=False).head(1)
    out = pd.DataFrame({"srch_id": first["srch_id"].to_numpy(), "position": 1})
    for a in HOTEL_ATTRIBUTES:
        out[a] = np.float32(0)
    out["outside"] = np.float32(1)
    for c in OUTSIDE_COVARIATES:
        out[f"out_{c}"] = first[c].to_numpy().astype(np.float32)
    booked = hotels.groupby("srch_id", sort=False)["chosen"].any()
    out["chosen"] = ~booked.reindex(out["srch_id"]).to_numpy()
    # the outside row comes first in its search; position 1 gives it no position term
    out["_order"] = 0
    hotels["_order"] = 1
    both = pd.concat([out, hotels[["srch_id", "position", "chosen", *HOTEL_ATTRIBUTES, *OUTSIDE_COLUMNS]].assign(_order=1)],
                     ignore_index=True)
    return both.sort_values(["srch_id", "_order", "position"], kind="stable").drop(columns="_order")


RIDGE = 0.01  # see choice.fit; negligible with the real data's ~12k bookings


def fit_booking_model(x: pd.DataFrame, ridge: float = RIDGE) -> choice.ChoiceFit:
    """Fit the outside-option model. Search covariates that do not vary (e.g. every search
    for two adults in a small sample) are dropped, since they would duplicate the constant."""
    both = with_outside_rows(x)
    out_rows = both["outside"] == 1
    active = ["outside"] + [c for c in OUTSIDE_COLUMNS[1:] if both.loc[out_rows, c].std() > 0]
    # Rows are in (search, outside first, position) order; build() re-sorts by (srch_id,
    # position) stably, which keeps the outside row (position 1) before the hotel at position 1.
    return choice.fit(choice.build(both, "chosen", attributes=HOTEL_ATTRIBUTES + active), ridge=ridge)


@dataclass
class EvalSet:
    """Held-out searches in a form that makes re-ranking cheap."""
    search: np.ndarray        # search index per hotel row (0..S-1), rows sorted by search then position
    slots: np.ndarray         # displayed slot per row, ascending within each search
    A: np.ndarray             # hotel attributes (rows x len(HOTEL_ATTRIBUTES))
    W: np.ndarray             # outside covariates per search (S x len(OUTSIDE_COVARIATES))
    value: np.ndarray         # nightly price x nights per row
    review: np.ndarray
    no_reviews: np.ndarray
    price: np.ndarray
    booked_any: np.ndarray    # observed: did the search end in a booking (logged order)
    starts: np.ndarray
    search_ids: np.ndarray

    @property
    def n_searches(self) -> int:
        return len(self.starts)


def make_eval_set(x: pd.DataFrame) -> EvalSet:
    sid = x["srch_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    sizes = np.diff(np.r_[starts, len(sid)])
    search = np.repeat(np.arange(len(starts)), sizes)
    first = x.iloc[starts]
    return EvalSet(search=search, slots=x["position"].to_numpy().astype(int),
                   A=np.column_stack([x[a].to_numpy().astype(np.float64) for a in HOTEL_ATTRIBUTES]),
                   W=np.column_stack([first[c].to_numpy().astype(np.float64) for c in OUTSIDE_COVARIATES]),
                   value=x["value_if_booked"].to_numpy().astype(np.float64),
                   review=x["prop_review_score"].to_numpy().astype(np.float64),
                   no_reviews=x["no_reviews"].to_numpy().astype(np.float64),
                   price=x["price_usd"].to_numpy().astype(np.float64),
                   booked_any=x.groupby("srch_id", sort=False)["booking_bool"].max().to_numpy().astype(float),
                   starts=starts, search_ids=sid[starts])


@dataclass
class Params:
    b_hotel: np.ndarray       # coefficients on HOTEL_ATTRIBUTES
    b_out: np.ndarray         # coefficients on OUTSIDE_COLUMNS (constant first)
    pos_effect: np.ndarray    # utility by slot (index = slot), 0 at slot 1

    @classmethod
    def from_vector(cls, names: list[str], beta: np.ndarray, max_slot: int = 40) -> "Params":
        b = dict(zip(names, beta))
        pos = np.zeros(max_slot + 1)
        for nm, v in b.items():
            if nm.startswith("pos_"):
                pos[int(nm[4:])] = v
        return cls(b_hotel=np.array([b.get(a, 0.0) for a in HOTEL_ATTRIBUTES]),
                   b_out=np.array([b.get(c, 0.0) for c in OUTSIDE_COLUMNS]), pos_effect=pos)


def hotel_utility(e: EvalSet, p: Params) -> np.ndarray:
    """Attribute utility of each hotel, without any position term."""
    return e.A @ p.b_hotel


def assign_slots(e: EvalSet, score: np.ndarray) -> np.ndarray:
    """Within each search, give the highest score the search's best (lowest) slot."""
    order = np.lexsort((-score, e.search))  # by search, then score descending
    new = np.empty_like(e.slots)
    new[order] = e.slots  # e.slots is ascending within each search, in the same row blocks
    return new


def outcomes(e: EvalSet, p: Params, slots: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Expected bookings and booking value per search when hotels sit in ``slots``."""
    v = hotel_utility(e, p) + p.pos_effect[slots]
    w_out = p.b_out[0] + e.W @ p.b_out[1:]
    vmax = np.maximum(np.maximum.reduceat(v, e.starts), w_out)
    ev = np.exp(v - vmax[e.search])
    denom = np.add.reduceat(ev, e.starts) + np.exp(w_out - vmax)
    pb = ev / denom[e.search]
    book = np.add.reduceat(pb, e.starts)
    value = np.add.reduceat(pb * e.value, e.starts)
    if weights is None:
        weights = np.ones(e.n_searches)
    W = weights.sum()
    return {"bookings": float(book @ weights / W), "value": float(value @ weights / W),
            "per_search_bookings": book, "per_search_value": value, "row_probs": pb}


def guardrail_metrics(e: EvalSet, p: Params, slots: np.ndarray) -> dict:
    """The two guardrails in reports/metrics_framework.md, as expected values under ``slots``:
    the share of bookings that go to hotels with no reviews, and the nightly price of the
    average booked hotel."""
    pb = outcomes(e, p, slots)["row_probs"]
    return {"no_review_booking_share": float(pb @ e.no_reviews / pb.sum()),
            "booked_nightly_price_usd": float(pb @ e.price / pb.sum())}


def crossfit_check(x_train: pd.DataFrame, e: EvalSet, lam: float = 0.3, ridge: float = RIDGE) -> pd.DataFrame:
    """Does the evaluator flatter the rankings built from the same model?

    The training searches are split in two halves; each half's model builds the model rankings,
    and the *other* half's model scores them (cross), next to the same model scoring its own
    rankings (same). Baselines do not use the model, so they are only scored. Results are
    averaged over the two directions.
    """
    half = ((x_train["srch_id"] // 5) % 2 == 0).to_numpy()
    fits = [fit_booking_model(x_train[half], ridge), fit_booking_model(x_train[~half], ridge)]
    params = [Params.from_vector(f.names, f.beta) for f in fits]
    rows = []
    for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
        build, score = params[i], params[j]
        sc = policy_scores(e, build)
        sc[f"blend_lambda_{lam}"] = blend_scores(e, build, lam)
        for k, s in sc.items():
            o = outcomes(e, score, assign_slots(e, s))
            rows.append({"policy": k, "scoring": "same model" if i == j else "other half's model",
                         "bookings": o["bookings"], "value": o["value"]})
    return pd.DataFrame(rows).groupby(["policy", "scoring"])[["bookings", "value"]].mean().unstack("scoring")


def policy_scores(e: EvalSet, p: Params) -> dict[str, np.ndarray]:
    u = hotel_utility(e, p)
    return {
        "B1_reviews_first": e.review - 1e3 * e.no_reviews,       # no-review hotels last
        "B2_cheapest_first": -e.price,
        "M1_most_likely_booked": u,
        "M2_highest_expected_value": u + np.log(e.value),           # log(P x value) up to a search constant
    }


def blend_scores(e: EvalSet, p: Params, lam: float) -> np.ndarray:
    """Exploratory family between M1 (lam=0) and M2 (lam=1)."""
    return hotel_utility(e, p) + lam * np.log(e.value)


def evaluate(e: EvalSet, fit_: choice.ChoiceFit, n_boot: int = 200, seed: int = 0) -> dict:
    """Point estimates and bootstrap draws for every policy and the logged (random) order.

    Each replicate draws coefficients from N(beta_hat, cov) and Poisson weights over searches,
    so intervals reflect both model uncertainty and which searches happened to be held out.
    Policies that use the model (M1, M2) are re-ranked with the drawn coefficients.
    """
    rng = np.random.default_rng(seed)
    p_hat = Params.from_vector(fit_.names, fit_.beta)
    names = list(policy_scores(e, p_hat)) + ["logged_random_order"]

    def run(p: Params, weights=None):
        sc = policy_scores(e, p)
        res = {k: outcomes(e, p, assign_slots(e, s), weights) for k, s in sc.items()}
        res["logged_random_order"] = outcomes(e, p, e.slots, weights)
        return {k: (r["bookings"], r["value"]) for k, r in res.items()}

    point = run(p_hat)
    draws = {k: np.empty((n_boot, 2)) for k in names}
    w, V = np.linalg.eigh((fit_.cov + fit_.cov.T) / 2)
    L = V * np.sqrt(np.clip(w, 0, None))  # robust square root of the coefficient covariance
    for b in range(n_boot):
        beta = fit_.beta + L @ rng.standard_normal(len(fit_.beta))
        res = run(Params.from_vector(fit_.names, beta), rng.poisson(1.0, e.n_searches).astype(float))
        for k in names:
            draws[k][b] = res[k]
    return {"point": point, "draws": draws}


def interval(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


BASELINES = ("B1_reviews_first", "B2_cheapest_first")
MODEL_POLICIES = ("M1_most_likely_booked", "M2_highest_expected_value")


def apply_decision_rule(result: dict, value_tolerance: float = 0.02) -> dict:
    """The rule committed in reports/decision_rule.md, applied mechanically.

    B* and M* are the baseline and model policy with the most expected bookings (point
    estimates). Recommend testing M* if (1) the 95% interval of M* - B* bookings is above zero
    and (2) M*'s expected booking value is at most ``value_tolerance`` below B*'s.
    """
    pt, dr = result["point"], result["draws"]
    b_star = max(BASELINES, key=lambda k: pt[k][0])
    m_star = max(MODEL_POLICIES, key=lambda k: pt[k][0])
    diff = dr[m_star][:, 0] - dr[b_star][:, 0]
    lo, hi = interval(diff)
    value_change = pt[m_star][1] / pt[b_star][1] - 1
    cond1, cond2 = lo > 0, value_change >= -value_tolerance
    failed = [c for c, ok in (("bookings interval not above zero", cond1),
                              (f"booking value more than {value_tolerance:.0%} lower", cond2)) if not ok]
    b2 = dr["B2_cheapest_first"][:, 0] - dr["B1_reviews_first"][:, 0]
    return {
        "B_star": b_star, "M_star": m_star,
        "bookings_diff": pt[m_star][0] - pt[b_star][0], "bookings_diff_ci": [lo, hi],
        "bookings_rel_change": pt[m_star][0] / pt[b_star][0] - 1,
        "value_rel_change": value_change,
        "condition_1_bookings": bool(cond1), "condition_2_value": bool(cond2),
        "recommend": m_star if cond1 and cond2 else b_star,
        "failed": failed,
        "cheapest_first_vs_reviews_first": {
            "bookings_diff": pt["B2_cheapest_first"][0] - pt["B1_reviews_first"][0],
            "bookings_diff_ci": list(interval(b2)),
            "value_rel_change": pt["B2_cheapest_first"][1] / pt["B1_reviews_first"][1] - 1,
        },
    }


def evaluator_calibration(e: EvalSet, fit_: choice.ChoiceFit, bins: int = 10) -> pd.DataFrame:
    """Predicted P(any booking) under the order actually shown vs the observed booking rate,
    by decile of prediction, on held-out searches. Since the shown order was random, searches
    where good hotels happened to land on top get higher predictions; if bookings rise with
    them, the model has learned how the order affects whether a booking happens."""
    p = Params.from_vector(fit_.names, fit_.beta)
    pred = outcomes(e, p, e.slots)["per_search_bookings"]
    q = pd.qcut(pred, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": pred, "obs": e.booked_any, "bin": q}).groupby("bin").mean()


def order_luck_check(e: EvalSet, fit_: choice.ChoiceFit, n_shuffles: int = 20, bins: int = 5,
                     seed: int = 0) -> pd.DataFrame:
    """The sharpest test of the mechanism the policies rely on.

    In each held-out search, compare the predicted booking probability under the order that was
    actually shown with its average over random re-shuffles of the same hotels. The difference
    is the search's "order luck": positive when attractive hotels happened to land on top.
    If the model is right, searches with more luck should book more than their shuffled
    average by about the predicted amount.

    The shuffled average is estimated twice, from independent sets of ``n_shuffles`` shuffles:
    one set for the predicted uplift, the other for the observed one. With a single set, its
    Monte Carlo error enters both sides with the same sign and pulls the slope towards 1 even
    when order does not matter. With independent sets the slope is unbiased at 0; the remaining
    Monte Carlo error in the predicted side attenuates it slightly (about 1/(1 + 1/n_shuffles)).
    """
    rng = np.random.default_rng(seed)
    p = Params.from_vector(fit_.names, fit_.beta)
    logged = outcomes(e, p, e.slots)["per_search_bookings"]

    def shuffled_mean() -> np.ndarray:
        acc = np.zeros(e.n_searches)
        for _ in range(n_shuffles):
            acc += outcomes(e, p, assign_slots(e, rng.random(len(e.slots))))["per_search_bookings"]
        return acc / n_shuffles

    shuffled_x, shuffled_y = shuffled_mean(), shuffled_mean()
    luck = logged - shuffled_x
    q = pd.qcut(luck, bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"predicted_uplift": luck, "observed_uplift": e.booked_any - shuffled_y, "bin": q})
    out = df.groupby("bin").agg(searches=("bin", "size"), predicted_uplift=("predicted_uplift", "mean"),
                                observed_uplift=("observed_uplift", "mean"))
    out["observed_se"] = df.groupby("bin")["observed_uplift"].std() / np.sqrt(out["searches"])
    # One-number summary: slope of observed on predicted uplift across searches (1 if the model
    # is right about how order affects booking, 0 if order does not matter).
    xv, yv = df["predicted_uplift"].to_numpy(), df["observed_uplift"].to_numpy()
    xc = xv - xv.mean()
    slope = float(xc @ (yv - yv.mean()) / (xc @ xc))
    resid = yv - yv.mean() - slope * xc
    se = float(np.sqrt((resid @ resid) / (len(yv) - 2) / (xc @ xc)))
    out.attrs["slope"] = {"slope": slope, "se": se, "ci": [slope - 1.96 * se, slope + 1.96 * se]}
    return out
