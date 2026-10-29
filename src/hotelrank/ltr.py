"""Learning to rank: LightGBM LambdaMART next to the choice-model rankings of phase 4.

Three questions, kept apart on purpose:

1. **Algorithm.** Does LambdaMART order hotels better than the conditional logit when both see
   the same eight attributes? (``core`` features)
2. **Features.** How much more comes from the signals the logit was not given: within-search
   comparisons, the hotel's history in Expedia's own ranked traffic, location score 2,
   competitor prices? (``extended`` features)
3. **Training data.** What changes when the ranker learns from Expedia-ordered searches, whose
   clicks are inflated at the top by position, instead of the randomly ordered ones?

Every ranker is judged on the same held-out random-order searches as phase 4. Two yardsticks:

* **NDCG on what users did** (model-free). Positions in these searches were random, so which
  hotel was clicked or booked depended on its slot only through noise that is unrelated to any
  ranking's scores. A ranking that puts the hotels users chose nearer the top scores higher.
  Gains follow the ICDM 2013 competition: 5 for a booking, 1 for a click, 0 otherwise.
* **The phase 4 booking model** (expected bookings and booking value per search). It sees only
  its own eight attributes, so it can credit a ranker for other features only through their
  correlation with those eight. It is reported, with that limit stated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import choice

LABEL_GAIN = [0.0, 1.0, 5.0]  # label 0 nothing, 1 click, 2 booking
TOP_K = 38                    # the deepest displayed slot; the competition scored NDCG@38

CORE = list(choice.ATTRIBUTES)
HOTEL_HISTORY = ["hist_log_impressions", "hist_ctr", "hist_booking_rate", "hist_mean_position"]
EXTENDED = CORE + [
    # within-search comparisons: how this hotel looks next to the others on the page
    "price_pct_in_search", "log_price_vs_search_median", "star_vs_search_mean", "review_vs_search_mean",
    "loc2_pct_in_search",
    # the hotel itself, beyond the logit's eight
    "prop_location_score2", "prop_log_historical_price", "log_price_vs_historical",
    "comp_n_known", "comp_n_expedia_cheaper", "comp_n_expedia_dearer",
    # the search
    "log_n_results", "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
    "srch_children_count", "srch_room_count", "srch_saturday_night_bool", "orig_destination_distance",
    # the hotel's record in Expedia-ranked searches kept apart for this (see split_ranked)
    *HOTEL_HISTORY,
]

PARAMS = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [TOP_K], "label_gain": LABEL_GAIN,
          "lambdarank_truncation_level": TOP_K, "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 1.0, "num_threads": 0, "verbosity": -1}
MAX_ROUNDS = 2000
EARLY_STOP = 100


def labels(x: pd.DataFrame) -> np.ndarray:
    """0 = not clicked, 1 = clicked, 2 = booked."""
    return (x["click_bool"].to_numpy() > 0).astype(int) + (x["booking_bool"].to_numpy() > 0).astype(int)


def split_ranked(x_ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expedia-ranked searches, split in two by search id: one half supplies the hotel history
    features, the other half is the pool for the ranker trained on ranked traffic. Keeping them
    apart stops that ranker from learning its own labels through the history features."""
    half = (x_ranked["srch_id"] % 2 == 0).to_numpy()
    return x_ranked[half], x_ranked[~half]


def hotel_history(x_hist: pd.DataFrame, prior_n: float = 20.0) -> pd.DataFrame:
    """Per-hotel record in the history searches, shrunk toward the overall rate.

    With ``prior_n`` pseudo-impressions at the overall rate, a hotel seen 3 times does not get a
    click rate of 0% or 67%. ``hist_mean_position`` is where Expedia's own ranking tends to put
    the hotel: in effect, a summary of the production ranker.
    """
    g = x_hist.groupby("prop_id")
    n = g.size()
    ctr0, br0 = x_hist["click_bool"].mean(), x_hist["booking_bool"].mean()
    pos0 = x_hist["position"].mean()
    out = pd.DataFrame({
        "hist_log_impressions": np.log1p(n),
        "hist_ctr": (g["click_bool"].sum() + prior_n * ctr0) / (n + prior_n),
        "hist_booking_rate": (g["booking_bool"].sum() + prior_n * br0) / (n + prior_n),
        "hist_mean_position": (g["position"].sum() + prior_n * pos0) / (n + prior_n),
    })
    out.attrs["prior"] = {"hist_log_impressions": 0.0, "hist_ctr": ctr0, "hist_booking_rate": br0,
                          "hist_mean_position": pos0}
    return out.astype(np.float32)


def add_features(x: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """The extended feature set. ``x`` must already have the choice-model attributes
    (see policy.prepare and choice.add_hotel_mean_price) and be sorted by search."""
    x = x.copy()
    sid = x["srch_id"]
    lp = x["log_price"].astype(float)
    x["price_pct_in_search"] = lp.groupby(sid).rank(pct=True).astype(np.float32)
    x["log_price_vs_search_median"] = (lp - lp.groupby(sid).transform("median")).astype(np.float32)
    star = x["prop_starrating"].astype(float)
    x["star_vs_search_mean"] = (star - star.groupby(sid).transform("mean")).astype(np.float32)
    rev = x["prop_review_score"].astype(float)
    x["review_vs_search_mean"] = (rev - rev.groupby(sid).transform("mean")).astype(np.float32)
    loc2 = x["prop_location_score2"].astype(float)
    x["loc2_pct_in_search"] = loc2.groupby(sid).rank(pct=True).astype(np.float32)
    hist_price = x["prop_log_historical_price"].astype(float).replace(0.0, np.nan)  # 0 = not recorded
    x["prop_log_historical_price"] = hist_price.astype(np.float32)
    x["log_price_vs_historical"] = (lp - hist_price).astype(np.float32)
    if "log_n_results" not in x:
        x["log_n_results"] = np.log(sid.groupby(sid).transform("size")).astype(np.float32)
    h = history.reindex(x["prop_id"].to_numpy())
    for c in HOTEL_HISTORY:
        # a hotel never seen in the history searches gets the overall rate, and 0 impressions
        x[c] = h[c].fillna(history.attrs["prior"][c]).to_numpy().astype(np.float32)
    for c in EXTENDED:
        x[c] = x[c].astype(np.float32)
    return x


def group_sizes(x: pd.DataFrame) -> np.ndarray:
    """Rows per search, in row order. Rows must be contiguous by search."""
    sid = x["srch_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    return np.diff(np.r_[starts, len(sid)])


@dataclass
class Ranker:
    name: str
    features: list[str]
    model: object
    best_iteration: int
    valid_ndcg: float

    def score(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x[self.features], num_iteration=self.best_iteration)


def fit_ranker(name: str, train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int = 0,
               **overrides) -> Ranker:
    """LambdaMART, stopped when validation NDCG@38 has not improved for EARLY_STOP rounds."""
    import lightgbm as lgb

    params = {**PARAMS, "seed": seed, "deterministic": True, "force_row_wise": True, **overrides}
    dtrain = lgb.Dataset(train[features], labels(train), group=group_sizes(train), free_raw_data=True)
    dvalid = lgb.Dataset(valid[features], labels(valid), group=group_sizes(valid), reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return Ranker(name, features, model, int(model.best_iteration),
                  float(model.best_score["valid"][f"ndcg@{TOP_K}"]))


def relevance(lab: np.ndarray) -> np.ndarray:
    """Competition gains: 5 for the booked hotel, 1 for a clicked one, 0 otherwise."""
    return np.asarray(LABEL_GAIN)[lab]


def search_weights(search: np.ndarray, starts: np.ndarray, rel: np.ndarray, slots: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
    """Inverse-propensity weight per search: 1 / (chance that the slot of its chosen hotel was
    looked at, relative to slot 1). The chosen hotel is the one with the largest gain (the booked
    one, else the first clicked one). Searches with no gain get NaN.

    Why per search and not per hotel: almost every search has a single click. With one relevant
    hotel, NDCG is a ratio in which any weight on that hotel cancels, so re-weighting hotels
    changes nothing. What position distorts is *which* hotel ended up chosen, so the correction
    re-weights searches: a choice made at a slot users rarely look at counts for more.
    """
    order = np.lexsort((np.arange(len(rel)), -rel, search))   # best gain first within each search
    first = order[starts]
    w = 1.0 / theta[slots[first]]
    return np.where(rel[first] > 0, w, np.nan)


def ndcg_per_search(search: np.ndarray, starts: np.ndarray, rel: np.ndarray, score: np.ndarray,
                    k: int = TOP_K) -> np.ndarray:
    """NDCG@k for each search when its hotels are ordered by ``score``.

    ``rel`` is the linear gain of each row (5 / 1 / 0, or that divided by the chance the slot
    was looked at, see ``examination``). Ties in ``score`` are broken by row order, which does
    not depend on the ranking. Rows must be sorted by search. Searches with no gain get NaN.
    """
    rank = np.arange(len(search)) - starts[search]  # sorting within searches keeps the same blocks
    disc = np.where(rank < k, 1.0 / np.log2(rank + 2.0), 0.0)
    dcg = np.add.reduceat(rel[np.lexsort((-score, search))] * disc, starts)
    idcg = np.add.reduceat(rel[np.lexsort((-rel, search))] * disc, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(idcg > 0, dcg / idcg, np.nan)


def examination(fit_click, max_slot: int = 40) -> np.ndarray:
    """Relative chance that a hotel in each slot is looked at, slot 1 = 1 (index = slot).

    Taken from a click conditional logit with position dummies fitted on random-order searches
    (the phase 3 model): under a position-based click model, exp(position coefficient) is the
    examination probability relative to slot 1. Slots without a coefficient (the reserved ones)
    take the nearest earlier slot's value.
    """
    theta = np.full(max_slot + 1, np.nan)
    theta[1] = 1.0
    for name, b in zip(fit_click.names, fit_click.beta):
        if name.startswith("pos_") and int(name[4:]) <= max_slot:
            theta[int(name[4:])] = np.exp(b)
    filled = pd.Series(theta[1:]).ffill().bfill().to_numpy()
    return np.r_[np.nan, filled]


def shown_order_agreement(search: np.ndarray, slots: np.ndarray, score: np.ndarray) -> float:
    """Average within-search rank correlation between a ranking and the order actually shown.

    In truly random searches it is zero for any ranking built from hotel attributes. A ranking
    that agrees with the shown order is credited for clicks that position, not preference,
    produced; the examination weights remove that credit.
    """
    df = pd.DataFrame({"s": search, "top": -np.asarray(slots, dtype=float), "score": score})
    for src, dst in (("top", "a"), ("score", "b")):
        r = df.groupby("s")[src].rank()
        df[dst] = r - r.groupby(df["s"]).transform("mean")
    num = (df["a"] * df["b"]).groupby(df["s"]).sum()
    den = np.sqrt((df["a"] ** 2).groupby(df["s"]).sum() * (df["b"] ** 2).groupby(df["s"]).sum())
    return float((num / den.replace(0, np.nan)).mean())


def booked_rank(search: np.ndarray, starts: np.ndarray, booked: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Rank (1 = top) the booked hotel would get under ``score``; NaN for searches with no
    booking. Rows must be sorted by search."""
    order = np.lexsort((-score, search))
    rank = np.empty(len(search))
    rank[order] = np.arange(len(search)) - starts[search[order]] + 1
    out = np.full(len(starts), np.nan)
    hit = np.flatnonzero(booked > 0)
    out[search[hit]] = rank[hit]
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of a - b over searches, with a percentile interval from resampling searches."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(ok.sum())}
