"""Position effects: raw curves by arm with search-clustered bootstrap intervals, and a
within-search linear probability model that adjusts for hotel attributes.

Everything is vectorized over integer search codes; the real algorithm arm has about 6.6M
impressions and 267k searches.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import RESERVED_SLOTS

CONTROLS = ["prop_starrating", "prop_review_score", "no_reviews", "prop_brand_bool",
            "prop_location_score1", "rel_log_price", "promotion_flag"]


def analysis_sample(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Keep clean searches (no bad price, not sorted by price), drop the reserved slots and add
    the derived columns used by the models."""
    f = flags.set_index("srch_id") if "srch_id" in flags.columns else flags
    keep = ~f["bad_price"] & ~f["price_sorted"].fillna(False).astype(bool)
    x = df[keep.reindex(df["srch_id"]).to_numpy() & ~df["position"].isin(RESERVED_SLOTS).to_numpy()].copy()
    lp = np.log(x["price_usd"].astype(float))
    x["rel_log_price"] = (lp - lp.groupby(x["srch_id"]).transform("mean")).astype(np.float32)
    rev = x["prop_review_score"].astype(np.float32)
    x["no_reviews"] = (rev.isna() | (rev == 0)).astype(np.float32)
    x["prop_review_score"] = rev.fillna(0.0)
    for c in ("prop_brand_bool", "promotion_flag", "click_bool", "booking_bool"):
        x[c] = x[c].astype(np.float32)
    return x


def _codes(s: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(s, sort=False)
    return codes, len(uniq)


def _group_sum(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum rows of ``values`` (1-D or 2-D) by group code. Uses ``np.add.reduceat`` when the
    rows of each group are contiguous (true for this data, which is ordered by search) and
    falls back to ``np.add.at`` otherwise."""
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    out_shape = (n_groups,) + values.shape[1:]
    out = np.zeros(out_shape)
    np.add.at(out, codes[starts], np.add.reduceat(values, starts, axis=0))
    return out


def balanced_panel(x: pd.DataFrame, max_position: int = 30) -> pd.DataFrame:
    """Searches that show a hotel in every non-reserved slot from 1 to ``max_position``,
    restricted to those slots. Every position is then measured on the same searches, so the
    curve is not distorted by long result lists having lower click rates per hotel."""
    slots = [p for p in range(1, max_position + 1) if p not in RESERVED_SLOTS]
    y = x[x["position"] <= max_position]
    full = y.groupby("srch_id")["position"].transform("nunique") == len(slots)
    return y[full.to_numpy()]


def rate_by_position(x: pd.DataFrame, outcome: str, n_boot: int = 200, seed: int = 0,
                     max_position: int = 38) -> pd.DataFrame:
    """Rate of ``outcome`` at each position with a 95% interval from a search-level (Poisson)
    bootstrap, plus the rate relative to position 1 with its own interval.

    On a balanced panel (every search has one row per slot) the bootstrap is a single matrix
    product; otherwise it loops over replicates with ``np.bincount``.
    """
    x = x[x["position"] <= max_position]
    s_codes, n_s = _codes(x["srch_id"])
    pos = x["position"].to_numpy().astype(int)
    y = x[outcome].to_numpy().astype(float)
    P = max_position + 1
    n = np.bincount(pos, minlength=P).astype(float)
    k = np.bincount(pos, weights=y, minlength=P)
    rng = np.random.default_rng(seed)
    present = n > 0
    if np.all(n[present] == n_s):  # balanced: one row per search and slot
        Y = np.zeros((n_s, P), dtype=np.float32)
        Y[s_codes, pos] = y
        boot = np.empty((n_boot, P))
        for a in range(0, n_boot, 25):  # blocks of replicates keep the weight matrix small
            W = rng.poisson(1.0, (min(25, n_boot - a), n_s)).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[a:a + len(W)] = (W @ Y) / W.sum(axis=1, keepdims=True)
        boot[:, ~present] = np.nan
    else:
        boot = np.empty((n_boot, P))
        for b in range(n_boot):
            w = rng.poisson(1.0, n_s).astype(float)[s_codes]
            nb = np.bincount(pos, weights=w, minlength=P)
            kb = np.bincount(pos, weights=w * y, minlength=P)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[b] = kb / nb
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty slots give all-NaN columns
        rate = k / n
        rel_boot = boot / boot[:, [1]]
        lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
        rel_lo, rel_hi = np.nanpercentile(rel_boot, 2.5, axis=0), np.nanpercentile(rel_boot, 97.5, axis=0)
    out = pd.DataFrame({
        "position": np.arange(P), "impressions": n.astype(int), "events": k.astype(int), "rate": rate,
        "lo": lo, "hi": hi, "rel_to_1": rate / rate[1], "rel_lo": rel_lo, "rel_hi": rel_hi,
    })
    out = out[(out["position"] >= 1) & (out["impressions"] > 0) & ~out["position"].isin(RESERVED_SLOTS)]
    return out.reset_index(drop=True)


def ratio_summary(curve: pd.DataFrame, ks=(10, 20, 30)) -> dict:
    c = curve.set_index("position")
    return {f"pos1_over_pos{k}": {"ratio": round(float(1 / c.loc[k, "rel_to_1"]), 3),
                                  "lo": round(float(1 / c.loc[k, "rel_hi"]), 3),
                                  "hi": round(float(1 / c.loc[k, "rel_lo"]), 3)} for k in ks if k in c.index}


def within_search_lpm(x: pd.DataFrame, outcome: str, controls: list[str] | None = None,
                      position: str = "dummies", max_position: int = 38, chunk: int = 200_000) -> dict:
    """OLS of ``outcome`` on position (dummies relative to position 1, or a linear term) and
    optional hotel controls, with search fixed effects (by demeaning) and standard errors
    clustered by search.

    The design matrix is never held in memory whole: group means are computed one column at
    a time and X'X, X'y and the per-search scores are accumulated over row chunks, so the
    full algorithm arm (about 6.7M rows) fits on a laptop.
    """
    x = x[x["position"] <= max_position]
    codes, G = _codes(x["srch_id"])
    cnt = _group_sum(np.ones(len(codes)), codes, G)
    posv = x["position"].to_numpy()
    yraw = x[outcome].to_numpy().astype(np.float64)
    ctrl = {c: x[c].to_numpy().astype(np.float64) for c in controls or []}

    if position == "dummies":
        levels = sorted(int(p) for p in np.unique(posv) if p != 1)
        names = [f"pos_{p}" for p in levels]
    else:
        levels, names = [], ["position"]
    names = names + list(ctrl)

    def column(j: int, sl=slice(None)) -> np.ndarray:
        nm = names[j]
        if nm.startswith("pos_"):
            return (posv[sl] == int(nm[4:])).astype(np.float64)
        if nm == "position":
            return posv[sl].astype(np.float64)
        return ctrl[nm][sl]

    k = len(names)
    means = np.column_stack([_group_sum(column(j), codes, G) / cnt for j in range(k)])
    ymean = _group_sum(yraw, codes, G) / cnt
    n = len(yraw)

    def chunk_X(sl):
        return np.column_stack([column(j, sl) for j in range(k)]) - means[codes[sl]]

    XtX = np.zeros((k, k))
    Xty = np.zeros(k)
    ss = np.zeros(k)
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        X = chunk_X(sl)
        y = yraw[sl] - ymean[codes[sl]]
        XtX += X.T @ X
        Xty += X.T @ y
        ss += (X * X).sum(axis=0)
    ok = ss > 1e-9 * n  # drop columns with no within-search variation
    dropped = [nm for nm, keep in zip(names, ok) if not keep]
    idx = np.flatnonzero(ok)
    XtX, Xty = XtX[np.ix_(idx, idx)], Xty[idx]
    beta = np.linalg.solve(XtX, Xty)
    S = np.zeros((G, len(idx)))
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        X = chunk_X(sl)[:, idx]
        e = (yraw[sl] - ymean[codes[sl]]) - X @ beta
        S += _group_sum(X * e[:, None], codes[sl], G)
    inv = np.linalg.inv(XtX)
    kk = len(idx)
    corr = G / (G - 1) * (n - 1) / max(n - kk - G, 1)
    V = corr * inv @ (S.T @ S) @ inv
    return {"names": [names[i] for i in idx], "beta": beta, "se": np.sqrt(np.diag(V)), "n": int(n),
            "searches": int(G), "dropped": dropped, "mean_outcome": float(yraw.mean()),
            "mean_at_pos1": float(yraw[posv == 1].mean())}


def lpm_curve(fit: dict) -> pd.DataFrame:
    """Position coefficients from a dummies fit as a curve: pp change vs position 1 with 95% CI."""
    rows = [(1, 0.0, 0.0)]
    for nm, b, s in zip(fit["names"], fit["beta"], fit["se"]):
        if nm.startswith("pos_"):
            rows.append((int(nm[4:]), b, s))
    c = pd.DataFrame(rows, columns=["position", "diff", "se"]).sort_values("position")
    c["lo"], c["hi"] = c["diff"] - 1.96 * c["se"], c["diff"] + 1.96 * c["se"]
    base = fit["mean_at_pos1"]
    c["rel_to_1"] = (base + c["diff"]) / base
    return c.reset_index(drop=True)


def sample_searches(x: pd.DataFrame, n_searches: int, seed: int = 0) -> pd.DataFrame:
    ids = x["srch_id"].drop_duplicates()
    if len(ids) <= n_searches:
        return x
    keep = ids.sample(n_searches, random_state=seed)
    return x[x["srch_id"].isin(keep)]
