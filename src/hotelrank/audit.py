"""Data-audit helpers. All are vectorized: the real table has about 10M rows and 400k searches."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PRICE_CAP, PRICE_SORT_RHO, RESERVED_SLOTS

ARM_NAMES = {False: "algorithm", True: "random"}


def per_search(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("srch_id", sort=False)
    return pd.DataFrame({
        "random": g["random_bool"].first(),
        "n": g.size(),
        "max_position": g["position"].max(),
        "clicks": g["click_bool"].sum(),
        "bookings": g["booking_bool"].sum(),
    })


def arm_summary(searches: pd.DataFrame) -> dict:
    out = {}
    for flag, name in ARM_NAMES.items():
        s = searches[searches["random"] == flag]
        out[name] = {
            "searches": int(len(s)),
            "impressions": int(s["n"].sum()),
            "share_of_searches": round(len(s) / len(searches), 4),
            "searches_with_click": round(float((s["clicks"] > 0).mean()), 4),
            "searches_with_booking": round(float((s["bookings"] > 0).mean()), 4),
            "clicks_per_search": round(float(s["clicks"].mean()), 4),
            "median_results": float(s["n"].median()),
        }
    return out


def slot_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Impressions per position and arm, and each slot's count relative to its neighbours."""
    t = df.groupby(["random_bool", "position"]).size().unstack(0, fill_value=0)
    t.columns = [ARM_NAMES[c] for c in t.columns]
    t = t.reindex(range(1, int(df["position"].max()) + 1), fill_value=0)
    for arm in list(t.columns):
        neighbours = (t[arm].shift(1) + t[arm].shift(-1)) / 2
        t[f"{arm}_vs_neighbours"] = (t[arm] / neighbours).round(4)
    return t


def gaps_per_search(searches: pd.DataFrame) -> dict:
    gaps = (searches["max_position"] - searches["n"]).clip(lower=0)
    return {int(k): int(v) for k, v in gaps.value_counts().sort_index().items()}


def within_search_rank_corr(df: pd.DataFrame, col: str, min_n: int = 5) -> pd.Series:
    """Per-search Pearson correlation between the rank of ``position`` and the rank of ``col``.

    +1 means the column rises perfectly with position (e.g. results sorted by price ascending).
    Searches with fewer than ``min_n`` valid rows, or a constant column, get NaN.
    """
    x = df[["srch_id", "position", col]].dropna()
    rp = x.groupby("srch_id")["position"].rank()
    rc = x.groupby("srch_id")[col].rank()
    t = pd.DataFrame({"s": x["srch_id"].to_numpy(), "a": rp.to_numpy(), "b": rc.to_numpy()})
    t["ab"], t["aa"], t["bb"] = t.a * t.b, t.a * t.a, t.b * t.b
    g = t.groupby("s").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"), ab=("ab", "sum"),
                           aa=("aa", "sum"), bb=("bb", "sum"))
    cov = g.ab - g.a * g.b / g.n
    va, vb = g.aa - g.a ** 2 / g.n, g.bb - g.b ** 2 / g.n
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    r[(g.n < min_n) | (vb <= 1e-12)] = np.nan
    return r


def price_flags(df: pd.DataFrame, searches: pd.DataFrame) -> pd.DataFrame:
    """Add per-search flags: any price outside (0, PRICE_CAP]; results sorted by price."""
    bad = (df["price_usd"] <= 0) | (df["price_usd"] > PRICE_CAP)
    out = searches.copy()
    out["bad_price"] = bad.groupby(df["srch_id"]).any().reindex(out.index, fill_value=False)
    rho = within_search_rank_corr(df.loc[~bad], "price_usd")
    out["price_rho"] = rho.reindex(out.index)
    out["price_sorted"] = out["price_rho"] >= PRICE_SORT_RHO
    return out


def price_summary(df: pd.DataFrame) -> dict:
    p = df["price_usd"]
    q = p.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    med = p.groupby(df["srch_id"]).transform("median")
    return {
        "quantiles": {f"{k:g}": round(float(v), 2) for k, v in q.items()},
        "max": float(p.max()),
        "impressions_price_le_0": int((p <= 0).sum()),
        "share_above_cap": round(float((p > PRICE_CAP).mean()), 6),
        "cap": PRICE_CAP,
        "share_above_10x_search_median": round(float((p > 10 * med).mean()), 6),
        "median_by_length_of_stay": {int(k): round(float(v), 1) for k, v in
                                     p.groupby(df["srch_length_of_stay"].clip(upper=7)).median().items()},
    }


def balance(df: pd.DataFrame, cols: list[str], keep: pd.Series | None = None) -> dict:
    """Within-search correlation between position and each column, per arm.

    Each column is demeaned within its search, so differences between markets cancel and
    only the ordering inside a search remains. Under true random ordering every value is
    close to 0.
    """
    x = df if keep is None else df[keep.reindex(df["srch_id"]).to_numpy()]
    out = {}
    pos = x["position"] - x.groupby("srch_id")["position"].transform("mean")
    for flag, name in ARM_NAMES.items():
        m = (x["random_bool"] == flag).to_numpy()
        res = {}
        for c in cols:
            v = x[c].astype(float)
            v = v - v.groupby(x["srch_id"]).transform("mean")
            ok = m & v.notna().to_numpy()
            res[c] = round(float(np.corrcoef(pos.to_numpy()[ok], v.to_numpy()[ok])[0, 1]), 4)
        out[name] = res
    return out


def balance_by_bucket(df: pd.DataFrame, col: str, arm: bool, keep: pd.Series | None = None,
                      bins=(0, 4, 10, 20, 30, 40)) -> pd.Series:
    """Mean of ``col`` minus its search mean, by position bucket, for one arm."""
    x = df if keep is None else df[keep.reindex(df["srch_id"]).to_numpy()]
    x = x[x["random_bool"] == arm]
    v = x[col].astype(float) - x.groupby("srch_id")[col].transform("mean")
    b = pd.cut(x["position"], list(bins))
    return v.groupby(b, observed=True).mean()


def missing_share(df: pd.DataFrame) -> dict:
    m = df.isna().mean()
    return {k: round(float(v), 4) for k, v in m[m > 0].sort_values(ascending=False).items()}


def reserved_slot_share(occupancy: pd.DataFrame) -> dict:
    out = {}
    for arm in ("algorithm", "random"):
        out[arm] = {int(s): float(occupancy.loc[s, f"{arm}_vs_neighbours"]) for s in RESERVED_SLOTS
                    if s in occupancy.index}
    return out
