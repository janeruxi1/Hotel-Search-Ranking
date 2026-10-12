"""Conditional logit: which hotel gets chosen within a search.

Each search with exactly one chosen hotel is one observation: the probability that hotel i
is chosen is exp(u_i) / sum_j exp(u_j) over the hotels shown in that search, with
u_i = x_i' beta. Anything common to the whole search (date, destination, how many results,
how likely the user was to click at all) cancels out of that ratio. That suits this data:
the release contains only searches with a click, and the model conditions on exactly that.

The fit is a hand-written Newton routine with analytic gradient and information matrix,
because the random arm alone has about 2.6M rows. Position enters as a categorical index, so
its 30-odd dummies are never materialised; group operations use ``np.add.reduceat`` over rows
sorted by search.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import RESERVED_SLOTS

ATTRIBUTES = ["log_price", "hotel_mean_log_price", "prop_starrating", "prop_review_score", "no_reviews",
              "prop_brand_bool", "prop_location_score1", "promotion_flag"]


@dataclass
class ChoiceData:
    """Rows sorted by search. ``X`` holds the attribute columns; position enters through a
    categorical index ``cat`` (0 = position 1, the reference) so the 30-odd position dummies
    never have to be materialised."""
    X: np.ndarray                 # (rows, k_attr) float64
    cat: np.ndarray               # (rows,) int, 0 = reference level
    levels: list[int]             # position of each non-reference level, cat = 1..len(levels)
    chosen: np.ndarray            # (rows,) bool, exactly one True per search
    starts: np.ndarray            # first row of each search
    attr_names: list[str]
    search_ids: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_searches(self) -> int:
        return len(self.starts)

    @property
    def names(self) -> list[str]:
        return list(self.attr_names) + [f"pos_{p}" for p in self.levels]

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(np.r_[self.starts, len(self.chosen)])

    @property
    def search_of_row(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_searches), self.sizes)


def add_hotel_mean_price(x: pd.DataFrame, full: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``log_price`` and the hotel's average log price over all its impressions
    (``hotel_mean_log_price``). With both in the model, the price coefficient is identified by
    the same hotel being cheaper or dearer than usual (the Mundlak device), not by cheap hotels
    differing from expensive ones."""
    src = x if full is None else full
    means = np.log(src["price_usd"].astype(float)).groupby(src["prop_id"]).mean()
    x = x.copy()
    x["log_price"] = np.log(x["price_usd"].astype(float)).astype(np.float32)
    x["hotel_mean_log_price"] = x["prop_id"].map(means).astype(np.float32)
    return x


def build(x: pd.DataFrame, outcome: str, attributes: list[str] | None = None, position_dummies: bool = True,
          max_position: int = 38) -> ChoiceData:
    """Design for searches in ``x`` where exactly one hotel has ``outcome`` == 1."""
    attributes = ATTRIBUTES if attributes is None else attributes
    x = x[(x["position"] <= max_position) & ~x["position"].isin(RESERVED_SLOTS)]
    y = x[outcome].astype(bool)
    n_pos = y.groupby(x["srch_id"]).transform("sum")
    x = x[(n_pos == 1).to_numpy()].sort_values(["srch_id", "position"], kind="stable")
    pos = x["position"].to_numpy().astype(int)
    if position_dummies:
        levels = sorted(int(p) for p in np.unique(pos) if p != 1)
        lookup = np.zeros(max(pos.max(), max_position) + 1, dtype=np.int64)
        for i, p in enumerate(levels, start=1):
            lookup[p] = i
        cat = lookup[pos]
    else:
        levels, cat = [], np.zeros(len(x), dtype=np.int64)
    # Attributes that never vary carry no information (and would make the fit singular).
    dropped = [a for a in attributes if x[a].astype(float).std() == 0]
    attributes = [a for a in attributes if a not in dropped]
    X = np.column_stack([x[a].to_numpy().astype(np.float64) for a in attributes]) if attributes else np.zeros((len(x), 0))
    sid = x["srch_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    return ChoiceData(X=X, cat=cat, levels=levels, chosen=x[outcome].to_numpy().astype(bool), starts=starts,
                      attr_names=list(attributes), search_ids=sid[starts],
                      meta={"rows": int(len(x)), "searches": int(len(starts)), "dropped": dropped})


def subset(d: ChoiceData, keep_search: np.ndarray) -> ChoiceData:
    """Keep the searches where ``keep_search`` (one flag per search) is True."""
    row_keep = np.repeat(keep_search, d.sizes)
    new_sizes = d.sizes[keep_search]
    starts = np.r_[0, np.cumsum(new_sizes)[:-1]].astype(int)
    return ChoiceData(X=d.X[row_keep], cat=d.cat[row_keep], levels=d.levels, chosen=d.chosen[row_keep],
                      starts=starts, attr_names=d.attr_names, search_ids=d.search_ids[keep_search],
                      meta={"rows": int(row_keep.sum()), "searches": int(keep_search.sum())})


def _split(d: ChoiceData, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = d.X.shape[1]
    return beta[:k], np.r_[0.0, beta[k:]]


def utilities(d: ChoiceData, beta: np.ndarray) -> np.ndarray:
    b_attr, b_pos = _split(d, beta)
    return d.X @ b_attr + b_pos[d.cat]


def probabilities(d: ChoiceData, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choice probability of every row, and log-sum-exp per search."""
    u = utilities(d, beta)
    sizes = d.sizes
    umax = np.maximum.reduceat(u, d.starts)
    e = np.exp(u - np.repeat(umax, sizes))
    s = np.add.reduceat(e, d.starts)
    return e / np.repeat(s, sizes), umax + np.log(s)


def loglik(d: ChoiceData, beta: np.ndarray) -> float:
    """Sum of log choice probabilities, computed as u - logsumexp so it never takes log(0)."""
    _, lse = probabilities(d, beta)
    return float(utilities(d, beta)[d.chosen].sum() - lse.sum())


def _score_parts(d: ChoiceData, w: np.ndarray) -> np.ndarray:
    """X' w for the full parameter vector (attributes, then position levels)."""
    L = len(d.levels)
    pos_part = np.bincount(d.cat, weights=w, minlength=L + 1)[1:]
    return np.r_[d.X.T @ w, pos_part]


@dataclass
class ChoiceFit:
    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    loglik: float
    loglik_null: float
    n_searches: int
    rows: int
    converged: bool

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"beta": self.beta, "se": self.se}, index=self.names)

    @property
    def mcfadden_r2(self) -> float:
        return 1 - self.loglik / self.loglik_null


def _information(d: ChoiceData, p: np.ndarray) -> np.ndarray:
    """Observed information sum_s [Z_s' diag(p_s) Z_s - (Z_s' p_s)(Z_s' p_s)'], Z = [X, position one-hot]."""
    ka, L = d.X.shape[1], len(d.levels)
    k = ka + L
    H = np.zeros((k, k))
    Xp = d.X * p[:, None]
    H[:ka, :ka] = Xp.T @ d.X
    for j in range(ka):
        H[j, ka:] = np.bincount(d.cat, weights=Xp[:, j], minlength=L + 1)[1:]
    H[ka:, :ka] = H[:ka, ka:].T
    H[ka:, ka:] = np.diag(np.bincount(d.cat, weights=p, minlength=L + 1)[1:])
    M = np.zeros((d.n_searches, k))
    for j in range(ka):
        M[:, j] = np.add.reduceat(Xp[:, j], d.starts)
    if L:
        mpos = np.zeros((d.n_searches, L + 1))
        np.add.at(mpos, (d.search_of_row, d.cat), p)
        M[:, ka:] = mpos[:, 1:]
    return H - M.T @ M


def fit(d: ChoiceData, max_iter: int = 50, tol: float = 1e-9, ridge: float = 0.0) -> ChoiceFit:
    """Maximum-likelihood conditional logit by Newton's method with step halving (the
    log-likelihood is concave, so this converges in a handful of iterations). Standard errors
    come from the observed information at the optimum.

    ``ridge`` adds a small L2 penalty, 0.5 * ridge * |beta|^2. With thousands of choices it is
    negligible; it keeps the fit finite when a rare level is never chosen in a small sample
    (otherwise its coefficient runs off to minus infinity). The reported log-likelihood is
    always the unpenalised one."""
    k = d.X.shape[1] + len(d.levels)
    xc = _score_parts(d, d.chosen.astype(np.float64))
    b = np.zeros(k)

    def objective(beta):
        return loglik(d, beta) - 0.5 * ridge * beta @ beta

    obj = objective(b)
    converged = False
    for _ in range(max_iter):
        p, _ = probabilities(d, b)
        grad = xc - _score_parts(d, p) - ridge * b
        H = _information(d, p) + ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:  # e.g. a position level never chosen in a tiny sample
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        while True:
            new = b + t * step
            obj_new = objective(new)
            if obj_new >= obj - 1e-12 or t < 1e-6:
                break
            t /= 2
        b, gain, obj = new, obj_new - obj, obj_new
        if abs(gain) < tol * max(1.0, abs(obj)) and np.max(np.abs(t * step)) < 1e-6:
            converged = True
            break
    p, _ = probabilities(d, b)
    info = _information(d, p) + ridge * np.eye(k)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov, converged = np.linalg.pinv(info), False
    return ChoiceFit(names=d.names, beta=b, cov=cov, loglik=loglik(d, b),
                     loglik_null=float(-np.log(d.sizes).sum()), n_searches=d.n_searches, rows=len(p),
                     converged=converged)


def ratio_with_se(fit_: ChoiceFit, num: str | list[tuple[str, float]], den: str) -> tuple[float, float]:
    """``-(a' beta) / beta_den`` and its delta-method standard error, where ``num`` is a
    coefficient name or a linear combination [(name, weight), ...]."""
    w = np.zeros(len(fit_.names))
    for nm, wt in ([(num, 1.0)] if isinstance(num, str) else num):
        w[fit_.names.index(nm)] += wt
    j = fit_.names.index(den)
    a, bd = float(w @ fit_.beta), float(fit_.beta[j])
    g = -w / bd
    g[j] += a / bd ** 2
    return -a / bd, float(np.sqrt(g @ fit_.cov @ g))


def price_equivalent(log_change: float, price: float) -> dict:
    """Turn a utility gain expressed in log-price units into the equivalent price cut."""
    pct = 1 - np.exp(-log_change)
    return {"log_units": round(log_change, 4), "pct_price_cut": round(100 * pct, 2), "usd": round(price * pct, 2)}


def holdout_metrics(d: ChoiceData, beta: np.ndarray) -> dict:
    p, _ = probabilities(d, beta)
    sizes = np.diff(np.r_[d.starts, len(p)])
    ll = float(np.log(p[d.chosen]).sum())
    ll0 = float(-np.log(sizes).sum())
    # top-1: does the highest-probability hotel in each search match the chosen one?
    pmax = np.maximum.reduceat(p, d.starts)
    top1 = float(np.mean(p[d.chosen] >= pmax[np.searchsorted(d.starts, np.flatnonzero(d.chosen), side="right") - 1]))
    return {"searches": int(len(sizes)), "loglik_per_search": round(ll / len(sizes), 5),
            "null_loglik_per_search": round(ll0 / len(sizes), 5), "mcfadden_r2": round(1 - ll / ll0, 4),
            "top1_hit_rate": round(top1, 4), "uniform_top1": round(float(np.mean(1 / sizes)), 4)}


def calibration(d: ChoiceData, beta: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed choice share against predicted probability, by decile of prediction."""
    p, _ = probabilities(d, beta)
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": p, "obs": d.chosen.astype(float), "bin": q}).groupby("bin").mean()


def simulate(n_searches: int, beta: dict, seed: int = 0, size: tuple[int, int] = (10, 30)) -> pd.DataFrame:
    """Searches whose single choice follows a conditional logit with the given coefficients
    (used by tests to check that ``fit`` recovers them)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_searches):
        n = int(rng.integers(*size))
        X = {nm: rng.normal(0, 1, n) for nm in beta}
        u = sum(beta[nm] * X[nm] for nm in beta)
        p = np.exp(u - u.max())
        p /= p.sum()
        c = rng.choice(n, p=p)
        rows.append(pd.DataFrame({"srch_id": s, "position": 1,
                                  "chosen": np.arange(n) == c, **X}))
    return pd.concat(rows, ignore_index=True)
