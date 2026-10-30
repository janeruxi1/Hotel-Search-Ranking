"""Sizing the online test that confirms (or refutes) the offline ranking results.

Formulas are the standard normal-approximation ones; `simulate_power` checks them by Monte Carlo.
Per-search metrics: booking (0/1) is the primary metric; booking value per search is a
non-inferiority guardrail, as in the phase-4 decision rule.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _z(q: float) -> float:
    return float(norm.ppf(q))


def n_two_proportions(p0: float, rel_lift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Searches per arm to detect p0 -> p0 * (1 + rel_lift) with a two-sided test at `alpha`."""
    p1 = p0 * (1 + rel_lift)
    if not (0 < p0 < 1 and 0 < p1 < 1) or rel_lift == 0:
        raise ValueError("need 0 < p0, p1 < 1 and a non-zero lift")
    pbar = (p0 + p1) / 2
    num = _z(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar)) + _z(power) * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return math.ceil((num / (p1 - p0)) ** 2)


def power_two_proportions(p0: float, rel_lift: float, n: int, alpha: float = 0.05) -> float:
    """Power of the two-sided two-proportion test with n searches per arm."""
    p1 = p0 * (1 + rel_lift)
    pbar = (p0 + p1) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
    return float(norm.cdf((abs(p1 - p0) - _z(1 - alpha / 2) * se0) / se1))


def mde_two_proportions(p0: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest relative lift detectable with n searches per arm (bisection on the power function)."""
    lo, hi = 1e-6, 1 / p0 - 1 - 1e-9
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if power_two_proportions(p0, mid, n, alpha) >= power else (mid, hi)
    return hi


def n_noninferiority_mean(mean: float, sd: float, margin_rel: float, alpha: float = 0.025,
                          power: float = 0.8, true_rel_diff: float = 0.0) -> int:
    """Searches per arm to show treatment mean > control mean * (1 - margin_rel), one-sided `alpha`.

    Assumes equal variance in both arms and a true relative difference `true_rel_diff`.
    """
    gap = (true_rel_diff + margin_rel) * mean
    if gap <= 0:
        raise ValueError("the true difference must lie above the margin")
    return math.ceil(2 * (sd * (_z(1 - alpha) + _z(power)) / gap) ** 2)


def per_search_value_stats(value: np.ndarray, cap_quantile: float | None = None) -> dict:
    """Mean, sd and coefficient of variation of booking value per search, optionally capped."""
    v = np.asarray(value, dtype=float)
    cap = None
    if cap_quantile is not None:
        booked = v[v > 0]
        cap = float(np.quantile(booked, cap_quantile)) if booked.size else 0.0
        v = np.minimum(v, cap)
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    return {"mean": mean, "sd": sd, "cv": sd / mean if mean else float("nan"), "cap": cap}


def simulate_power(p0: float, rel_lift: float, n: int, alpha: float = 0.05, n_sim: int = 4000,
                   seed: int = 0) -> float:
    """Monte Carlo power of the pooled two-proportion z-test, to check the closed form."""
    rng = np.random.default_rng(seed)
    p1 = p0 * (1 + rel_lift)
    x0 = rng.binomial(n, p0, n_sim)
    x1 = rng.binomial(n, p1, n_sim)
    pooled = (x0 + x1) / (2 * n)
    se = np.sqrt(2 * pooled * (1 - pooled) / n)
    z = (x1 - x0) / n / se
    return float(np.mean(np.abs(z) > _z(1 - alpha / 2)))


def size_design(p0: float, rel_lift: float, value_mean: float, value_sd: float, n_arms: int,
                n_comparisons: int, margin: float = 0.02, power: float = 0.8, family_alpha: float = 0.05,
                guardrail_alpha: float = 0.025) -> dict:
    """Per-arm and total size of an equal-allocation test, as in phase 5.

    Bookings: each of ``n_comparisons`` pairwise comparisons gets two-sided alpha
    family_alpha / n_comparisons (Bonferroni; the analysis uses Holm, which is never less
    powerful). Value guardrail: one-sided non-inferiority at ``guardrail_alpha`` with no further
    adjustment, because an arm ships only if it passes both (an intersection-union test).
    The arm size is the larger of the two requirements.
    """
    alpha = family_alpha / n_comparisons
    n_book = n_two_proportions(p0, rel_lift, alpha, power)
    n_guard = n_noninferiority_mean(value_mean, value_sd, margin, guardrail_alpha, power)
    n_arm = max(n_book, n_guard)
    return {"arms": n_arms, "comparisons": n_comparisons, "alpha_per_comparison": alpha,
            "n_booking": n_book, "n_guardrail": n_guard, "n_per_arm": n_arm, "n_total": n_arms * n_arm,
            "binding": "guardrail" if n_guard >= n_book else "booking",
            "mde_at_n_per_arm": mde_two_proportions(p0, n_arm, alpha, power)}
