"""Loading the prepared impressions table, or a synthetic stand-in when it is absent.

The real table (``data/interim/train.parquet``) is built by ``scripts/prepare_data.py`` from
Kaggle data that may not be redistributed, so it never reaches GitHub. CI and anyone without
the data get the synthetic table from :mod:`hotelrank.synthetic`, which has the same columns.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "interim" / "train.parquet"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

# Slots Expedia left (almost) empty in this release; see notebook 01.
RESERVED_SLOTS = (5, 11, 17, 23)
# Prices outside (0, PRICE_CAP] are treated as data errors (notebook 01 records how many).
PRICE_CAP = 5_000.0
# A search whose prices rise with position this consistently was sorted by price, not at random.
PRICE_SORT_RHO = 0.95


def real_data_available(path: Path = PARQUET) -> bool:
    """True when the prepared real table exists and HOTELRANK_SYNTHETIC is not set."""
    return Path(path).exists() and os.environ.get("HOTELRANK_SYNTHETIC", "") not in ("1", "true", "yes")


def load(columns: list[str] | None = None, path: Path = PARQUET, synthetic_if_missing: bool = True,
         seed: int = 0) -> tuple[pd.DataFrame, str]:
    """Return ``(impressions, source)`` where source is ``"real"`` or ``"synthetic"``."""
    if real_data_available(path):
        return pd.read_parquet(path, columns=columns), "real"
    if not synthetic_if_missing:
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py (see data/raw/README.md)")
    from .synthetic import make_impressions

    df = make_impressions(seed=seed)
    return (df[columns] if columns else df), "synthetic"
