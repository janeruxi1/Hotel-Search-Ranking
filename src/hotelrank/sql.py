"""The audit's search-level work, in SQL run by DuckDB.

Notebook 01 uses these queries to build the per-search table, the data-quality flags and the
analysis sample that every later notebook reads. The SQL files are plain DuckDB SQL that also run
from the DuckDB CLI once a view called ``impressions`` exists; this module creates that view,
loads the shared definitions and returns results in the shapes the notebooks use.
``audit.per_search`` / ``audit.price_flags`` are the independent pandas implementations that
tests/test_sql.py checks them against.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import ROOT

SQL_DIR = ROOT / "sql"
QUERIES = ("01_arm_summary", "02_reserved_slots", "03_analysis_sample")


def connect(source: Path | pd.DataFrame):
    """A DuckDB connection with ``impressions`` and the shared views from 00_views.sql."""
    import duckdb

    con = duckdb.connect()
    if isinstance(source, pd.DataFrame):
        con.register("impressions_df", source)
        con.execute("CREATE VIEW impressions AS SELECT * FROM impressions_df")
    else:
        path = str(Path(source).resolve()).replace("'", "''")
        con.execute(f"CREATE VIEW impressions AS SELECT * FROM read_parquet('{path}')")
    con.execute((SQL_DIR / "00_views.sql").read_text(encoding="utf-8"))
    return con


def query(con, name: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")).df()


def search_flags(con) -> pd.DataFrame:
    """One row per search, indexed by srch_id: random, n, max_position, clicks, bookings,
    bad_price, price_rho, price_sorted (the columns of audit.per_search + audit.price_flags)."""
    f = con.execute("SELECT srch_id, random, n, max_position, clicks, bookings, bad_price, price_rho, price_sorted "
                    "FROM search_flags ORDER BY srch_id").df().set_index("srch_id")
    f.index.name = None
    for c in ("n", "max_position", "clicks", "bookings"):
        f[c] = f[c].astype("int64")
    return f


def arm_summary(con) -> dict:
    """The dict audit.arm_summary returns, from sql/01_arm_summary.sql."""
    out = {}
    for r in query(con, "01_arm_summary").itertuples():
        out[r.arm] = {"searches": int(r.searches), "impressions": int(r.impressions),
                      "share_of_searches": float(r.share_of_searches),
                      "searches_with_click": float(r.searches_with_click),
                      "searches_with_booking": float(r.searches_with_booking),
                      "clicks_per_search": float(r.clicks_per_search), "median_results": float(r.median_results)}
    return out


def reserved_slots(con) -> dict:
    """{arm: {slot: impressions relative to the two neighbouring slots}}, from sql/02_reserved_slots.sql."""
    out: dict = {}
    for r in query(con, "02_reserved_slots").itertuples():
        out.setdefault(r.arm, {})[int(r.position)] = float(r.vs_neighbours)
    return out


def analysis_sample(con) -> dict:
    """Search counts behind the analysis sample, from sql/03_analysis_sample.sql."""
    r = query(con, "03_analysis_sample").iloc[0]
    return {"searches_bad_price": int(r["searches_bad_price"]),
            "searches_price_sorted": int(r["searches_price_sorted"]),
            "searches_kept": {"algorithm": int(r["kept_algorithm"]), "random": int(r["kept_random"])},
            "impressions_kept": int(r["impressions_kept"])}
