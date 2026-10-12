"""Notebook 01 builds its search-level tables in SQL; each must match the independent pandas
implementation in hotelrank.audit exactly (on synthetic data)."""
import numpy as np
import pytest

pytest.importorskip("duckdb")

from hotelrank import audit, position, sql  # noqa: E402
from hotelrank.data import PRICE_CAP, PRICE_SORT_RHO, RESERVED_SLOTS  # noqa: E402
from hotelrank.synthetic import make_impressions  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df = make_impressions(n_searches=3000, seed=2)
    return df, sql.connect(df)


def test_sql_parameters_match_the_python_constants(data):
    _, con = data
    p = con.execute("SELECT price_cap, price_sort_rho FROM params").fetchone()
    assert p == (PRICE_CAP, PRICE_SORT_RHO)
    slots = tuple(r[0] for r in con.execute("SELECT position FROM reserved_slots ORDER BY 1").fetchall())
    assert slots == tuple(sorted(RESERVED_SLOTS))


def test_arm_summary(data):
    df, con = data
    expected = audit.arm_summary(audit.per_search(df))
    got = sql.query(con, "01_arm_summary").set_index("arm")
    for arm, e in expected.items():
        for k, v in e.items():
            assert got.loc[arm, k] == pytest.approx(v, abs=1e-4), (arm, k)


def test_reserved_slots(data):
    df, con = data
    expected = audit.reserved_slot_share(audit.slot_occupancy(df))
    got = sql.query(con, "02_reserved_slots")
    for r in got.itertuples():
        assert r.vs_neighbours == pytest.approx(expected[r.arm][r.position], abs=1e-4)


def test_search_flags_and_sample(data):
    df, con = data
    flags = audit.price_flags(df, audit.per_search(df))
    got = con.execute("SELECT srch_id, bad_price, price_rho, price_sorted, keep FROM search_flags").df().set_index("srch_id")
    got = got.reindex(flags.index)
    assert (got["bad_price"] == flags["bad_price"]).all()
    assert (got["price_sorted"] == flags["price_sorted"].fillna(False).astype(bool)).all()
    both = got["price_rho"].notna() & flags["price_rho"].notna()
    assert (got["price_rho"].notna() == flags["price_rho"].notna()).all()
    assert np.allclose(got.loc[both, "price_rho"], flags.loc[both, "price_rho"])
    s = sql.query(con, "03_analysis_sample").iloc[0]
    x = position.analysis_sample(df, flags.reset_index().rename(columns={"index": "srch_id"}))
    assert s["kept_algorithm"] + s["kept_random"] == x["srch_id"].nunique()


def test_search_flags_table_matches_per_search(data):
    """The table notebook 01 saves as data/interim/search_flags.parquet."""
    df, con = data
    got = sql.search_flags(con)
    expected = audit.per_search(df).sort_index()
    for c in ("random", "n", "max_position", "clicks", "bookings"):
        assert (got[c].to_numpy() == expected.loc[got.index, c].to_numpy()).all(), c
    assert sql.arm_summary(con) == audit.arm_summary(expected)
