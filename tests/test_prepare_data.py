"""Tests for scripts/prepare_data.py on a small synthetic file with the real schema."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RAW_COLUMNS = [
    "srch_id", "date_time", "site_id", "visitor_location_country_id", "visitor_hist_starrating",
    "visitor_hist_adr_usd", "prop_country_id", "prop_id", "prop_starrating", "prop_review_score",
    "prop_brand_bool", "prop_location_score1", "prop_location_score2", "prop_log_historical_price",
    "position", "price_usd", "promotion_flag", "srch_destination_id", "srch_length_of_stay",
    "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count",
    "srch_saturday_night_bool", "srch_query_affinity_score", "orig_destination_distance", "random_bool",
    *prep.COMP_COLUMNS, "click_bool", "gross_bookings_usd", "booking_bool",
]


def fake_train(n_searches=30, per_search=8, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_searches + 1):
        random_order = s % 3 == 0
        booked = rng.integers(per_search)
        for pos in range(1, per_search + 1):
            row = {c: np.nan for c in RAW_COLUMNS}
            row.update(
                srch_id=s, date_time=f"2013-0{1 + s % 6}-0{1 + pos % 9} 10:00:00", site_id=5,
                visitor_location_country_id=219, prop_country_id=219, prop_id=1000 + s * 10 + pos,
                prop_starrating=3, prop_review_score=4.0, prop_brand_bool=1, prop_location_score1=2.5,
                prop_log_historical_price=4.8, position=pos, price_usd=100.0 + pos, promotion_flag=pos % 2,
                srch_destination_id=7, srch_length_of_stay=2, srch_booking_window=10, srch_adults_count=2,
                srch_children_count=0, srch_room_count=1, srch_saturday_night_bool=0,
                random_bool=int(random_order), click_bool=int(pos - 1 == booked), booking_bool=int(pos - 1 == booked),
            )
            row["gross_bookings_usd"] = 210.0 if row["booking_bool"] else np.nan
            row["comp2_rate"] = 1.0 if pos % 3 == 0 else (-1.0 if pos % 3 == 1 else np.nan)
            row["comp5_rate"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "train.csv"
    fake_train().to_csv(p, index=False)
    return p


def test_prepare_writes_typed_parquet_and_manifest(csv_path, tmp_path):
    out = tmp_path / "interim" / "train.parquet"
    summary = prep.prepare(csv_path, out, expected=None, chunksize=50)  # several chunks
    df = pd.read_parquet(out)
    assert summary["rows"] == len(df) == 240
    assert summary["searches"] == 30
    assert summary["random_search_share"] == pytest.approx(10 / 30, abs=1e-4)
    assert summary["max_bookings_per_search"] == 1
    assert not any(c.startswith("comp1_") for c in df.columns)
    assert df["click_bool"].dtype == bool and df["prop_id"].dtype == np.int32
    assert (out.parent / "prepare_manifest.json").exists()
    assert not out.with_suffix(".parquet.partial").exists()


def test_competitor_counts(csv_path, tmp_path):
    out = tmp_path / "t.parquet"
    prep.prepare(csv_path, out, expected=None)
    df = pd.read_parquet(out).set_index(["srch_id", "position"])
    # position 3: comp2 = +1 (Expedia cheaper), comp5 = 0
    assert df.loc[(1, 3), ["comp_n_known", "comp_n_expedia_cheaper", "comp_n_expedia_dearer"]].tolist() == [2, 1, 0]
    # position 1: comp2 = -1 (Expedia dearer)
    assert df.loc[(1, 1), ["comp_n_known", "comp_n_expedia_cheaper", "comp_n_expedia_dearer"]].tolist() == [2, 0, 1]
    # position 2: only comp5 known
    assert df.loc[(1, 2), "comp_n_known"] == 1


def test_searches_split_across_chunks_are_counted_once(csv_path, tmp_path):
    summary = prep.prepare(csv_path, tmp_path / "t.parquet", expected=None, chunksize=7)
    assert summary["searches"] == 30
    assert summary["max_bookings_per_search"] == 1


def test_mismatch_with_published_shape_refuses_to_finish(csv_path, tmp_path):
    out = tmp_path / "t.parquet"
    with pytest.raises(SystemExit, match="does not match"):
        prep.prepare(csv_path, out)  # default EXPECTED is the full 9.9M-row file
    assert not out.exists()


def test_check_reports_each_problem():
    good = {"rows": 9_917_530, "searches": 399_344, "hotels": 136_886, "date_min": "2012-11-01",
            "date_max": "2013-06-30", "random_search_share": 0.3044}
    assert prep.check(good, prep.EXPECTED) == []
    bad = {**good, "rows": 10, "random_search_share": 0.5}
    problems = prep.check(bad, prep.EXPECTED)
    assert len(problems) == 2 and "rows" in problems[0]


def test_main_explains_missing_file(tmp_path, capsys):
    assert prep.main(["--source", str(tmp_path / "nope.csv")]) == 1
    assert "data/raw/README.md" in capsys.readouterr().out
