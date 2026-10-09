"""Convert the Kaggle ``train.csv`` into a typed parquet file and validate it.

Run from the project root::

    python scripts/prepare_data.py                      # data/raw/train.csv -> data/interim/train.parquet
    python scripts/prepare_data.py --source other.csv --no-expect

What it does, in one streaming pass (the CSV is 2.4 GB, so it is never loaded whole):

* keeps every column except the 24 competitor columns, which are 55-98% missing, and
  replaces them with three counts: competitors with a known price, competitors Expedia
  undercuts, competitors cheaper than Expedia;
* downcasts types (float32, small integers, booleans) and parses ``date_time``;
* checks the published shape of the data (rows, searches, hotels, dates, share of
  randomly ordered searches) and refuses to finish if it does not match;
* writes ``data/interim/prepare_manifest.json`` with the counts it found.

The raw data are Kaggle competition data and are never committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUT = ROOT / "data" / "interim" / "train.parquet"

# Shape of the full training file (competition paper, arXiv 1311.7679, and our profile).
EXPECTED = {
    "rows": 9_917_530,
    "searches": 399_344,
    "hotels": 136_886,
    "date_min": "2012-11-01",
    "date_max": "2013-06-30",
    "random_search_share": (0.29, 0.32),
}

COMP = [f"comp{i}" for i in range(1, 9)]
COMP_COLUMNS = [f"{c}_{s}" for c in COMP for s in ("rate", "inv", "rate_percent_diff")]

DTYPES = {
    "srch_id": "int32",
    "site_id": "int16",
    "visitor_location_country_id": "int16",
    "visitor_hist_starrating": "float32",
    "visitor_hist_adr_usd": "float32",
    "prop_country_id": "int16",
    "prop_id": "int32",
    "prop_starrating": "int8",
    "prop_review_score": "float32",
    "prop_brand_bool": "int8",
    "prop_location_score1": "float32",
    "prop_location_score2": "float32",
    "prop_log_historical_price": "float32",
    "position": "int8",
    "price_usd": "float64",
    "promotion_flag": "int8",
    "srch_destination_id": "int32",
    "srch_length_of_stay": "int16",
    "srch_booking_window": "int16",
    "srch_adults_count": "int8",
    "srch_children_count": "int8",
    "srch_room_count": "int8",
    "srch_saturday_night_bool": "int8",
    "srch_query_affinity_score": "float32",
    "orig_destination_distance": "float32",
    "random_bool": "int8",
    "click_bool": "int8",
    "gross_bookings_usd": "float32",
    "booking_bool": "int8",
}
READ_DTYPES = {**DTYPES, **{c: "float32" for c in COMP_COLUMNS}}


def transform(chunk: pd.DataFrame) -> pd.DataFrame:
    """Type one chunk and replace the competitor columns with three counts."""
    out = chunk.drop(columns=COMP_COLUMNS).copy()
    out["date_time"] = pd.to_datetime(out["date_time"])
    rates = chunk[[f"{c}_rate" for c in COMP]].to_numpy()
    # comp*_rate: +1 Expedia's price is lower, 0 the same, -1 higher, NaN unknown.
    out["comp_n_known"] = np.isfinite(rates).sum(axis=1).astype("int8")
    out["comp_n_expedia_cheaper"] = (rates == 1).sum(axis=1).astype("int8")
    out["comp_n_expedia_dearer"] = (rates == -1).sum(axis=1).astype("int8")
    for col in ("random_bool", "click_bool", "booking_bool", "promotion_flag",
                "prop_brand_bool", "srch_saturday_night_bool"):
        out[col] = out[col].astype(bool)
    return out


def summarize(searches: pd.DataFrame, rows: int, hotels: set, dmin, dmax) -> dict:
    return {
        "rows": int(rows),
        "searches": int(len(searches)),
        "hotels": int(len(hotels)),
        "date_min": str(pd.Timestamp(dmin).date()),
        "date_max": str(pd.Timestamp(dmax).date()),
        "random_search_share": round(float(searches["random"].mean()), 4),
        "searches_with_click": round(float((searches["clicks"] > 0).mean()), 4),
        "searches_with_booking": round(float((searches["bookings"] > 0).mean()), 4),
        "max_bookings_per_search": int(searches["bookings"].max()),
    }


def check(summary: dict, expected: dict) -> list[str]:
    problems = []
    for key in ("rows", "searches", "hotels", "date_min", "date_max"):
        if summary[key] != expected[key]:
            problems.append(f"{key}: expected {expected[key]}, found {summary[key]}")
    lo, hi = expected["random_search_share"]
    if not lo <= summary["random_search_share"] <= hi:
        problems.append(f"random_search_share {summary['random_search_share']} outside [{lo}, {hi}]")
    return problems


def prepare(source, out: Path, expected: dict | None = EXPECTED, chunksize: int = 1_000_000) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.partial")
    writer = None
    rows, hotels, parts = 0, set(), []
    dmin = dmax = None
    try:
        for chunk in pd.read_csv(source, dtype=READ_DTYPES, chunksize=chunksize):
            df = transform(chunk)
            rows += len(df)
            hotels.update(df["prop_id"].unique().tolist())
            lo, hi = df["date_time"].min(), df["date_time"].max()
            dmin = lo if dmin is None else min(dmin, lo)
            dmax = hi if dmax is None else max(dmax, hi)
            parts.append(df.groupby("srch_id").agg(
                random=("random_bool", "max"), clicks=("click_bool", "sum"),
                bookings=("booking_bool", "sum")))
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            print(f"  {rows:>10,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()
    searches = pd.concat(parts).groupby(level=0).agg(
        random=("random", "max"), clicks=("clicks", "sum"), bookings=("bookings", "sum"))
    summary = summarize(searches, rows, hotels, dmin, dmax)
    if expected is not None:
        problems = check(summary, expected)
        if problems:
            raise SystemExit("train.csv does not match the published data:\n  " + "\n  ".join(problems)
                             + f"\nPartial output left at {tmp}")
    tmp.replace(out)
    manifest = {"source": str(source), "output": str(out), **summary,
                "dropped_columns": COMP_COLUMNS,
                "added_columns": ["comp_n_known", "comp_n_expedia_cheaper", "comp_n_expedia_dearer"]}
    (out.parent / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-expect", action="store_true", help="skip the checks against the full published file")
    args = ap.parse_args(argv)
    source = args.source
    if source != "-" and not Path(source).exists():
        print(f"Not found: {source}\nSee data/raw/README.md for how to download and extract train.csv.")
        return 1
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is required: pip install pyarrow")
        return 1
    summary = prepare(sys.stdin if source == "-" else source, Path(args.out),
                      expected=None if args.no_expect else EXPECTED)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
