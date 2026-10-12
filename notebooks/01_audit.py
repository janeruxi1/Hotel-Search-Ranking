# %% [markdown]
# # 01 · Data audit
#
# Before estimating anything: does the data match its published description, and where are
# the traps? This notebook writes `reports/metrics/01_audit.json`, two figures, and the
# per-search flags (`data/interim/search_flags.parquet`) that define the analysis sample for
# every later notebook.
#
# Without the real data (CI, or anyone who has not downloaded it) the notebook runs on the
# synthetic table from `hotelrank.synthetic` and writes nothing to `reports/`.

# %%
import sys
from pathlib import Path

def _project_root() -> Path:
    """The folder holding src/hotelrank, wherever the script or notebook was started."""
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "src" / "hotelrank" / "__init__.py").exists():
            return p
    for marker in here.glob("*/src/hotelrank/__init__.py"):
        return marker.parents[2]
    raise FileNotFoundError("run from inside the project folder")


sys.path.insert(0, str(_project_root() / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from hotelrank import audit
from hotelrank.data import FIGURES_DIR, PRICE_CAP, PRICE_SORT_RHO, RESERVED_SLOTS, ROOT as PROJECT_ROOT, load
from hotelrank.records import show_verdict, write_metrics

COLUMNS = ["srch_id", "date_time", "site_id", "prop_id", "srch_destination_id", "position", "random_bool",
           "click_bool", "booking_bool", "price_usd", "srch_length_of_stay", "prop_starrating",
           "prop_review_score", "prop_location_score1", "prop_location_score2", "promotion_flag",
           "visitor_hist_starrating", "visitor_hist_adr_usd", "srch_query_affinity_score",
           "orig_destination_distance", "gross_bookings_usd"]
df, SOURCE = load(COLUMNS)
print(f"source: {SOURCE}, {len(df):,} impressions")

# %% [markdown]
# ## 1. Shape and the two arms
#
# `random_bool = 1` marks searches whose results Expedia showed in random order.

# %%
searches = audit.per_search(df)
shape = {
    "impressions": int(len(df)),
    "searches": int(len(searches)),
    "hotels": int(df["prop_id"].nunique()),
    "destinations": int(df["srch_destination_id"].nunique()),
    "sites": int(df["site_id"].nunique()),
    "date_min": str(df["date_time"].min().date()),
    "date_max": str(df["date_time"].max().date()),
}
arms = audit.arm_summary(searches)
print(shape)
pd.DataFrame(arms)

# %% [markdown]
# Every search has at least one click: the release contains only searches with a click, so
# every result in this project describes engaged searches. Each search has at most one booking.
#
# The share of searches ending in a booking differs enormously between the arms (about 94% against
# 13%). A gap that size is very unlikely to come from the ordering alone and most likely reflects
# how the release was sampled; the documentation does not say. Either way the arms are not
# comparable on booking rates, so **booking rates are never compared between arms**; every
# comparison is made within an arm.

# %%
selection = {
    "searches_with_click": round(float((searches["clicks"] > 0).mean()), 4),
    "max_bookings_per_search": int(searches["bookings"].max()),
    "bookings_without_click": int((df["booking_bool"] & ~df["click_bool"]).sum()),
}
print(selection)

# %% [markdown]
# ## 2. Positions: four slots are (almost) never filled
#
# Slots 5, 11, 17 and 23 hold a few percent of the impressions of their neighbours in both
# arms, and most searches skip exactly four slots. Those slots were presumably used for
# something not released (an advert or a different unit). Positions are kept as logged; the
# position-effect notebook works with displayed slots and reports the reserved ones separately.

# %%
occ = audit.slot_occupancy(df)
reserved = audit.reserved_slot_share(occ)
gaps = audit.gaps_per_search(searches)
print("reserved slots, impressions relative to neighbours:", reserved)
print("skipped slots per search:", dict(list(gaps.items())[:6]))

fig, ax = plt.subplots(figsize=(9, 3.2))
for arm, colour in (("algorithm", "#1f77b4"), ("random", "#ff7f0e")):
    ax.plot(occ.index, occ[arm] / occ[arm].iloc[0], marker="o", ms=3, lw=1.2, label=f"{arm} order", color=colour)
for s in RESERVED_SLOTS:
    ax.axvline(s, color="grey", lw=0.6, ls=":")
ax.set(xlabel="position", ylabel="impressions, relative to position 1",
       title="Slots 5, 11, 17 and 23 are almost never filled")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# ## 3. Prices
#
# Prices are mostly per night (the median rises only modestly with length of stay), with a
# long tail of errors. Rule: a search containing any price outside (0, $5,000] is dropped
# from the analysis sample. Later notebooks use price relative to the other hotels in the
# same search, which removes currency and per-night/per-stay differences.

# %%
prices = audit.price_summary(df)
print(prices)

# %% [markdown]
# ## 4. Is the random arm actually random?
#
# Under true random ordering, a hotel's position says nothing about its attributes. The check
# demeans each attribute within its search (so differences between markets cancel) and
# correlates it with position. The algorithm arm should show strong correlations; the random
# arm should show none.
#
# Two problems turn up in the real data:
#
# 1. **Some searches are sorted by price.** Their prices rise with position almost perfectly
#    (about 5% of random-arm and 4% of algorithm-arm searches), most likely because the user
#    chose "sort by price". They are flagged and dropped from both arms.
# 2. **A residual tilt remains in the random arm**: hotels with more stars sit slightly higher.
#    It is diffuse rather than a few re-sorted searches (removing searches sorted by star rating
#    barely changes it) and is about 40% of the algorithm arm's tilt, but it is not zero. The
#    position-effect notebook therefore adjusts for hotel attributes and reports the estimate
#    with and without that adjustment.

# %%
flags = audit.price_flags(df, searches)
BALANCE_COLS = ["prop_starrating", "prop_review_score", "prop_location_score1", "promotion_flag", "price_usd"]
keep_price_ok = ~flags["bad_price"]
keep_clean = keep_price_ok & ~flags["price_sorted"].fillna(False)

price_sorted_share = {
    name: round(float(flags.loc[(flags["random"] == flag) & flags["price_rho"].notna(), "price_sorted"].mean()), 4)
    for flag, name in audit.ARM_NAMES.items()
}
balance_all = audit.balance(df, BALANCE_COLS, keep=keep_price_ok)
balance_clean = audit.balance(df, BALANCE_COLS, keep=keep_clean)
# Sensitivity: also drop searches whose star ratings fall almost perfectly with position.
flags["star_rho"] = audit.within_search_rank_corr(df, "prop_starrating").reindex(flags.index)
star_sorted = flags["star_rho"] <= -0.9
balance_no_star_sorted = audit.balance(df, BALANCE_COLS, keep=keep_clean & ~star_sorted.fillna(False))
star_sorted_share = {name: round(float(star_sorted[(flags["random"] == flag) & flags["star_rho"].notna()].mean()), 4)
                     for flag, name in audit.ARM_NAMES.items()}
print("share of searches sorted by price:", price_sorted_share)
print("share sorted by stars (rho <= -0.9):", star_sorted_share,
      "random-arm star tilt without them:", balance_no_star_sorted["random"]["prop_starrating"])
pd.DataFrame({"algorithm": balance_all["algorithm"], "random (all)": balance_all["random"],
              "random (price-sorted removed)": balance_clean["random"]})

# %%
fig2, ax2 = plt.subplots(figsize=(7, 3.4))
series = {
    "algorithm order": audit.balance_by_bucket(df, "prop_starrating", False, keep=keep_clean),
    "random order": audit.balance_by_bucket(df, "prop_starrating", True, keep=keep_clean),
}
labels = [f"{int(i.left) + 1}–{int(i.right)}" for i in series["algorithm order"].index]
width = 0.38
for i, (name, s) in enumerate(series.items()):
    ax2.bar([j + (i - 0.5) * width for j in range(len(s))], s.values, width, label=name,
            color=["#1f77b4", "#ff7f0e"][i])
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(range(len(labels)), labels)
ax2.set(xlabel="position", ylabel="stars minus search average",
        title="Random order removes most, not all, of the star-rating tilt")
ax2.legend(frameon=False)
fig2.tight_layout()

# %% [markdown]
# ## 5. Analysis sample and missing data

# %%
sample = {
    "rule": f"drop searches with any price outside (0, {PRICE_CAP:,.0f}]; drop searches sorted by price "
            f"(within-search rank correlation of price with position >= {PRICE_SORT_RHO})",
    "searches_bad_price": int(flags["bad_price"].sum()),
    "searches_price_sorted": int((flags["price_sorted"].fillna(False) & ~flags["bad_price"]).sum()),
    "searches_kept": {name: int((keep_clean & (flags["random"] == flag)).sum())
                      for flag, name in audit.ARM_NAMES.items()},
    "impressions_kept": int(keep_clean.reindex(df["srch_id"]).sum()),
}
missing = audit.missing_share(df)
print(sample)
print(missing)

# %%
record = {
    "shape": shape, "arms": arms, "selection": selection,
    "reserved_slots_vs_neighbours": reserved, "skipped_slots_per_search": gaps,
    "prices": prices, "price_sorted_share": price_sorted_share,
    "balance_within_search_corr": {"price_ok": balance_all, "clean": balance_clean,
                                   "clean_without_star_sorted": balance_no_star_sorted},
    "star_sorted_share": star_sorted_share,
    "stars_minus_search_mean_by_position": {k: {str(i): round(float(v), 4) for i, v in s.items()}
                                            for k, s in series.items()},
    "analysis_sample": sample, "missing_share": missing,
}
path = write_metrics("01_audit", record, SOURCE)
print("wrote", path.relative_to(PROJECT_ROOT) if SOURCE == "real" else path.name)
if SOURCE == "real":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "01_reserved_slots.png", dpi=150)
    fig2.savefig(FIGURES_DIR / "01_random_arm_balance.png", dpi=150)
    flags.reset_index().rename(columns={"index": "srch_id"}).to_parquet(
        PROJECT_ROOT / "data" / "interim" / "search_flags.parquet", index=False)

# %% [markdown]
# ## Verdict
#
# What the audit establishes, and the rules the later phases inherit. The verdict is rendered from `record`, the same dictionary written to
# `reports/metrics/01_audit.json`, so the summary cannot drift from the analysis above.

# %%
a, r = record["arms"]["algorithm"], record["arms"]["random"]
corr = record["balance_within_search_corr"]["clean"]
s = record["analysis_sample"]
fill = [v for arm in record["reserved_slots_vs_neighbours"].values() for v in arm.values()]
verdict = [
    f"{'The data matches the published release' if SOURCE == 'real' else 'Synthetic stand-in data'}: {shape['impressions']:,} impressions in {shape['searches']:,} searches, "
    f"{shape['date_min']} to {shape['date_max']}. Every search has a click and at most one booking, so this is a "
    "selected sample of engaged searches: all rates are conditional on at least one click.",
    f"The two arms are not comparable on booking: {a['searches_with_booking']:.1%} of algorithm-order searches book "
    f"against {r['searches_with_booking']:.1%} of random-order searches. Every comparison is made within an arm.",
    f"Positions {', '.join(map(str, RESERVED_SLOTS))} are filled at only {min(fill):.1%}-{max(fill):.1%} of their "
    "neighbours' rate. They are reserved slots, so they are left out of position curves and choice sets.",
    f"{record['price_sorted_share']['algorithm']:.1%} of algorithm-order and {record['price_sorted_share']['random']:.1%} "
    "of random-order searches are sorted by price: the user re-sorted, so position no longer reflects the ranking "
    "being studied. They are dropped.",
    f"The random arm is close to random, not perfectly random: within a search, corr(position, stars) is "
    f"{corr['random']['prop_starrating']:+.3f} against {corr['algorithm']['prop_starrating']:+.3f} in algorithm order. "
    "Phase 2 therefore reports attribute-adjusted curves next to the raw ones.",
    f"Analysis sample: {s['searches_kept']['algorithm']:,} algorithm-order and {s['searches_kept']['random']:,} "
    f"random-order searches ({s['impressions_kept']:,} impressions), after dropping {s['searches_bad_price']:,} searches "
    f"with impossible prices and {s['searches_price_sorted']:,} price-sorted searches.",
]
show_verdict("Phase 1 verdict: data audit", verdict,
             "Phase 2 uses the random arm to measure how much position itself changes clicks and bookings.")
