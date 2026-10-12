"""Hotel Search Ranking: interactive companion to the analysis.

Reads only the committed aggregate records in reports/metrics (no Kaggle data), so it runs
anywhere the repository is cloned: `streamlit run app/streamlit_app.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import streamlit as st  # noqa: E402

from hotelrank import demo  # noqa: E402
from hotelrank.records import read_metrics  # noqa: E402

REPO = "https://github.com/janeruxi1/Hotel-Search-Ranking"
MODES = ["The answer", "Position effects", "What a top slot is worth", "Rankings compared", "Size the test"]

st.set_page_config(page_title="Hotel Search Ranking", layout="wide")


@st.cache_data
def records() -> dict:
    return {n: read_metrics(n) for n in ("01_audit", "02_position_effects", "03_choice_model",
                                         "04_policy_evaluation", "05_ab_test_design")}


R = records()
P2, P3, P4, P5 = R["02_position_effects"], R["03_choice_model"], R["04_policy_evaluation"], R["05_ab_test_design"]

st.sidebar.title("Hotel Search Ranking")
mode = st.sidebar.radio("Explore", MODES)
st.sidebar.markdown(
    f"Expedia's hotel-search logs (ICDM 2013): {R['01_audit']['shape']['impressions']:,} impressions, "
    f"{R['01_audit']['shape']['searches']:,} searches, 30% shown in random order.\n\n"
    f"Every number here is read from the project's committed results. [Code and notebooks]({REPO}) · "
    f"[Decision memo]({REPO}/blob/main/reports/decision_memo.md)")

if mode == "The answer":
    st.title("Which hotel ranking should we A/B test?")
    st.caption("The ranking team proposed putting cheaper hotels first. Analysis by Xi Ru.")
    pol, dec = P4["policies"], P4["decision"]
    f = P2["naive_over_random_ratio"]["click_bool"]["pos1_over_pos30"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Naive read overstates position effects", f"{f['factor']:.1f}×")
    c1.caption(f"95% CI {f['lo']:.1f}–{f['hi']:.1f}, position 1 vs 30, clicks")
    c2.metric("Cheapest first vs reviews first, booking value",
              f"{dec['cheapest_first_vs_reviews_first']['value_rel_change']:+.0%}")
    c2.caption(f"and {100 * dec['cheapest_first_vs_reviews_first']['bookings_diff']:+.2f} bookings per 100 searches")
    c3.metric("Clicked searches per arm for the test", f"{P5['n_per_arm']:,}")
    c3.caption("set by the 2% booking-value guardrail")
    st.markdown(
        "- **Don't test cheapest first.** Offline it loses to a plain reviews-first sort on bookings and on "
        "booking value.\n"
        "- **Test two candidates against the current ranking:** reviews first (the pick of a rule fixed before the "
        "analysis) and a blend of two model rankings (better on both metrics offline, found after looking, so "
        "labelled exploratory).\n"
        f"- **Why a test at all:** the offline gains assume a better top of the page makes a booking more likely. "
        f"The direct check of that is inconclusive (slope {P4['evaluator']['order_luck_slope']['slope']:.2f}, "
        f"95% CI {P4['evaluator']['order_luck_slope']['ci'][0]:.2f} to {P4['evaluator']['order_luck_slope']['ci'][1]:.2f}).")
    st.info("Use the sidebar to explore each step, or try the test-sizing calculator.")

elif mode == "Position effects":
    st.title("Position effects: naive versus randomized")
    outcome = st.radio("Outcome", ["Clicks", "Bookings"], horizontal=True)
    key = "click_bool" if outcome == "Clicks" else "booking_bool"
    c = demo.position_curves(P2, key)
    colours = alt.Scale(domain=["Expedia's ranking (naive)", "Random order"], range=["#d95f02", "#1f77b4"])
    base = alt.Chart(c).encode(x=alt.X("position:Q", title="position"),
                               color=alt.Color("series:N", title=None, scale=colours,
                                               legend=alt.Legend(orient="top", labelLimit=300)))
    band = base.mark_area(opacity=0.2).encode(y=alt.Y("lo:Q", scale=alt.Scale(type="log"),
                                                      title="rate ÷ rate at position 1 (log scale)"), y2="hi:Q",
                                              color=alt.Color("series:N", scale=colours, legend=None))
    line = base.mark_line(point=True).encode(y="relative to position 1:Q",
                                              tooltip=["series", "position", alt.Tooltip("relative to position 1:Q", format=".3f")])
    st.altair_chart((band + line).properties(height=380), width="stretch")
    raw = P2["raw_ratios"]
    f = P2["naive_over_random_ratio"][key]["pos1_over_pos30"]
    st.markdown(
        f"Position 1 vs 30: **{raw[f'algorithm/{key}']['pos1_over_pos30']['ratio']:.1f}×** in Expedia's ranked "
        f"traffic, **{raw[f'random/{key}']['pos1_over_pos30']['ratio']:.1f}×** when hotels were shown in random "
        f"order. The naive reading overstates the effect **{f['factor']:.2f}×** (95% CI {f['lo']:.2f}–{f['hi']:.2f}), "
        "because the ranking already puts the best hotels on top. Searches that fill slots 1–30 only; 95% bands "
        "from a bootstrap over searches.")
    if key == "booking_bool":
        st.caption("The click comparison is the cleaner one: the release kept far more booking searches in the "
                   "ranked arm, so the booking curves come from differently selected samples.")

elif mode == "What a top slot is worth":
    st.title("What is a better position worth, in dollars?")
    st.caption("From a conditional logit on the randomly ordered searches; price sensitivity is measured "
               "within hotel (the same hotel cheaper or dearer than usual).")
    kind = st.radio("Choice", ["click", "booking"], horizontal=True)
    fit = P3["fits"][kind]
    positions = demo.available_positions(fit)
    c1, c2, c3 = st.columns(3)
    frm = c1.selectbox("Hotel now at position", positions, index=positions.index(10))
    to = c2.selectbox("Moved to position", positions, index=0)
    price = c3.number_input("Nightly price, USD", min_value=20.0, max_value=2000.0,
                            value=float(round(P3["median_price_usd"])), step=10.0)
    if frm == to:
        st.write("Pick two different positions.")
    else:
        v = demo.position_value(fit, frm, to, price)
        word = "cut" if v["usd"] >= 0 else "rise"
        lo, hi = sorted((abs(v["usd_lo"]), abs(v["usd_hi"])))
        st.metric(f"Worth the same as a price {word} of", f"${abs(v['usd']):,.0f}")
        st.caption(f"{abs(v['pct']):.0%} of the price · 95% CI \\${lo:,.0f}–\\${hi:,.0f}")
    st.caption("Price was never randomized. If hotels raise prices when demand is high, price sensitivity is "
               "understated and these dollar values are upper bounds. Positions 5, 11, 17 and 23 were reserved "
               "slots and have no estimate.")

elif mode == "Rankings compared":
    st.title("Four rankings, scored offline")
    st.caption(f"{P4['sample']['heldout_searches']:,} held-out random-order searches; booking model fitted on "
               f"{P4['sample']['train_searches']:,} others. Intervals combine a bootstrap over searches with draws "
               "of the model's coefficients.")
    st.dataframe(demo.policy_table(P4).style.format({"bookings per 100 searches": "{:.2f}",
                                                     "booking value per search (USD)": "${:.2f}"}),
                 hide_index=True, width="stretch")
    fr = demo.frontier(P4)
    lam = st.select_slider("Blend weight on booking value (0 = most likely booked, 1 = highest expected value)",
                           options=[round(x, 1) for x in fr["lambda"]], value=0.3)
    pts = demo.policy_table(P4)
    base = alt.Chart(fr).mark_line(point=alt.OverlayMarkDef(color="grey"), color="grey").encode(
        x=alt.X("booking value per search (USD):Q", scale=alt.Scale(zero=False)),
        y=alt.Y("bookings per 100 searches:Q", scale=alt.Scale(zero=False)), tooltip=["lambda"])
    pol = alt.Chart(pts).mark_point(size=90, filled=True).encode(
        x="booking value per search (USD):Q", y="bookings per 100 searches:Q",
        color=alt.Color("ranking:N", title=None, legend=alt.Legend(orient="bottom", columns=2, labelLimit=300)),
        tooltip=["ranking"])
    sel = alt.Chart(fr[fr["lambda"].round(1) == lam]).mark_point(size=250, shape="diamond", color="black").encode(
        x="booking value per search (USD):Q", y="bookings per 100 searches:Q")
    st.altair_chart((base + pol + sel).properties(height=400), width="stretch")
    row = fr[fr["lambda"].round(1) == lam].iloc[0]
    b1 = P4["policies"]["B1_reviews_first"]
    st.markdown(f"Blend at {lam}: **{row['bookings per 100 searches']:.2f}** bookings per 100 searches and "
                f"**\\${row['booking value per search (USD)']:.2f}** per search, against reviews first's "
                f"{100 * b1['bookings_per_search']:.2f} and \\${b1['value_per_search_usd']:.2f}. The grey line "
                "traces the blends; the diamond is the one selected.")
    st.caption("The blends are exploratory: the pre-declared rule compared only the four rankings in the table.")

else:
    st.title("Size the A/B test")
    st.caption("Three arms (current ranking, reviews first, model blend), equal thirds, sized on the gap between "
               "the two candidates. Turn the knobs a stakeholder would turn.")
    slope = P4["evaluator"]["order_luck_slope"]["slope"]
    c1, c2 = st.columns(2)
    margin = c1.slider("Value guardrail: largest drop we must rule out", 0.01, 0.10, 0.02, 0.005, format="%.3f")
    options = sorted({round(x / 20, 2) for x in range(2, 21)} | {slope})
    discount = c1.select_slider("Share of the offline lift we expect online", options=options, value=slope,
                                format_func=lambda v: f"{v:.2f}",
                                help=f"Default {slope:.2f}: the order-luck slope from phase 4, as in the memo.")
    comparisons = c2.selectbox("Comparisons sharing α = 0.05 (Bonferroni)", [3, 2, 1], index=0)
    daily = c2.number_input("Clicked searches entering the test per day", 5_000, 2_000_000, 100_000, 5_000)
    cap = c2.checkbox("Cap booking value at the 99th percentile", value=True)
    s = demo.size_test(P5, P4, margin=margin, discount=discount, comparisons=comparisons, daily_searches=daily,
                       cap_value=cap)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Per arm", f"{s['n_per_arm']:,}")
    m2.metric("Total", f"{s['n_total']:,}")
    m3.metric("Days to run", f"{s['days']}")
    m4.metric("Detectable booking lift", f"{s['mde']:.1%}")
    st.markdown(
        f"The booking comparison ({s['lift']:+.1%} on a {P5['inputs']['baseline_booking_rate']:.1%} booking rate, "
        f"α = {s['alpha']:.4f}) needs **{s['n_booking']:,}** per arm; the value guardrail needs "
        f"**{s['n_guardrail']:,}**. The **{s['binding']}** sets the size. Runs at least 14 days so every weekday is "
        "seen twice.")
    st.caption("Sizes are per clicked search and treat searches as independent. In production the analysis "
               "population is every search with results and the unit is the visitor, which both raise the "
               "requirement.")
