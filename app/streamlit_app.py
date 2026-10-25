"""Hotel Search Ranking: interactive companion to the analysis.

Reads only the committed aggregate records in reports/metrics (no Kaggle data), so it runs
anywhere the repository is cloned: `streamlit run app/streamlit_app.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from hotelrank import demo  # noqa: E402
from hotelrank.records import read_metrics  # noqa: E402

REPO = "https://github.com/janeruxi1/Hotel-Search-Ranking"
TABS = ["📋 The answer", "📍 Position effects", "💲 What a top slot is worth", "🏆 Rankings compared",
        "🧪 Size the test"]
ROLE_COLOR = {"Cheapest first (the proposal)": "#e34948", "Reviews first": "#1baf7a",
              "Model: most likely booked": "#2a78d6", "Model: highest expected value": "#6250d6",
              "Random order, as shown": "#8a8985"}
NAIVE, RANDOM = "#eb6834", "#2a78d6"

st.set_page_config(page_title="Hotel Search Ranking", page_icon="🏨", layout="wide")


@st.cache_data
def records() -> dict:
    return {n: read_metrics(n) for n in ("01_audit", "02_position_effects", "03_choice_model",
                                         "04_policy_evaluation", "05_ab_test_design")}


R = records()
A1, P2, P3, P4, P5 = (R[k] for k in ("01_audit", "02_position_effects", "03_choice_model", "04_policy_evaluation",
                                     "05_ab_test_design"))
naive = P2["naive_over_random_ratio"]["click_bool"]["pos1_over_pos30"]
dec = P4["decision"]
slot = demo.position_value(P3["fits"]["click"], 10, 1, P3["median_price_usd"])
slope = P4["evaluator"]["order_luck_slope"]

# ------------------------------------------------------------------------------------------ header
st.title("🏨 Which hotel ranking should we A/B test?")
st.markdown(
    "The ranking team wants to put **cheaper hotels first**. Before spending a test slot on it: how much does a "
    "top position really matter, what makes people book, and which ranking deserves the test? Built on Expedia's "
    f"hotel-search logs (ICDM 2013): **{A1['shape']['impressions']:,} impressions** from "
    f"{A1['shape']['searches']:,} searches, **30% of them shown in random order**, which is what makes position "
    f"effects measurable. [Code and notebooks]({REPO}) · [Decision memo]({REPO}/blob/main/reports/decision_memo.md)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Naive reading overstates position effects", f"{naive['factor']:.1f}×", border=True,
          help=f"Position 1 vs 30, clicks; 95% CI {naive['lo']:.1f}–{naive['hi']:.1f}. The ranking already puts "
               "the best hotels on top, so ranked traffic exaggerates what the slot itself does.")
k2.metric("Moving a hotel from 10th to 1st", f"≈ ${slot['usd']:,.0f} off", border=True,
          help=f"Worth the same as cutting a ${P3['median_price_usd']:.0f} nightly price by this much, for clicks.")
k3.metric("Cheapest first vs reviews first", f"{dec['cheapest_first_vs_reviews_first']['value_rel_change']:+.0%}",
          border=True, help="Booking value per search, offline on held-out random-order searches.")
k4.metric("Searches per arm for the test", f"{P5['n_per_arm']:,}", border=True,
          help="Clicked searches per arm, set by the 2% booking-value guardrail.")

with st.expander("How to use this demo", icon="🧭"):
    st.markdown(
        "1. **The answer**: the recommendation and the five rankings side by side.\n"
        "2. **Position effects**: how much a slot matters once the ranking's own choices are removed.\n"
        "3. **What a top slot is worth**: pick two positions and see the move priced in dollars.\n"
        "4. **Rankings compared**: the offline scores, and blends of the two model rankings.\n"
        "5. **Size the test**: turn the knobs a stakeholder would turn and see how long the A/B test runs.")

tab_ans, tab_pos, tab_slot, tab_rank, tab_size = st.tabs(TABS)

# ------------------------------------------------------------------------------------------ answer
with tab_ans:
    st.subheader("Don't test cheapest first; test reviews first and a model blend")
    st.markdown(
        "- **Don't test cheapest first.** Offline it loses to a plain reviews-first sort on bookings "
        f"({100 * dec['cheapest_first_vs_reviews_first']['bookings_diff']:+.2f} per 100 searches) and on booking "
        f"value ({dec['cheapest_first_vs_reviews_first']['value_rel_change']:+.0%}).\n"
        "- **Test two candidates against the current ranking:** reviews first (the pick of a rule written before the "
        "analysis) and a blend of the two model rankings (better on both metrics offline, but found after looking, "
        "so labelled exploratory).\n"
        "- **Why a test and not a launch:** the offline gains assume that a better top of the page makes a booking "
        f"more likely. The direct check of that is inconclusive (slope {slope['slope']:.2f}, 95% CI "
        f"{slope['ci'][0]:.2f} to {slope['ci'][1]:.2f}), so the test is sized for a discounted gain.")
    t = demo.policy_table(P4)
    left, right = st.columns(2, gap="large")
    for col, field, title, fmt in [
            (left, "bookings per 100 searches", "Expected bookings per 100 searches", ".2f"),
            (right, "booking value per search (USD)", "Expected booking value per search (USD)", "$.2f")]:
        lo, hi = float(t[field].min()), float(t[field].max())
        bars = alt.Chart(t).mark_bar().encode(
            y=alt.Y("ranking:N", sort=list(ROLE_COLOR), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X(f"{field}:Q", title=None, scale=alt.Scale(domain=[0.9 * lo, 1.06 * hi], zero=False, clamp=True)),
            color=alt.Color("ranking:N", scale=alt.Scale(domain=list(ROLE_COLOR), range=list(ROLE_COLOR.values())),
                            legend=None),
            tooltip=["ranking", alt.Tooltip(f"{field}:Q", format=fmt)])
        text = bars.mark_text(align="left", dx=4).encode(text=alt.Text(f"{field}:Q", format=fmt))
        col.altair_chart((bars + text).properties(title=title, height=230), width="stretch")
    st.caption("Red: the proposal. Green: the pick of the rule written before the analysis. Blue and purple: the two "
               "model rankings. Grey: the random order as shown.")
    st.caption(f"{P4['sample']['heldout_searches']:,} held-out random-order searches, scored with a booking model "
               "fitted on the others. The axes do not start at zero, to show the gaps.")

# ------------------------------------------------------------------------------------------ position effects
with tab_pos:
    st.subheader("How much does the slot itself matter?")
    st.markdown("In Expedia's normal ranking the best hotels are placed on top, so the top slots look stronger than "
                "they are. In the **randomly ordered** searches, position is the only thing that differs, so the gap "
                "there is the slot's own effect.")
    outcome = st.radio("Outcome", ["Clicks", "Bookings"], horizontal=True, key="pos_outcome")
    key = "click_bool" if outcome == "Clicks" else "booking_bool"
    c = demo.position_curves(P2, key)
    colours = alt.Scale(domain=["Expedia's ranking (naive)", "Random order"], range=[NAIVE, RANDOM])
    base = alt.Chart(c).encode(x=alt.X("position:Q", title="position on the page", scale=alt.Scale(domain=[1, 30])))
    band = base.mark_area(opacity=0.2).encode(
        y=alt.Y("lo:Q", scale=alt.Scale(type="log"), title=f"{outcome.lower()} rate ÷ rate at position 1 (log scale)"),
        y2="hi:Q", color=alt.Color("series:N", scale=colours, legend=None))
    line = base.mark_line(point=True).encode(
        y="relative to position 1:Q",
        color=alt.Color("series:N", title=None, scale=colours, legend=alt.Legend(orient="top", labelLimit=300)),
        tooltip=["series", "position", alt.Tooltip("relative to position 1:Q", format=".3f")])
    st.altair_chart(alt.layer(band, line).resolve_legend(color="independent").properties(height=380), width="stretch")
    raw = P2["raw_ratios"]
    f = P2["naive_over_random_ratio"][key]["pos1_over_pos30"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Position 1 vs 30, Expedia's ranking", f"{raw[f'algorithm/{key}']['pos1_over_pos30']['ratio']:.1f}×",
              border=True)
    m2.metric("Position 1 vs 30, random order", f"{raw[f'random/{key}']['pos1_over_pos30']['ratio']:.1f}×", border=True)
    m3.metric("Overstatement from reading ranked traffic", f"{f['factor']:.2f}×", border=True,
              help=f"95% CI {f['lo']:.2f}–{f['hi']:.2f}, bootstrap over searches.")
    st.caption("Both curves start at 1 at position 1; the further a curve falls, the more the slot seems to matter. "
               "Searches that fill slots 1–30 only; shaded bands are 95% bootstrap intervals.")
    if key == "booking_bool":
        st.info("The click comparison is the cleaner one: the release kept far more booking searches in the ranked "
                "arm, so the two booking curves come from differently selected samples.", icon="ℹ️")

# ------------------------------------------------------------------------------------------ slot value
with tab_slot:
    st.subheader("What is a better position worth, in dollars?")
    st.markdown("A choice model fitted on the randomly ordered searches puts position and price on the same scale, "
                "so moving a hotel up can be priced as the discount that would have the same effect.")
    kind = st.radio("Measured on", ["Clicks", "Bookings"], horizontal=True, key="slot_kind")
    fit = P3["fits"]["click" if kind == "Clicks" else "booking"]
    positions = demo.available_positions(fit)
    c1, c2, c3 = st.columns(3)
    frm = c1.selectbox("Hotel now at position", positions, index=positions.index(10), key="slot_from")
    to = c2.selectbox("Moved to position", positions, index=0, key="slot_to")
    price = c3.number_input("Nightly price, USD", min_value=20.0, max_value=2000.0,
                            value=float(round(P3["median_price_usd"])), step=10.0, key="slot_price")
    if frm == to:
        st.info("Pick two different positions.", icon="ℹ️")
    else:
        v = demo.position_value(fit, frm, to, price)
        word = "cut" if v["usd"] >= 0 else "rise"
        lo, hi = sorted((abs(v["usd_lo"]), abs(v["usd_hi"])))
        st.metric(f"Worth the same as a price {word} of", f"${abs(v['usd']):,.0f}", border=True,
                  help=f"{abs(v['pct']):.0%} of the price; 95% CI ${lo:,.0f}–${hi:,.0f}.")
    vals = pd.DataFrame([{"position": p, "usd": demo.position_value(fit, frm, p, price)["usd"]}
                         for p in positions if p != frm])
    vals["direction"] = vals["usd"].map(lambda u: "move up: worth a price cut" if u >= 0 else "move down: needs a price rise")
    bars = alt.Chart(vals).mark_bar().encode(
        x=alt.X("position:O", title=f"moved from position {frm} to …"),
        y=alt.Y("usd:Q", title="equivalent price change (USD)"),
        color=alt.Color("direction:N", title=None,
                        scale=alt.Scale(domain=["move up: worth a price cut", "move down: needs a price rise"],
                                        range=[RANDOM, NAIVE]),
                        legend=alt.Legend(orient="top", labelLimit=300)),
        tooltip=["position", alt.Tooltip("usd:Q", format="$,.0f")])
    st.altair_chart(bars.properties(height=300, title=f"Every move from position {frm}, priced at ${price:,.0f} a night"),
                    width="stretch")
    st.caption("Price was never randomized. If hotels raise prices when demand is high, price sensitivity is "
               "understated and these dollar values are upper bounds. Positions 5, 11, 17 and 23 were reserved slots "
               "and have no estimate.")

# ------------------------------------------------------------------------------------------ rankings
with tab_rank:
    st.subheader("Five rankings, scored on searches the model never saw")
    st.caption(f"{P4['sample']['heldout_searches']:,} held-out random-order searches; booking model fitted on "
               f"{P4['sample']['train_searches']:,} others. Intervals combine a bootstrap over searches with draws of "
               "the model's coefficients.")
    pt = demo.policy_table(P4)
    pt["value 95% interval"] = [f"${v['value_ci'][0]:.2f}–${v['value_ci'][1]:.2f}" for v in P4["policies"].values()]
    st.dataframe(pt.style.format({"bookings per 100 searches": "{:.2f}", "booking value per search (USD)": "${:.2f}"}),
                 hide_index=True, width="stretch")
    st.markdown("**Blending the two model rankings (exploratory).** The grey curve trades bookings for booking value; "
                "move the slider to pick a blend.")
    fr = demo.frontier(P4)
    lam = st.select_slider("Weight on booking value (0 = most likely booked, 1 = highest expected value)",
                           options=[round(x, 1) for x in fr["lambda"]], value=0.3)
    xf, yf = "booking value per search (USD)", "bookings per 100 searches"
    curve = alt.Chart(fr).mark_line(point=alt.OverlayMarkDef(color="grey"), color="grey").encode(
        x=alt.X(f"{xf}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{yf}:Q", scale=alt.Scale(zero=False)),
        tooltip=["lambda"])
    pts = alt.Chart(pt).encode(x=f"{xf}:Q", y=f"{yf}:Q",
                               color=alt.Color("ranking:N", scale=alt.Scale(domain=list(ROLE_COLOR),
                                                                             range=list(ROLE_COLOR.values())),
                                               legend=None),
                               tooltip=["ranking", alt.Tooltip(f"{yf}:Q", format=".2f"), alt.Tooltip(f"{xf}:Q", format="$.2f")])
    dots = pts.mark_point(size=110, filled=True)
    labels = pts.mark_text(align="left", dx=8, dy=-8, fontSize=12).encode(text="ranking:N")
    sel = alt.Chart(fr[fr["lambda"].round(1) == lam]).mark_point(size=260, shape="diamond", color="black").encode(
        x=f"{xf}:Q", y=f"{yf}:Q")
    st.altair_chart((curve + dots + labels + sel).properties(height=420), width="stretch")
    row = fr[fr["lambda"].round(1) == lam].iloc[0]
    b1 = P4["policies"]["B1_reviews_first"]
    better = row[yf] > 100 * b1["bookings_per_search"] and row[xf] > b1["value_per_search_usd"]
    st.markdown(f"Blend at {lam}: **{row[yf]:.2f}** bookings per 100 searches and **\\${row[xf]:.2f}** per search, "
                f"against reviews first's {100 * b1['bookings_per_search']:.2f} and \\${b1['value_per_search_usd']:.2f}"
                + (": better on both, in point estimates." if better else ": not better on both."))
    st.caption("Up and to the right is better. The blends are exploratory: the rule written before the analysis "
               "compared only the rankings in the table.")

# ------------------------------------------------------------------------------------------ test size
with tab_size:
    st.subheader("How big must the A/B test be?")
    st.markdown("Three arms (current ranking, reviews first, model blend) get a third of traffic each. The test must "
                "detect the blend's booking gain over reviews first **and** rule out a drop in booking value larger "
                "than the guardrail. Whichever needs more searches sets the size.")
    c1, c2 = st.columns(2)
    margin = c1.slider("Value guardrail: largest drop in booking value we must rule out (%)", 1.0, 10.0, 2.0, 0.5,
                       format="%.1f%%", key="size_margin") / 100
    options = sorted({round(x / 20, 2) for x in range(2, 21)} | {slope["slope"]})
    discount = c1.select_slider("Share of the offline gain we expect to see online", options=options,
                                value=slope["slope"], format_func=lambda v: f"{v:.0%}",
                                help=f"Default {slope['slope']:.0%}: the order-luck slope from phase 4, as in the memo.")
    comparisons = c2.selectbox("Comparisons sharing the 5% false-positive budget (Bonferroni)", [3, 2, 1], index=0)
    daily = c2.number_input("Clicked searches entering the test per day", 5_000, 2_000_000, 100_000, 5_000)
    cap = c2.checkbox("Cap booking value at the 99th percentile", value=True,
                      help="A few very expensive bookings make value noisy; capping them shrinks the test.")
    s = demo.size_test(P5, P4, margin=margin, discount=discount, comparisons=comparisons, daily_searches=daily,
                       cap_value=cap)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Searches per arm", f"{s['n_per_arm']:,}", border=True)
    m2.metric("Searches in total", f"{s['n_total']:,}", border=True)
    m3.metric("Days to run", f"{s['days']}", border=True, help="At least 14, so every weekday is seen twice.")
    m4.metric("Smallest booking gain it can detect", f"{s['mde']:.1%}", border=True,
              help="Relative lift in booking rate detectable with 80% power at this size.")
    need = pd.DataFrame({"requirement": ["Detect the booking gain", "Rule out the value drop"],
                         "searches per arm": [s["n_booking"] or 0, s["n_guardrail"]]})
    need["sets the size"] = need["searches per arm"] == need["searches per arm"].max()
    bars = alt.Chart(need).mark_bar().encode(
        y=alt.Y("requirement:N", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("searches per arm:Q", title="clicked searches per arm", scale=alt.Scale(domainMax=1.15 * float(need["searches per arm"].max()))),
        color=alt.Color("sets the size:N", scale=alt.Scale(domain=[True, False], range=[NAIVE, "#c9c8c3"]), legend=None),
        tooltip=["requirement", alt.Tooltip("searches per arm:Q", format=",")])
    st.altair_chart((bars + bars.mark_text(align="left", dx=4).encode(text=alt.Text("searches per arm:Q", format=",")))
                    .properties(height=170, title="Searches each requirement needs (the larger one sets the size)"),
                    width="stretch")
    st.markdown(
        f"The booking comparison ({s['lift']:+.1%} on a {P5['inputs']['baseline_booking_rate']:.1%} booking rate, "
        f"α = {s['alpha']:.4f}) needs **{s['n_booking']:,}** per arm; the value guardrail needs "
        f"**{s['n_guardrail']:,}**. The **{s['binding']}** sets the size.")
    st.caption("Sizes are per clicked search and treat searches as independent. In production the analysis "
               "population is every search with results and the unit is the visitor, which both raise the requirement.")

st.divider()
st.caption("Data: Expedia's hotel-search logs released for the ICDM 2013 Kaggle competition; the data belong to Expedia "
           "and are not redistributed. This app reads only aggregate results. Not affiliated with Expedia.")
