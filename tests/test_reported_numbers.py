"""Numbers quoted in the decision memo and README must match the committed records."""
from hotelrank.data import ROOT
from hotelrank.records import read_metrics

MEMO = (ROOT / "reports" / "decision_memo.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _expected():
    p2, p3 = read_metrics("02_position_effects"), read_metrics("03_choice_model")
    p4, p5 = read_metrics("04_policy_evaluation"), read_metrics("05_ab_test_design")
    raw = p2["raw_ratios"]
    f = p2["naive_over_random_ratio"]["click_bool"]["pos1_over_pos30"]
    pol = p4["policies"]
    blend = next(r for r in p4["frontier_exploratory"] if abs(r["lambda"] - 0.3) < 1e-9)
    dec, ev = p4["decision"], p4["evaluator"]
    lift = p5["lifts"]["T2 blend vs T1 reviews first"]
    shared = {
        f"{raw['algorithm/click_bool']['pos1_over_pos30']['ratio']:.1f}×",
        f"{raw['random/click_bool']['pos1_over_pos30']['ratio']:.1f}×",
        f"{f['lo']:.1f}–{f['hi']:.1f}",
        f"${p3['dollar_values']['click']['position 1 instead of 10']['usd']:.0f}",
        f"${p3['dollar_values']['booking']['position 1 instead of 10']['usd']:.0f}",
        f"{ev['predicted_booking_rate']:.2%} predicted",
        f"{ev['observed_booking_rate']:.2%} observed",
        f"{blend['bookings'] * 100:.2f}", f"${blend['value']:.2f}",
    }
    g, c = p4["guardrails"], p4["crossfit"][dec["M_star"]]
    shared |= {f"{g['blend_lambda_0.3']['no_review_booking_share']:.2%}", f"{g['B1_reviews_first']['no_review_booking_share']:.2%}",
               f"{g['B2_cheapest_first']['no_review_booking_share']:.2%}",
               f"${g['blend_lambda_0.3']['booked_nightly_price_usd']:.0f} a night",
               f"${g['B1_reviews_first']['booked_nightly_price_usd']:.0f}",
               f"{100 * c['bookings_cross']:.2f} against {100 * c['bookings_same']:.2f}",
               f"{p5['n_per_arm']:,}", f"{p5['n_primary']:,}"}
    shared |= {f"{v['bookings_per_search'] * 100:.2f}" for k, v in pol.items() if k != "logged_random_order"}
    shared |= {f"${v['value_per_search_usd']:.2f}" for k, v in pol.items() if k != "logged_random_order"}
    memo = shared | {
        f"about {f['factor']:.1f}-fold",
        f"{dec['bookings_rel_change']:+.1%} bookings", f"by {dec['value_rel_change']:+.1%}",
        f"{dec['cheapest_first_vs_reviews_first']['value_rel_change']:+.0%})",
        f"about {ev['order_luck_slope']['slope']:.2f}",
        f"from {ev['order_luck_slope']['ci'][0]:.2f} to {ev['order_luck_slope']['ci'][1]:.2f}",
        f"{p5['n_per_arm']:,} clicked searches per arm", f"{p5['n_total']:,} in total",
        f"{p5['n_primary']:,} per arm", f"{p5['mde_at_n_per_arm']:.1%} relative",
        f"{p5['inputs']['baseline_booking_rate']:.1%} booking rate",
        f"{lift['offline']:+.1%} offline", f"to {lift['discounted']:+.1%}",
        f"α = {p5['design']['alpha_per_comparison']:.4f}",
        f"about {p5['n_total'] / 1e6:.2f} million",
        f"about {round(max(p5['guardrail_n_by_margin']['5%'], p5['n_primary']), -3):,} per arm",
        f"about {max(p5['guardrail_n_by_margin']['5%'], p5['n_primary']) * 3 / 1e6:.2f} million",
        f"{next(d['days to run'] for d in p5['duration_days'] if d['clicked searches per day in test'] == 100_000)} days",
    }
    return memo, shared


def test_memo_numbers_match_records():
    memo, _ = _expected()
    missing = sorted(s for s in memo if s not in MEMO.replace("−", "-"))
    assert not missing, f"decision_memo.md does not quote: {missing}"


def test_readme_numbers_match_records():
    _, shared = _expected()
    missing = sorted(s for s in shared if s not in README.replace("−", "-"))
    assert not missing, f"README.md does not quote: {missing}"


def _expected_ltr():
    p6 = read_metrics("06_learning_to_rank")
    nd, pr, g, ck = p6["ndcg"], p6["paired"], p6["booking_model_evaluator"], p6["ndcg_checks"]

    def dd(k):
        v = pr[k]
        return f"{v['ndcg_diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"

    shared = {dd("L2_extended_random - blend_lambda_0.3"),
              f"{nd['L2_extended_random']['booked_in_top5']:.0%}", f"{nd['blend_lambda_0.3']['booked_in_top5']:.0%}",
              f"{g['L2_extended_random']['bookings_per_search'] * 100:.2f}",
              f"{g['blend_lambda_0.3']['bookings_per_search'] * 100:.2f}",
              f"{g['L2_extended_random']['no_review_booking_share']:.2%}",
              f"{g['B1_reviews_first']['no_review_booking_share']:.2%}"}
    readme = {f"{nd[k]['ndcg']:.3f}" for k in ("B1_reviews_first", "M1_most_likely_booked", "blend_lambda_0.3",
                                              "L1_core_random", "L2_extended_random", "L3_extended_ranked")}
    readme |= {dd(k) for k in ("L1_core_random - M1_most_likely_booked", "L2_extended_random - L1_core_random",
                               "L3_extended_ranked - L2_extended_random")}
    readme |= {f"{ck['order actually shown']['raw']:.3f}", f"{ck['order actually shown']['adjusted']:.3f}",
               f"{ck['random shuffle']['adjusted']:.3f}",
               f"{pr['L2_extended_random - L1_core_random']['raw_ndcg_diff']:+.3f}"}
    fa = p6["fourth_arm_sizing"]["designs"]
    base, four = fa["phase 5 as planned (3 arms)"], fa["add L2 (4 arms, all pairs)"]
    shared |= {f"{four['n_total']:,}", f"{base['n_total']:,}", f"{four['days_at_100k_per_day']} days instead of "
               f"{base['days_at_100k_per_day']}"}
    return shared, shared - {dd("L2_extended_random - blend_lambda_0.3")} | readme


def test_learning_to_rank_numbers_match_record():
    memo, readme = _expected_ltr()
    missing_memo = sorted(s for s in memo if s not in MEMO.replace("−", "-"))
    missing_readme = sorted(s for s in readme if s not in README.replace("−", "-"))
    assert not missing_memo, f"decision_memo.md does not quote: {missing_memo}"
    assert not missing_readme, f"README.md does not quote: {missing_readme}"
