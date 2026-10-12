import numpy as np
import pytest

from hotelrank import audit, choice, policy, position
from hotelrank.synthetic import make_impressions


@pytest.fixture(scope="module")
def setup():
    df = make_impressions(n_searches=2500, seed=4)
    flags = audit.price_flags(df, audit.per_search(df)).reset_index().rename(columns={"index": "srch_id"})
    x = position.analysis_sample(df, flags)
    x = policy.prepare(choice.add_hotel_mean_price(x[x["random_bool"].astype(bool)], full=x))
    test = (x["srch_id"] % 5 == 0).to_numpy()
    return x, policy.fit_booking_model(x[~test]), policy.make_eval_set(x[test])


def test_outside_rows_one_per_search_and_one_choice(setup):
    x, _, _ = setup
    both = policy.with_outside_rows(x)
    g = both.groupby("srch_id")
    assert (g["outside"].sum() == 1).all() and (g["chosen"].sum() == 1).all()
    booked = x.groupby("srch_id")["booking_bool"].max().astype(bool)
    out_chosen = both[both["outside"] == 1].set_index("srch_id")["chosen"]
    assert (out_chosen == ~booked.reindex(out_chosen.index)).all()


def test_assign_slots_gives_best_score_the_top_slot(setup):
    _, _, e = setup
    score = np.random.default_rng(0).random(len(e.slots))
    new = policy.assign_slots(e, score)
    for s in range(5):
        rows = np.flatnonzero(e.search == s)
        assert sorted(new[rows]) == sorted(e.slots[rows])           # same slots, re-assigned
        assert new[rows[np.argmax(score[rows])]] == e.slots[rows].min()


def test_outcomes_match_hand_calculation():
    e = policy.EvalSet(search=np.array([0, 0]), slots=np.array([1, 2]), A=np.zeros((2, len(policy.HOTEL_ATTRIBUTES))),
                       W=np.zeros((1, len(policy.OUTSIDE_COVARIATES))), value=np.array([100.0, 300.0]),
                       review=np.zeros(2), no_reviews=np.zeros(2), price=np.ones(2), booked_any=np.zeros(1),
                       starts=np.array([0]), search_ids=np.array([7]))
    pos = np.zeros(41)
    pos[2] = np.log(0.5)                         # slot 2 is half as attractive as slot 1
    p = policy.Params(b_hotel=np.zeros(len(policy.HOTEL_ATTRIBUTES)), b_out=np.zeros(len(policy.OUTSIDE_COLUMNS)),
                      pos_effect=pos)
    out = policy.outcomes(e, p, e.slots)
    # exp-utilities: hotel1 1, hotel2 0.5, outside 1 -> P(book) = 1.5 / 2.5
    assert out["bookings"] == pytest.approx(0.6)
    assert out["value"] == pytest.approx((1 * 100 + 0.5 * 300) / 2.5)


def test_most_likely_booked_beats_random_orders_under_the_model(setup):
    _, f, e = setup
    p = policy.Params.from_vector(f.names, f.beta)
    p.pos_effect = -np.log(np.arange(len(p.pos_effect)) + 1.0)   # strictly decreasing position effect
    best = policy.outcomes(e, p, policy.assign_slots(e, policy.hotel_utility(e, p)))["bookings"]
    rng = np.random.default_rng(1)
    for _ in range(5):
        other = policy.outcomes(e, p, policy.assign_slots(e, rng.random(len(e.slots))))["bookings"]
        assert best >= other - 1e-12


def test_decision_rule_cases():
    def result(m1, m2, b1, b2):
        rng = np.random.default_rng(0)
        pts = {"B1_reviews_first": b1, "B2_cheapest_first": b2, "M1_most_likely_booked": m1,
               "M2_highest_expected_value": m2}
        draws = {k: np.column_stack([v[0] + rng.normal(0, 0.001, 300), v[1] + rng.normal(0, 0.1, 300)])
                 for k, v in pts.items()}
        return {"point": pts, "draws": draws}

    ok = policy.apply_decision_rule(result((0.15, 50), (0.14, 52), (0.13, 50), (0.12, 40)))
    assert ok["recommend"] == "M1_most_likely_booked" and ok["failed"] == []
    rev = policy.apply_decision_rule(result((0.15, 45), (0.14, 52), (0.13, 50), (0.12, 40)))
    assert rev["recommend"] == "B1_reviews_first" and rev["failed"] == ["booking value more than 2% lower"]
    tie = policy.apply_decision_rule(result((0.1301, 50), (0.12, 52), (0.13, 50), (0.12, 40)))
    assert tie["recommend"] == "B1_reviews_first" and "bookings interval not above zero" in tie["failed"]


def test_evaluator_checks_run(setup):
    _, f, e = setup
    cal = policy.evaluator_calibration(e, f)
    luck = policy.order_luck_check(e, f, n_shuffles=3)
    assert len(cal) > 1 and {"slope", "se", "ci"} <= set(luck.attrs["slope"])
