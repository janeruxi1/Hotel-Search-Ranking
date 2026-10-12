import pytest

from hotelrank import demo
from hotelrank.records import read_metrics

P3, P4, P5 = (read_metrics(n) for n in ("03_choice_model", "04_policy_evaluation", "05_ab_test_design"))


def test_position_value_matches_the_notebook_table():
    price = P3["median_price_usd"]
    for kind in ("click", "booking"):
        for k in (2, 10, 30):
            v = demo.position_value(P3["fits"][kind], from_pos=k, to_pos=1, price=price)
            t = P3["dollar_values"][kind][f"position 1 instead of {k}"]
            assert v["usd"] == pytest.approx(t["usd"], abs=0.01)
            assert v["usd_lo"] == pytest.approx(t["usd_lo"], abs=0.01)
            assert v["usd_hi"] == pytest.approx(t["usd_hi"], abs=0.01)


def test_position_value_is_antisymmetric_and_rejects_reserved_slots():
    f = P3["fits"]["click"]
    up, down = demo.position_value(f, 10, 3, 100), demo.position_value(f, 3, 10, 100)
    assert up["log_units"] == pytest.approx(-down["log_units"])
    with pytest.raises(ValueError):
        demo.position_value(f, 5, 1, 100)


def test_size_test_reproduces_phase_5():
    s = demo.size_test(P5, P4, margin=P5["design"]["value_margin"], discount=P4["evaluator"]["order_luck_slope"]["slope"])
    # the demo's default settings must reproduce the memo's design exactly
    assert (s["n_booking"], s["n_guardrail"], s["n_total"]) == (P5["n_primary"], P5["n_guardrail"], P5["n_total"])


def test_tables_have_every_policy_and_blend():
    assert len(demo.policy_table(P4)) == len(P4["policies"])
    assert len(demo.frontier(P4)) == len(P4["frontier_exploratory"])
    c = demo.position_curves(read_metrics("02_position_effects"), "click_bool")
    assert set(c["series"]) == {"Expedia's ranking (naive)", "Random order"}
