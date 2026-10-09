import pandas as pd

from hotelrank.synthetic import make_impressions


def test_schema_matches_prepared_table(synth):
    expected = ["srch_id", "date_time", "site_id", "visitor_location_country_id", "visitor_hist_starrating",
                "visitor_hist_adr_usd", "prop_country_id", "prop_id", "prop_starrating", "prop_review_score",
                "prop_brand_bool", "prop_location_score1", "prop_location_score2", "prop_log_historical_price",
                "position", "price_usd", "promotion_flag", "srch_destination_id", "srch_length_of_stay",
                "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count",
                "srch_saturday_night_bool", "srch_query_affinity_score", "orig_destination_distance",
                "random_bool", "click_bool", "gross_bookings_usd", "booking_bool", "comp_n_known",
                "comp_n_expedia_cheaper", "comp_n_expedia_dearer"]
    assert list(synth.columns) == expected
    assert synth["date_time"].dtype == "datetime64[ns]"


def test_selection_rules(synth):
    g = synth.groupby("srch_id")
    assert (g["click_bool"].sum() >= 1).all()
    assert (g["booking_bool"].sum() <= 1).all()
    assert not (synth["booking_bool"] & ~synth["click_bool"]).any()


def test_deterministic():
    a, b = make_impressions(n_searches=50, seed=3), make_impressions(n_searches=50, seed=3)
    pd.testing.assert_frame_equal(a, b)
