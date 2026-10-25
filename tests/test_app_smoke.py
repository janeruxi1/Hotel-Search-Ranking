"""The demo is what most visitors open; nothing else runs it, so render it headlessly."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TABS = ["The answer", "Position effects", "What a top slot is worth", "Rankings compared", "Size the test"]


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_every_tab_renders(app):
    labels = [t.label for t in app.tabs]
    assert all(any(name in lab for lab in labels) for name in TABS), labels
    assert app.title and len(app.subheader) >= len(TABS)


def test_position_outcome_switch(app):
    app.radio(key="pos_outcome").set_value("Bookings").run()
    assert not app.exception, app.exception


def test_slot_value_handles_a_move_down_the_list(app):
    app.selectbox(key="slot_from").set_value(2).run()
    app.selectbox(key="slot_to").set_value(20).run()
    assert not app.exception, app.exception
    assert any("rise" in m.label for m in app.metric)


def test_test_size_reacts_to_the_guardrail(app):
    before = [m.value for m in app.metric]
    app.slider(key="size_margin").set_value(5.0).run()
    assert not app.exception, app.exception
    assert [m.value for m in app.metric] != before
