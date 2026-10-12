"""The demo is what most visitors open; nothing else runs it, so render every mode headlessly."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
MODES = ["The answer", "Position effects", "What a top slot is worth", "Rankings compared", "Size the test"]


def test_landing_page_offers_every_mode():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert list(at.sidebar.radio[0].options) == MODES


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_renders(mode):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception, at.exception
    assert at.title


def test_slot_value_handles_a_move_down_the_list():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("What a top slot is worth").run()
    at.selectbox[0].set_value(2).run()
    at.selectbox[1].set_value(20).run()
    assert not at.exception, at.exception
    assert "rise" in at.metric[0].label
