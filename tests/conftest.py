import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synth():
    from hotelrank.synthetic import make_impressions

    return make_impressions(n_searches=1500, seed=1)
