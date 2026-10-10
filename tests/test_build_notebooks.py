import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)


def test_parse_percent_splits_markdown_and_code():
    text = "# %% [markdown]\n# # Title\n#\n# Some text\n\n# %%\nx = 1\n\n\n# %%\nprint(x)\n"
    cells = bn.parse_percent(text)
    assert [c["cell_type"] for c in cells] == ["markdown", "code", "code"]
    assert cells[0]["source"] == "# Title\n\nSome text"
    assert cells[1]["source"] == "x = 1"


def test_notebook_has_ids_and_empty_outputs():
    nb = bn.to_notebook(bn.parse_percent("# %%\na = 2\n"))
    assert nb["nbformat"] == 4 and nb["cells"][0]["id"] == "0"
    assert nb["cells"][0]["outputs"] == [] and nb["cells"][0]["execution_count"] is None


def test_every_committed_notebook_matches_its_source():
    assert bn.check() == []


def test_committed_notebooks_hold_no_local_paths():
    for nb in (ROOT / "notebooks").glob("*.ipynb"):
        text = nb.read_text(encoding="utf-8")
        assert "C:\\\\Users" not in text and "/sessions/" not in text and "/home/" not in text, nb.name
        json.loads(text)
