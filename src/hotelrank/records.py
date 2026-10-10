"""Writing measured records. Only results computed on the real data are written to
``reports/metrics``; synthetic runs go to a scratch folder so they never overwrite them."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from .data import METRICS_DIR


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def metrics_dir(source: str) -> Path:
    if source == "real":
        return METRICS_DIR
    d = Path(tempfile.gettempdir()) / "hotelrank_synthetic_metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_metrics(name: str, payload: dict, source: str) -> Path:
    out = metrics_dir(source) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:  # LF on every OS, so git sees no churn
        fh.write(json.dumps(_clean({"source": source, **payload}), indent=2) + "\n")
    return out


def read_metrics(name: str, directory: Path = METRICS_DIR) -> dict:
    return json.loads((Path(directory) / f"{name}.json").read_text())
