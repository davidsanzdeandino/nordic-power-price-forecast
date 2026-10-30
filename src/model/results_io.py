"""
results_io.py
=============
Helpers to write the published result files (results/*.json) from code:
numpy and pandas values become plain JSON, NaN becomes null, and the keys of
pandas objects become strings. No local paths are written: callers pass paths
relative to the repository root.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def to_jsonable(obj):
    """Recursively convert numpy / pandas objects into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def rounded(obj, ndigits: int = 4):
    """Round every float in a JSON-safe structure."""
    if isinstance(obj, dict):
        return {k: rounded(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, ndigits) for v in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


def write_json(path, obj, ndigits: int | None = 4) -> Path:
    """Write obj as indented UTF-8 JSON (floats rounded to ndigits)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_jsonable(obj)
    if ndigits is not None:
        data = rounded(data, ndigits)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path
