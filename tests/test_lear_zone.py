"""(d) The LEAR baseline uses the exogenous wind (and temperature) of its own zone."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.baselines import LEARBaseline

ROOT = Path(__file__).resolve().parents[1]
ZONES = ["DK1", "DK2", "FI"]


def _extra(rng, idx):
    """Distinct wind and temperature per zone, named as features.py names them."""
    df = pd.DataFrame(index=idx)
    for k, z in enumerate(ZONES):
        df[f"feat_{z}_wind_total_fc"] = rng.uniform(0, 3000, len(idx)) + 10_000 * k
        df[f"feat_wx_{z}_temp_avg"] = rng.normal(5, 3, len(idx)) + 100 * k
    return df


@pytest.mark.parametrize("zone", ZONES)
def test_build_features_takes_own_zone_columns(rng, zone):
    idx = pd.date_range("2025-01-01", periods=24 * 20, freq="h", tz="UTC")
    extra = _extra(rng, idx)
    series = pd.Series(rng.normal(60, 10, len(idx)), index=idx)
    X = LEARBaseline(zone=zone)._build_features(series, extra)
    pd.testing.assert_series_equal(X["wind_forecast"], extra[f"feat_{zone}_wind_total_fc"],
                                   check_names=False)
    pd.testing.assert_series_equal(X["temperature"], extra[f"feat_wx_{zone}_temp_avg"],
                                   check_names=False)


def test_fallback_to_entsoe_forecast_columns_of_own_zone(rng):
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    extra = pd.DataFrame(index=idx)
    extra["forecast_DK1_Wind Onshore"] = 1.0
    extra["forecast_DK2_Wind Onshore"] = 2.0
    extra["forecast_DK2_Wind Offshore"] = 3.0
    series = pd.Series(rng.normal(60, 10, len(idx)), index=idx)
    X = LEARBaseline(zone="DK2")._build_features(series, extra)
    assert (X["wind_forecast"] == 5.0).all()


def test_lear_fit_benefits_from_own_zone_wind(rng):
    """Price driven by FI wind: the FI LEAR must beat a LEAR fed with DK1 wind."""
    idx = pd.date_range("2025-01-01", periods=24 * 120, freq="h", tz="UTC")
    extra = _extra(rng, idx)
    w_fi = extra["feat_FI_wind_total_fc"]
    price = pd.Series(120 - 0.03 * (w_fi - w_fi.mean()) + rng.normal(0, 3, len(idx)), index=idx)
    test_start = "2025-04-01"

    m_fi = LEARBaseline(zone="FI").evaluate(price, test_start, df_extra=extra)
    # Same data, but this LEAR only sees DK1 wind, which does not drive the price.
    m_dk1 = LEARBaseline(zone="DK1").evaluate(price, test_start, df_extra=extra)
    assert m_fi["MAE"] < 0.5 * m_dk1["MAE"]


def _lear_calls_without_zone():
    missing = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None)) == "LEARBaseline"
                    and not any(kw.arg == "zone" for kw in node.keywords)
                    and len(node.args) < 2):
                missing.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    return missing


def test_lear_call_sites_in_baselines_and_dashboard_pass_zone():
    missing = [m for m in _lear_calls_without_zone() if not m.startswith("src/model/evaluate.py")]
    assert missing == []


def test_every_lear_call_site_passes_zone():
    """Includes src/model/evaluate.py, whose LEAR row once used DK1 weather for DK2 and FI."""
    assert _lear_calls_without_zone() == []
