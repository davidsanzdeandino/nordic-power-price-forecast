"""Energi Data Service rows are deduplicated by timestamp and zone, not by timestamp alone.

Every hour of Elspotprices, ElectricityBalanceNonv and ProductionConsumptionSettlement
holds one row per zone (DK1 and DK2). Deduplicating on the timestamp alone kept a single
zone per hour (see DT-23 in docs/decisions.md).
"""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "energinet_dk_extract", ROOT / "src" / "extract" / "04_energinet_dk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(hours: int = 48) -> pd.DataFrame:
    """API-shaped records: one row per zone and hour, plus an exact repeat of some rows."""
    ts = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for t in ts:
        for area, price in (("DK1", 50.0), ("DK2", 70.0)):
            rows.append({"HourUTC": t.strftime("%Y-%m-%dT%H:%M:%S"),
                         "HourDK": t.strftime("%Y-%m-%dT%H:%M:%S"),
                         "PriceArea": area, "SpotPriceEUR": price + t.hour})
    df = pd.DataFrame(rows)
    # A repeated page boundary: the first three hours arrive twice
    return pd.concat([df, df.iloc[:6]], ignore_index=True)


def test_process_and_save_keeps_both_zones(tmp_path, monkeypatch, quiet_logger):
    mod = _load_extractor()
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
    config = {"date_col": "HourUTC"}

    n = mod.process_and_save(_records(48), "Elspotprices", config, quiet_logger)

    saved = pd.read_parquet(tmp_path / "energinet_Elspotprices.parquet")
    assert n == len(saved) == 2 * 48
    assert saved.groupby("PriceArea").size().to_dict() == {"DK1": 48, "DK2": 48}
    assert not saved.reset_index().duplicated(subset=["timestamp_utc", "PriceArea"]).any()
    assert saved.index.is_monotonic_increasing


def test_drop_duplicate_rows_without_area_column():
    mod = _load_extractor()
    idx = pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 00:00", "2025-01-01 01:00"], tz="UTC")
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx)
    out = mod.drop_duplicate_rows(df)
    assert out["value"].tolist() == [1.0, 3.0]


def test_cleaning_remove_duplicates_by_zone():
    from src.transform.cleaning import remove_duplicates

    raw = _records(24)
    df = raw.assign(HourUTC=pd.to_datetime(raw["HourUTC"], utc=True)).set_index("HourUTC")

    by_zone, n_zone = remove_duplicates(df, area_col="PriceArea")
    assert len(by_zone) == 2 * 24 and n_zone == 6
    assert by_zone.groupby("PriceArea").size().to_dict() == {"DK1": 24, "DK2": 24}

    # Without area_col the old behaviour (index only) is unchanged
    by_index, _ = remove_duplicates(df)
    assert len(by_index) == 24
