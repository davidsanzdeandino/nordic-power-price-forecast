"""(a) Feature engineering only uses what is known at the day-ahead auction.

Method: build a synthetic master table with the column names the pipeline
produces, run the real ``generate_features`` on it, then rewrite every price
(and every same-instant congestion / flow column) for the hours after
``t0 - 24h`` and run it again. If no feature depends on prices with a lag below
24 h, every ``feat_*`` value up to and including ``t0`` must be unchanged.

A second perturbation test rewrites realised flows and commodity closes after
11:00 CET on D-1 and checks that no feature of delivery day D moves (they use a
48-hour lag), and a third checks that thresholds only use training rows.

The last part checks that the columns ``train.load_and_prepare`` hands to the
model exclude the master's same-instant flags (``is_congested_*``,
``is_negative_price_*``), raw prices and raw spreads, follow the per-zone rule
and are filled with training medians only.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import features as feat_mod
from src.model import train as train_mod
from src.transform import integration

START = "2025-08-01"
END = "2026-01-31 23:00"           # covers the train / val / test splits of train.py
T0 = pd.Timestamp("2025-11-15 12:00", tz="UTC")

PRICE_ZONES = ["DK1", "DK2", "FI", "NO2", "SE3", "SE4", "SE1", "EE"]
WX_LOCATIONS = ["EsbjergDK1", "AalborgDK1", "CopenhagenDK2", "KristiansandNO2",
                "StockholmSE3", "HelsinkiFI", "VaasaFI", "OlkiluotoFI", "PoriFI"]


def _diurnal(idx, amp, phase=0.0):
    return amp * np.sin(2 * np.pi * (idx.hour + phase) / 24)


def _make_base(rng) -> pd.DataFrame:
    idx = pd.date_range(START, END, freq="h", tz="UTC", name="timestamp_utc")
    n = len(idx)
    df = pd.DataFrame(index=idx)

    for k, z in enumerate(PRICE_ZONES):
        df[f"price_{z}"] = 60 + _diurnal(idx, 20, k) + rng.normal(0, 15, n)
    df["de_price_eur"] = 70 + _diurnal(idx, 25) + rng.normal(0, 15, n)

    for z in ["DK1", "DK2", "FI"]:
        df[f"forecast_{z}_Wind Onshore"] = rng.uniform(0, 2500, n)
        df[f"forecast_{z}_Solar"] = np.clip(_diurnal(idx, 800, -6), 0, None)
        df[f"load_fc_{z}_Forecasted Load"] = 3500 + _diurnal(idx, 600) + rng.normal(0, 100, n)
        df[f"load_{z}_Actual Load"] = df[f"load_fc_{z}_Forecasted Load"] + rng.normal(0, 80, n)
    for z in ["DK1", "DK2"]:
        df[f"forecast_{z}_Wind Offshore"] = rng.uniform(0, 1500, n)

    for loc in WX_LOCATIONS:
        df[f"wx_{loc}_wind_speed_100m"] = rng.uniform(0, 20, n)
        df[f"wx_{loc}_temperature_2m"] = 5 + _diurnal(idx, 4) + rng.normal(0, 3, n)
        df[f"wx_{loc}_solar_radiation"] = np.clip(_diurnal(idx, 400, -6), 0, None)

    for pair in ["DK1DK2", "DK1NO2"]:
        df[f"flow_{pair}_mw"] = rng.normal(0, 600, n)
        df[f"ntc_{pair}_mw"] = 1200.0

    df["ttf_eur_mwh"] = 35 + np.cumsum(rng.normal(0, 0.05, n))
    df["co2_eur_t"] = 70 + np.cumsum(rng.normal(0, 0.05, n))
    df["gas_marginal_cost_eur_mwh"] = df["ttf_eur_mwh"] / 0.55
    df["thermal_marginal_cost_eur_mwh"] = df["gas_marginal_cost_eur_mwh"] + 0.37 * df["co2_eur_t"]

    df["unavail_FI_total_mw"] = rng.uniform(0, 1600, n)
    df["unavail_SE3_total_mw"] = rng.uniform(0, 3000, n)
    df["unavail_SE3_nuclear_mw"] = df["unavail_SE3_total_mw"] * 0.6

    weekly = idx.tz_localize(None).to_period("W").astype(str)
    for z in ["NO2", "SE3"]:
        levels = pd.Series(rng.uniform(1e6, 5e6, weekly.nunique()), index=weekly.unique())
        df[f"reservoir_{z}_stored_mwh"] = levels.reindex(weekly).to_numpy()

    for a, b in [("DK1", "NO2"), ("FI", "EE")]:
        df[f"sched_{a}{b}_mw"] = rng.uniform(0, 1000, n)
        df[f"sched_{b}{a}_mw"] = rng.uniform(0, 1000, n)

    df["de_forecast_Wind Onshore"] = rng.uniform(0, 40000, n)
    df["de_forecast_Solar"] = np.clip(_diurnal(idx, 30000, -6), 0, None)
    df["de_load_forecast_mw"] = 55000 + _diurnal(idx, 8000)

    df["fingrid_FI_load_fc_mw"] = 9000 + _diurnal(idx, 1000)
    df["fingrid_FI_wind_fc_mw"] = rng.uniform(0, 6000, n)
    df["fingrid_FI_solar_fc_mw"] = np.clip(_diurnal(idx, 500, -6), 0, None)
    df["fingrid_FI_ol3_fc_mw"] = 1600.0

    return df


def _to_master(base: pd.DataFrame, logger) -> pd.DataFrame:
    """Add the columns integration.py derives (calendar, spreads, same-instant flags)."""
    m = base.copy()
    m["is_negative_price_DK1"] = (m["price_DK1"] < 0).astype(int)
    m = integration.step_calendar_features(m, logger)
    m = integration.step_derived_features(m, logger)
    return m


def _run_features(master: pd.DataFrame, tmp_path, name, monkeypatch, logger) -> pd.DataFrame:
    path = tmp_path / f"{name}.parquet"
    master.to_parquet(path)
    monkeypatch.setattr(feat_mod, "MASTER_PATH", path)
    monkeypatch.setattr(feat_mod, "get_logger", lambda _name: logger)
    monkeypatch.setattr(feat_mod, "save_run_metadata", lambda **_kw: None)
    return feat_mod.generate_features(save=False)


def _price_like_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c.startswith(("price_", "flow_")) or c == "de_price_eur"]


@pytest.fixture(scope="module")
def base_master():
    return _make_base(np.random.default_rng(7))


def test_no_feature_uses_prices_newer_than_24h(base_master, tmp_path, monkeypatch, quiet_logger):
    master = _to_master(base_master, quiet_logger)
    ref = _run_features(master, tmp_path, "ref", monkeypatch, quiet_logger)

    # Rewrite every price and realised flow from t0-23h onwards.
    perturbed = base_master.copy()
    late = perturbed.index > T0 - pd.Timedelta(hours=24)
    noise = np.random.default_rng(99)
    for col in _price_like_columns(perturbed):
        perturbed.loc[late, col] = perturbed.loc[late, col] + noise.normal(300, 200, late.sum())
    master_p = _to_master(perturbed, quiet_logger)
    # The derived same-instant flags really changed, so the test can see them.
    assert not master_p.loc[late, "is_congested_DK1_DK2"].equals(master.loc[late, "is_congested_DK1_DK2"])
    out = _run_features(master_p, tmp_path, "perturbed", monkeypatch, quiet_logger)

    feat_cols = [c for c in ref.columns if c.startswith("feat_")]
    assert len(feat_cols) > 150, "synthetic master should exercise most feature blocks"
    upto = ref.index <= T0
    a = ref.loc[upto, feat_cols].to_numpy(dtype=float)
    b = out.loc[upto, feat_cols].to_numpy(dtype=float)
    same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
    leaking = [c for c, ok in zip(feat_cols, same.all(axis=0)) if not ok]
    assert leaking == [], f"features that see prices with lag < 24 h: {leaking}"


def test_price_exactly_24h_old_is_visible(base_master, tmp_path, monkeypatch, quiet_logger):
    """Sanity check that the perturbation test above is able to detect a dependency."""
    master = _to_master(base_master, quiet_logger)
    ref = _run_features(master, tmp_path, "ref24", monkeypatch, quiet_logger)

    perturbed = base_master.copy()
    perturbed.loc[T0 - pd.Timedelta(hours=24), "price_DK1"] += 500.0
    out = _run_features(_to_master(perturbed, quiet_logger), tmp_path, "p24",
                        monkeypatch, quiet_logger)

    assert out.loc[T0, "feat_DK1_price_lag24h"] == pytest.approx(ref.loc[T0, "feat_DK1_price_lag24h"] + 500.0)
    # ...and nothing before t0 moved.
    before = ref.index < T0
    pd.testing.assert_series_equal(ref.loc[before, "feat_DK1_price_lag24h"],
                                   out.loc[before, "feat_DK1_price_lag24h"])


def test_training_matrix_excludes_same_instant_columns(base_master, tmp_path, monkeypatch, quiet_logger):
    master = _to_master(base_master, quiet_logger)
    feats = _run_features(master, tmp_path, "train_in", monkeypatch, quiet_logger)
    assert any(c.startswith("is_congested_") for c in feats.columns)
    assert any(c.startswith("spread_") for c in feats.columns)

    path = tmp_path / "dataset_features.parquet"
    feats.to_parquet(path)
    monkeypatch.setattr(train_mod, "FEATURES_PATH", path)

    for zone in ["DK1", "DK2", "FI"]:
        _df, X, y, feature_cols, splits, fill = train_mod.load_and_prepare(zone, quiet_logger)
        assert list(X.columns) == feature_cols
        forbidden = [c for c in feature_cols
                     if c.startswith(("is_congested_", "is_negative_price_", "price_",
                                      "spread_", "flow_", "ntc_", "load_"))
                     or c == "de_price_eur"]
        assert forbidden == [], f"{zone}: same-instant columns in the model input: {forbidden}"
        assert all(c.startswith("feat_") or c.startswith(("is_weekend", "is_holiday"))
                   for c in feature_cols)
        assert y.name == f"price_{zone}"
        for name in ("train", "val", "test"):
            assert splits[name].any()
        # Splits are disjoint and ordered in time.
        assert X.index[splits["train"]].max() < X.index[splits["val"]].min()
        assert X.index[splits["val"]].max() < X.index[splits["test"]].min()


DELIVERY_DAY = pd.Timestamp("2025-11-15", tz="Europe/Copenhagen")
REALISED_COLS = ("flow_", "ttf_", "co2_", "gas_marginal_", "thermal_marginal_")


def test_flows_and_commodities_after_11cet_of_d_minus_1_do_not_move_day_d(
        base_master, tmp_path, monkeypatch, quiet_logger):
    master = _to_master(base_master, quiet_logger)
    ref = _run_features(master, tmp_path, "ref48", monkeypatch, quiet_logger)

    known_until = (DELIVERY_DAY - pd.Timedelta(days=1) + pd.Timedelta(hours=11)).tz_convert("UTC")
    perturbed = base_master.copy()
    late = perturbed.index > known_until
    cols = [c for c in perturbed.columns if c.startswith(REALISED_COLS)]
    assert len(cols) >= 6
    noise = np.random.default_rng(5)
    for col in cols:
        perturbed.loc[late, col] = perturbed.loc[late, col] * 3 + noise.normal(50, 20, late.sum())
    out = _run_features(_to_master(perturbed, quiet_logger), tmp_path, "p48",
                        monkeypatch, quiet_logger)

    feat_cols = [c for c in ref.columns if c.startswith("feat_")]
    day_end = (DELIVERY_DAY + pd.Timedelta(days=1)).tz_convert("UTC")
    upto = ref.index < day_end                     # every hour up to the end of day D
    a = ref.loc[upto, feat_cols].to_numpy(dtype=float)
    b = out.loc[upto, feat_cols].to_numpy(dtype=float)
    same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
    leaking = [c for c, ok in zip(feat_cols, same.all(axis=0)) if not ok]
    assert leaking == [], f"features of day D that see data published after the auction: {leaking}"
    # The renamed 48-hour columns exist and the next day does move.
    for c in ("feat_ttf_lag48h", "feat_co2_lag48h", "feat_ntc_utilization_DK1DK2"):
        assert c in ref.columns
    later = ref.index >= day_end + pd.Timedelta(hours=24)
    assert not np.allclose(ref.loc[later, "feat_ttf_lag48h"], out.loc[later, "feat_ttf_lag48h"])


def test_thresholds_use_training_rows_only(base_master, tmp_path, monkeypatch, quiet_logger):
    master = _to_master(base_master, quiet_logger)
    ref = _run_features(master, tmp_path, "thr_ref", monkeypatch, quiet_logger)
    perturbed = base_master.copy()
    after = perturbed.index > pd.Timestamp(train_mod.TRAIN_END, tz="UTC")
    for col in [c for c in perturbed.columns if c.startswith("unavail_")]:
        perturbed.loc[after, col] *= 10.0
    perturbed.loc[after, "forecast_DK1_Wind Onshore"] *= 10.0
    out = _run_features(_to_master(perturbed, quiet_logger), tmp_path, "thr_p",
                        monkeypatch, quiet_logger)
    before = ref.index <= pd.Timestamp(train_mod.TRAIN_END, tz="UTC")
    for col in ("feat_unavail_high", "feat_DK1_unavail_low_wind", "feat_FI_unavail_low_wind"):
        pd.testing.assert_series_equal(ref.loc[before, col], out.loc[before, col])


def test_feature_rule_and_train_only_fill_values(base_master, tmp_path, monkeypatch, quiet_logger):
    from src.model.config import select_feature_cols
    master = _to_master(base_master, quiet_logger)
    feats = _run_features(master, tmp_path, "fill", monkeypatch, quiet_logger)
    # NaN in a feature column: the fill value must come from training rows only.
    col = "feat_DK1_residual_load"
    after = feats.index > pd.Timestamp(train_mod.TRAIN_END, tz="UTC")
    feats.loc[after, col] += 1e6                      # val/test values far away
    hole = feats.index[after][100]
    feats.loc[hole, col] = np.nan
    path = tmp_path / "dataset_features.parquet"
    feats.to_parquet(path)
    monkeypatch.setattr(train_mod, "FEATURES_PATH", path)

    _df, X, _y, cols_dk, splits, fill = train_mod.load_and_prepare("DK1", quiet_logger)
    train_median = feats.loc[feats.index[~after], col].median()
    assert fill[col] == pytest.approx(train_median)
    assert X.loc[hole, col] == pytest.approx(train_median)

    _df, _X, _y, cols_fi, _s, _f = train_mod.load_and_prepare("FI", quiet_logger)
    assert cols_dk == select_feature_cols(feats.columns, "DK1")
    fi_block = [c for c in cols_fi if c not in cols_dk]
    assert fi_block and all(
        c.startswith(("feat_FI_", "feat_SE1_", "feat_SE3_", "feat_SE4_", "feat_EE_",
                      "feat_sched_net_FI_", "feat_post_ol3")) for c in fi_block)
    assert [c for c in cols_fi if c in cols_dk] == cols_dk     # same order
