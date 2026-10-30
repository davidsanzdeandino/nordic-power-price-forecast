"""
leakage_sensitivity.py — Efecto medido de las limitaciones de disponibilidad de datos
=====================================================================================
Tres fuentes no guardan la hora en que se publicó cada dato, así que no se puede
reconstruir con exactitud lo que se sabía a las 12:00 CET de D-1. Este script
mide, con los modelos ya entrenados y solo en la predicción del test (ene-may
2026), cuánto cambia el MAE servido cuando esas entradas se sustituyen por
versiones más conservadoras:

  meteo_previous_day1 / meteo_previous_day2
      Meteo de la Previous Runs API de Open-Meteo (pasada del modelo de 1 o 2
      días antes) en lugar de la Historical Forecast API, que une las primeras
      horas de cada pasada (plazo más corto que en la subasta).
  remit_known_11cet
      Indisponibilidades REMIT: todas las horas de D con el valor vigente a las
      11:00 CET de D-1 (el master las agrega por inicio y fin del evento, sin
      la hora de publicación).
  wind_solar_fc_d_minus_1  (cota pesimista)
      Previsiones eólicas y solares de ENTSO-E (Alemania y zonas nórdicas) y de
      Fingrid sustituidas por las del día anterior (24 h antes). El plazo legal
      de la eólica y la solar day-ahead es las 18:00 de D-1 (Reglamento
      543/2013, art. 14.2.d). Mide cuánto depende el modelo de esas previsiones,
      no estima la fuga.

Método: copia del master -> modificación -> features.generate_features(save=False,
df_master=...) -> PricePredictor(zone, feature_data=...) -> predict_with_bands en
el test (predict.served_predictions). Una variante «control» regenera las
features del master sin tocar: su MAE tiene que coincidir con el servido.

Las tablas de cada variante se guardan (solo desde diciembre de 2025) en
data/raw/leakage_sensitivity/ y se reutilizan entre zonas. La descarga de
Open-Meteo va a data/raw/openmeteo_previous_runs.parquet. data/raw/ no se publica.

Escribe results/leakage_sensitivity_test2026.json (añade o sustituye la zona).

Uso:  python scripts/leakage_sensitivity.py DK1 [DK2 FI]
"""

import sys
import json
import time
from datetime import datetime
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MASTER_PATH = ROOT / "data" / "master" / "dataset_master.parquet"
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "leakage_sensitivity"
PREV_RUNS_PATH = RAW_DIR / "openmeteo_previous_runs.parquet"
OUT_PATH = ROOT / "results" / "leakage_sensitivity_test2026.json"

PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WX_VARS = ["temperature_2m", "precipitation", "wind_speed_10m",
           "wind_speed_100m", "wind_direction_100m", "shortwave_radiation"]
NON_NEGATIVE = ("precipitation", "wind_speed_10m", "wind_speed_100m", "solar_radiation")
# Ventana descargada y guardada: el test más el historial que usan los lags,
# los rolling de 7 días y el AR (residuos de diciembre).
WINDOW_START = "2025-12-01"
WINDOW_END = "2026-05-31"
KEEP_FROM = pd.Timestamp(WINDOW_START, tz="UTC")

VARIANTS = {
    "control": "features regenerated from the unmodified master (must match the served MAE)",
    "meteo_previous_day1": "Open-Meteo Previous Runs API, run of the previous day (_previous_day1)",
    "meteo_previous_day2": "Open-Meteo Previous Runs API, run of two days before (_previous_day2)",
    "remit_known_11cet": "REMIT unavailability for every hour of D set to the value in force "
                         "at 11:00 CET on D-1",
    "wind_solar_fc_d_minus_1": "pessimistic bound: ENTSO-E (Germany and Nordic zones) and Fingrid "
                               "wind and solar day-ahead forecasts replaced by those of the "
                               "previous day (24 h earlier)",
}


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weather_locations() -> dict:
    """{nombre en el master (p. ej. EsbjergDK1): (lat, lon)} desde los extractores."""
    m03 = _load_module("src/extract/03_openmeteo_weather.py", "om03")
    m08 = _load_module("src/extract/08_finland_coupling.py", "om08")
    locs = {k.replace("_", ""): (v["lat"], v["lon"]) for k, v in m03.LOCATIONS.items()}
    locs.update({k: (v["lat"], v["lon"]) for k, v in m08.FI_WEATHER.items()})
    return locs


def download_previous_runs(locations: dict) -> pd.DataFrame:
    """Descarga (o lee de data/raw/) la meteo de pasadas anteriores, columnas wx_<loc>_<var>_<lag>."""
    if PREV_RUNS_PATH.exists():
        return pd.read_parquet(PREV_RUNS_PATH)
    import requests
    hourly = [f"{v}_previous_day{d}" for v in WX_VARS for d in (1, 2)]
    frames = {}
    for name, (lat, lon) in locations.items():
        params = {"latitude": lat, "longitude": lon, "start_date": WINDOW_START,
                  "end_date": WINDOW_END, "hourly": ",".join(hourly), "timezone": "UTC"}
        r = requests.get(PREV_RUNS_URL, params=params, timeout=180)
        r.raise_for_status()
        h = r.json()["hourly"]
        idx = pd.to_datetime(h.pop("time"), utc=True)
        for key, vals in h.items():
            var, lag = key.rsplit("_previous_", 1)          # lag = day1 / day2
            var = "solar_radiation" if var == "shortwave_radiation" else var
            frames[f"wx_{name}_{var}_{lag}"] = pd.Series(vals, index=idx, dtype=float)
        print(f"  Open-Meteo previous runs: {name} OK")
        time.sleep(2.0)
    df = pd.DataFrame(frames)
    PREV_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PREV_RUNS_PATH)
    return df


def variant_master(name: str, master: pd.DataFrame) -> tuple:
    """Copia modificada del master para la variante y nº de celdas cambiadas."""
    m = master.copy()
    changed = 0
    if name == "control":
        return m, 0
    if name.startswith("meteo_previous_day"):
        lag = name.rsplit("_", 1)[-1]                      # day1 / day2
        prev = download_previous_runs(weather_locations())
        for col in [c for c in m.columns if c.startswith("wx_")]:
            src = f"{col}_{lag}"
            if src not in prev.columns:
                continue
            new = prev[src].reindex(m.index)
            if col.endswith(NON_NEGATIVE):
                new = new.clip(lower=0)
            ok = new.notna()
            changed += int((ok & (m[col] != new)).sum())
            m.loc[ok, col] = new[ok]
        return m, changed
    if name == "remit_known_11cet":
        local = m.index.tz_convert("Europe/Copenhagen").tz_localize(None)
        ref = (local.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=11))
        ref = ref.tz_localize("Europe/Copenhagen").tz_convert("UTC")
        for col in [c for c in m.columns if c.startswith("unavail_")]:
            new = pd.Series(m[col].reindex(ref).to_numpy(), index=m.index)
            ok = new.notna()
            changed += int((ok & (m[col] != new)).sum())
            m.loc[ok, col] = new[ok]
        return m, changed
    if name == "wind_solar_fc_d_minus_1":
        cols = [c for c in m.columns
                if (c.startswith(("forecast_", "de_forecast_")) and ("Wind" in c or "Solar" in c))
                or c in ("fingrid_FI_wind_fc_mw", "fingrid_FI_solar_fc_mw")]
        for col in cols:
            new = m[col].shift(freq="24h").reindex(m.index)
            changed += int((new != m[col]).sum())
            m[col] = new
        return m, changed
    raise ValueError(name)


def variant_features(name: str, master: pd.DataFrame) -> tuple:
    """Tabla de features de la variante (desde KEEP_FROM), con caché en data/raw/."""
    from src.model.features import generate_features
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"features_{name}.parquet"
    meta_path = CACHE_DIR / f"features_{name}.json"
    # La caché vale solo si es posterior al master y al código que genera las
    # features; si no, se regenera (si no, un cambio en features.py devolvería
    # sin aviso las features antiguas).
    inputs = (MASTER_PATH, ROOT / "src" / "model" / "features.py",
              ROOT / "src" / "model" / "config.py")
    newest_input = max(p.stat().st_mtime for p in inputs)
    if path.exists() and meta_path.exists() and path.stat().st_mtime > newest_input:
        return pd.read_parquet(path), json.load(open(meta_path))["cells_changed"]
    m, changed = variant_master(name, master)
    feats = generate_features(save=False, df_master=m)
    feats = feats[feats.index >= KEEP_FROM]
    feats.to_parquet(path)
    json.dump({"cells_changed": changed}, open(meta_path, "w"))
    return feats, changed


def main():
    from src.model.config import TEST_START, TEST_END
    from src.model.predict import PricePredictor, served_predictions
    from src.model.results_io import write_json

    zones = sys.argv[1:] or ["DK1"]
    master = pd.read_parquet(MASTER_PATH).sort_index()

    out = json.load(open(OUT_PATH, encoding="utf-8")) if OUT_PATH.exists() else {}
    out.update({
        "description": ("Change in the served test MAE (January to May 2026) when inputs whose "
                        "publication time is not stored are replaced by more conservative "
                        "versions. Models are not retrained; only the prediction changes."),
        "generated_by": "scripts/leakage_sensitivity.py",
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "method": ("copy of the master, one input replaced per variant, features regenerated "
                   "(features.generate_features), served models of src/model/train.py "
                   "unchanged, prediction with predict.served_predictions over the test window"),
        "test_window_utc": [TEST_START, TEST_END],
        "reference": ("served route: predict_with_bands on the feature table that step 14 "
                      "writes from the unmodified master (local, not in the repository)"),
        "variants": VARIANTS,
    })
    out.setdefault("zones", {})

    for zone in zones:
        ref = served_predictions(zone, TEST_START, TEST_END)
        mae_ref = float((ref["price_pred"] - ref["price_actual"]).abs().mean())
        zres = {"generated_on": datetime.now().strftime("%Y-%m-%d"),
                "n_hours": int(len(ref)), "MAE_served": mae_ref, "variants": {}}
        mae_control = None
        for name in VARIANTS:
            feats, changed = variant_features(name, master)
            pred = PricePredictor(zone=zone, feature_data=feats)
            g = served_predictions(zone, TEST_START, TEST_END, predictor=pred)
            g = g.reindex(ref.index).dropna(subset=["price_pred"])
            mae = float((g["price_pred"] - ref.loc[g.index, "price_actual"]).abs().mean())
            if name == "control":
                mae_control = mae
            row = {"MAE": mae, "delta_vs_served": mae - mae_ref,
                   "n_hours": int(len(g)), "master_cells_changed": int(changed)}
            if mae_control is not None and name != "control":
                row["delta_vs_control"] = mae - mae_control
            zres["variants"][name] = row
            print(f"{zone} {name:26s} MAE {mae:7.3f}  Δ vs served {mae - mae_ref:+.3f}")
        out["zones"][zone] = zres

    path = write_json(OUT_PATH, out)
    print(f"OK -> results/{path.name}")


if __name__ == "__main__":
    main()
