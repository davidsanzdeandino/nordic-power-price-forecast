"""
08_finland_coupling.py — Datos de acoplamiento del mercado finlandés (FI)
=========================================================================
Da a Finlandia la MISMA profundidad de acoplamiento que tiene Dinamarca vía
Alemania. Para FI el driver dominante es Suecia (SE3/SE4 ya presentes) + el
norte sueco SE1 (hidro/eólica barata que empuja FI a negativo) y Estonia (EE,
vía EstLink, la válvula de exportación del sur).

Trae 3 fuentes (todas gratuitas, reutilizando el token ENTSO-E existente +
Open-Meteo sin clave) y las UNE ADITIVAMENTE a dataset_master.parquet — DK queda
intacto:

  1. Precios day-ahead SE1 y EE (ENTSO-E)            -> price_SE1, price_EE
  2. Intercambios programados FI↔EE / FI↔SE3 / FI↔SE1 (ENTSO-E)
        -> sched_FIEE_mw, sched_EEFI_mw, sched_FISE_3_mw, ...
  3. Meteo Vaasa / Olkiluoto / Pori (Open-Meteo historical-forecast, leakage-safe)
        -> wx_VaasaFI_*, wx_OlkiluotoFI_*, wx_PoriFI_*

ANTI-LEAKAGE: los precios SE1/EE se fijan en la misma subasta EUPHEMIA que FI, así
que en features.py (add_sweden_features) se usan SOLO laggeados (≥24h). Aquí solo
se almacenan crudos. La meteo es la PREVISIÓN pasada (no reanálisis), igual que el
resto del pipeline.

Uso:
    python src/extract/08_finland_coupling.py
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("ENTSOE_API_KEY")

MASTER_PATH = ROOT / "data" / "master" / "dataset_master.parquet"

# Periodo (alineado con 01_entsoe_prices.py y la meteo)
START = pd.Timestamp("2023-01-01", tz="UTC")
END   = pd.Timestamp("2026-06-01", tz="UTC")
OM_START, OM_END = "2023-01-01", "2026-05-31"

# Precios de acoplamiento que FALTAN (SE3/SE4 ya están en el master)
COUPLING_PRICES = {"SE1": "SE_1", "EE": "EE"}      # nombre_limpio -> código entsoe

# Fronteras FI para intercambios programados (ambos sentidos -> neto en features)
EXCHANGE_BORDERS = [("FI", "EE"), ("FI", "SE_3"), ("FI", "SE_1")]

# Meteo finlandesa adicional (Vaasa = gran hub eólico onshore de la costa oeste)
OM_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_VARS = ["temperature_2m", "precipitation", "wind_speed_10m",
           "wind_speed_100m", "wind_direction_100m", "shortwave_radiation"]
FI_WEATHER = {
    "VaasaFI":      {"lat": 63.096, "lon": 21.616, "desc": "Vaasa — hub eólico onshore"},
    "OlkiluotoFI":  {"lat": 61.233, "lon": 21.483, "desc": "Olkiluoto — nuclear + eólica costera"},
    "PoriFI":       {"lat": 61.483, "lon": 21.783, "desc": "Pori — eólica oeste"},
}

REQUEST_DELAY = 2.0
HTTP_TIMEOUT = 180


def _to_hourly_utc(s: pd.Series) -> pd.Series:
    """tz→UTC + agregación horaria (media) si la serie es 15-min."""
    s = s[~s.index.isna()]
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    if s.index.floor("h").has_duplicates:
        s = s.resample("h").mean()
    return s


def fetch_prices(client, logger) -> dict:
    out = {}
    for clean, code in COUPLING_PRICES.items():
        try:
            logger.info(f"  Precio DA {clean} ({code})...")
            p = client.query_day_ahead_prices(code, start=START, end=END)
            out[f"price_{clean}"] = _to_hourly_utc(p)
            logger.info(f"    OK {len(out[f'price_{clean}']):,} h | media {float(out[f'price_{clean}'].mean()):.1f} €/MWh")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"    ERROR {clean}: {e}")
    return out


def fetch_exchanges(client, logger) -> dict:
    out = {}
    for a, b in EXCHANGE_BORDERS:
        for frm, to in [(a, b), (b, a)]:
            col = f"sched_{frm}{to}_mw"
            try:
                logger.info(f"  Intercambio programado {frm}→{to}...")
                s = client.query_scheduled_exchanges(frm, to, start=START, end=END)
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                out[col] = _to_hourly_utc(s.astype(float))
                logger.info(f"    OK {col}: {len(out[col]):,} h")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.warning(f"    WARN {col} no disponible: {str(e)[:80]}")
    return out


def fetch_weather(logger) -> dict:
    out = {}
    for name, meta in FI_WEATHER.items():
        try:
            logger.info(f"  Meteo {name} ({meta['desc']})...")
            params = {"latitude": meta["lat"], "longitude": meta["lon"],
                      "start_date": OM_START, "end_date": OM_END,
                      "hourly": ",".join(OM_VARS), "timezone": "UTC"}
            r = requests.get(OM_BASE, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            hourly = r.json().get("hourly")
            if not hourly or "time" not in hourly:
                logger.error(f"    ERROR {name}: respuesta sin datos horarios")
                continue
            df = pd.DataFrame(hourly)
            idx = pd.to_datetime(df.pop("time"), utc=True)
            df.index = idx
            df = df.rename(columns={"shortwave_radiation": "solar_radiation"})
            for var in df.columns:
                out[f"wx_{name}_{var}"] = df[var]
            logger.info(f"    OK {name}: {len(df):,} h × {df.shape[1]} vars")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"    ERROR {name}: {e}")
    return out


def main():
    logger = get_logger("finland_coupling")
    if not API_KEY or API_KEY == "PENDIENTE":
        raise ValueError("ENTSOE_API_KEY no configurada en .env")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"No existe {MASTER_PATH} — corre la integración primero")

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=API_KEY)

    logger.info(f"=== Finland coupling | {START.date()} → {END.date()} ===")
    new_cols = {}
    new_cols.update(fetch_prices(client, logger))
    new_cols.update(fetch_exchanges(client, logger))
    new_cols.update(fetch_weather(logger))

    if not new_cols:
        logger.error("No se obtuvo ninguna columna nueva — abortando sin tocar el master")
        return

    master = pd.read_parquet(MASTER_PATH)
    # Master en CET (tz-aware) -> reindexar nuevas series (UTC) a su índice
    master_utc_index = master.index.tz_convert("UTC")
    added = []
    for col, series in new_cols.items():
        series = series[~series.index.duplicated(keep="first")].sort_index()
        master[col] = series.reindex(master_utc_index).values
        cov = master[col].notna().mean() * 100
        added.append((col, round(cov, 1)))

    master.to_parquet(MASTER_PATH)
    logger.info(f"\nOK Master enriquecido: +{len(added)} columnas FI-coupling")
    for col, cov in added:
        flag = "" if cov >= 90 else "  WARN cobertura baja"
        logger.info(f"    {col:28s} cobertura {cov:5.1f}%{flag}")

    save_run_metadata(
        script_name="finland_coupling",
        params={"coupling_prices": list(COUPLING_PRICES),
                "exchange_borders": [f"{a}-{b}" for a, b in EXCHANGE_BORDERS],
                "weather_sites": list(FI_WEATHER)},
        stats={col: cov for col, cov in added},
    )


if __name__ == "__main__":
    main()
