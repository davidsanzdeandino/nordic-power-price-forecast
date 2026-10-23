"""
10_fingrid_balancing.py — Mercados de balancing y reservas FI (Fingrid Open Data)
=================================================================================
Además del arbitraje day-ahead, una batería nórdica obtiene ingresos en los
mercados de **reservas** (FCR-N, FCR-D, aFRR, mFRR) y en el de
**balancing/imbalance**. Con estos datos el proyecto pasa de un forecaster de un
solo mercado a un análisis de **flexibilidad multi-mercado** (arbitraje + reservas).

Trae (gratis, misma API key de Fingrid) y los UNE ADITIVAMENTE al master:
  Precios de CAPACIDAD de reserva (procurados ~D-1 → señal de estrés del sistema):
    317 → fingrid_FI_fcrn_price        (FCR-N, mercado horario)
    318 → fingrid_FI_fcrd_up_price     (FCR-D up)
    283 → fingrid_FI_fcrd_down_price   (FCR-D down)
    329 → fingrid_FI_mfrr_up_price     (mFRR capacity, up)
     52 → fingrid_FI_afrr_up_price     (aFRR price, up)
  Precios REALIZADOS ex-post (solo uso laggeado ≥24h como features de estrés):
    319 → fingrid_FI_imbalance_price
    244 → fingrid_FI_balup_price       (up-regulation, balancing energy)
    106 → fingrid_FI_baldown_price     (down-regulation, balancing energy)

ANTI-LEAKAGE: en features.py estas series se usan SOLO laggeadas (≥24h). Para el
cálculo de ingreso por reservas en el Value Calculator se usan los precios
REALIZADOS del periodo de test (análisis económico ex-post, igual que el arbitraje
de batería liquida a precio real) — no como input del forecast.

Uso:  python src/extract/10_fingrid_balancing.py
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
API_KEY = os.getenv("FINGRID_API_KEY")
MASTER_PATH = ROOT / "data" / "master" / "dataset_master.parquet"

START = "2023-01-01T00:00:00Z"
END   = "2026-06-01T00:00:00Z"

DATASETS = {
    317: "fingrid_FI_fcrn_price",
    318: "fingrid_FI_fcrd_up_price",
    283: "fingrid_FI_fcrd_down_price",
    329: "fingrid_FI_mfrr_up_price",
    52:  "fingrid_FI_afrr_up_price",
    319: "fingrid_FI_imbalance_price",
    244: "fingrid_FI_balup_price",
    106: "fingrid_FI_baldown_price",
}

BASE = "https://data.fingrid.fi/api/datasets"
PAGE_SIZE = 10000
WINDOW_DAYS = 30
THROTTLE = 0.8
MAX_RETRY = 8


def _get(url, logger):
    backoff = 6
    for _ in range(MAX_RETRY):
        r = requests.get(url, headers={"x-api-key": API_KEY}, timeout=40)
        if r.status_code == 429:
            time.sleep(backoff); backoff = min(backoff + 4, 30); continue
        return r
    return r


def fetch_dataset(did, logger) -> pd.Series:
    """Fetch por ventanas de 30 días (paginación de Fingrid poco fiable)."""
    rows = []
    win_start = pd.Timestamp(START)
    end_ts = pd.Timestamp(END)
    while win_start < end_ts:
        win_end = min(win_start + pd.Timedelta(days=WINDOW_DAYS), end_ts)
        url = (f"{BASE}/{did}/data"
               f"?startTime={win_start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
               f"&endTime={win_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
               f"&pageSize={PAGE_SIZE}&format=json")
        r = _get(url, logger)
        if r.status_code == 200:
            rows += r.json().get("data", [])
        else:
            logger.warning(f"    dataset {did} {win_start:%Y-%m-%d}: HTTP {r.status_code}")
        win_start = win_end
        time.sleep(THROTTLE)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(x["startTime"]): x["value"] for x in rows},
                  dtype=float).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if s.index.floor("h").has_duplicates:    # sub-horario → horario (media)
        s = s.resample("h").mean()
    return s


def main():
    logger = get_logger("fingrid_balancing")
    if not API_KEY:
        raise ValueError("FINGRID_API_KEY no configurada en .env")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"No existe {MASTER_PATH} — corre la integración primero")

    logger.info(f"=== Fingrid balancing/reservas FI | {START[:10]} → {END[:10]} ===")
    series = {}
    for did, col in DATASETS.items():
        logger.info(f"  [{did}] {col}...")
        s = fetch_dataset(did, logger)
        if len(s):
            logger.info(f"    OK {len(s):,} h | desde {s.index.min():%Y-%m-%d} | "
                        f"media {float(s.mean()):.1f}")
            series[col] = s
        else:
            logger.warning(f"    WARN {col}: sin datos")

    if not series:
        logger.error("Sin series de balancing — abortando sin tocar el master")
        return

    master = pd.read_parquet(MASTER_PATH)
    master_utc = master.index.tz_convert("UTC")
    added = []
    for col, s in series.items():
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")
        master[col] = s.reindex(master_utc).values
        added.append((col, round(master[col].notna().mean() * 100, 1)))

    master.to_parquet(MASTER_PATH)
    logger.info(f"\nOK Master enriquecido con balancing/reservas: +{len(added)} columnas")
    for col, cov in added:
        flag = "" if cov >= 80 else "  WARN cobertura parcial"
        logger.info(f"    {col:30s} cobertura {cov:5.1f}%{flag}")

    save_run_metadata(
        script_name="fingrid_balancing",
        params={"datasets": DATASETS, "start": START, "end": END},
        stats={col: cov for col, cov in added},
    )


if __name__ == "__main__":
    main()
