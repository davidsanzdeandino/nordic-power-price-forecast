"""
09_fingrid_fi.py — Físicos finlandeses de alta resolución (Fingrid Open Data)
=============================================================================
Fingrid (TSO de Finlandia) publica sus PROPIAS previsiones day-ahead del sistema
finlandés, potencialmente más precisas que las de ENTSO-E. Se añaden de forma
ADITIVA al master (DK y el resto intactos) y luego se mide si de verdad mejoran
el MAE de FI — si no aportan, se documentan y se quitan (sin inflar).

Datasets (PREVISIÓN para el día finlandés siguiente, 00-24 hora finlandesa; la
165, 246 y 247 se publican una vez al día a las 12:00 hora finlandesa, 11:00 CET,
y la 372 a las 10:00, con actualización desde las 11:00). Se guardan por su
startTime en UTC. La última hora CET de cada día de entrega (00-01 hora
finlandesa del día siguiente) se publica después de la subasta: features.py la
sustituye por la hora anterior (features._fingrid_known_at_auction).
  165 → fingrid_FI_load_fc_mw   (consumo/carga, prev. diaria)
  246 → fingrid_FI_wind_fc_mw   (eólica, prev. diaria)
  247 → fingrid_FI_solar_fc_mw  (solar, prev. diaria)
  372 → fingrid_FI_ol3_fc_mw    (Olkiluoto-3 protection scheme forecast, horario)

API: https://data.fingrid.fi/api/datasets/{id}/data  (header x-api-key).
Requiere FINGRID_API_KEY en .env (suscripción gratuita "Open Data starter").

Uso:  python src/extract/09_fingrid_fi.py
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
    165: "fingrid_FI_load_fc_mw",
    246: "fingrid_FI_wind_fc_mw",
    247: "fingrid_FI_solar_fc_mw",
    372: "fingrid_FI_ol3_fc_mw",
}

BASE = "https://data.fingrid.fi/api/datasets"
PAGE_SIZE = 10000   # máximo real del API de Fingrid
WINDOW_DAYS = 30    # trozo temporal: 30d × 5-min ≈ 8.640 ptos < PAGE_SIZE → una página
THROTTLE = 0.8      # s entre peticiones (respeta el rate-limit de Fingrid)
MAX_RETRY = 8


def _get(url, logger):
    """GET con reintento exponencial ante 429 (rate-limit de Fingrid)."""
    backoff = 6
    for _ in range(MAX_RETRY):
        r = requests.get(url, headers={"x-api-key": API_KEY}, timeout=40)
        if r.status_code == 429:
            time.sleep(backoff); backoff = min(backoff + 4, 30); continue
        return r
    return r


def fetch_dataset(did, logger) -> pd.Series:
    # ROBUSTEZ: la paginación del API de Fingrid es inconsistente (capa a 10k y el
    # puntero nextPage no es fiable). En su lugar troceamos el rango en ventanas de
    # 30 días: a resolución 5-min (la más densa, 2023) son ~8.640 puntos < 10k, así
    # que cada ventana cabe en UNA sola petición sin paginar. Determinista y completo.
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
        if r.status_code != 200:
            logger.warning(f"    dataset {did} {win_start:%Y-%m-%d}: HTTP {r.status_code}")
        else:
            j = r.json()
            batch = j.get("data", [])
            rows += batch
            total = (j.get("pagination") or {}).get("total", len(batch))
            if total > PAGE_SIZE:   # la ventana excedió una página (no debería con 30d)
                logger.warning(f"    WARN ventana {win_start:%Y-%m-%d} con {total} ptos > {PAGE_SIZE}")
        win_start = win_end
        time.sleep(THROTTLE)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(x["startTime"]): x["value"] for x in rows},
        dtype=float,
    ).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    # 15-min → horario (media). No-op si ya es horario (372).
    if s.index.floor("h").has_duplicates:
        s = s.resample("h").mean()
    return s


def main():
    logger = get_logger("fingrid_fi")
    if not API_KEY:
        raise ValueError("FINGRID_API_KEY no configurada en .env "
                         "(suscríbete gratis en data.fingrid.fi → 'Open Data starter').")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"No existe {MASTER_PATH} — corre la integración primero")

    logger.info(f"=== Fingrid FI | {START[:10]} → {END[:10]} ===")
    series = {}
    for did, col in DATASETS.items():
        logger.info(f"  [{did}] {col}...")
        s = fetch_dataset(did, logger)
        if len(s):
            cov_from = s.index.min()
            logger.info(f"    OK {len(s):,} h | desde {cov_from:%Y-%m-%d} | "
                        f"media {float(s.mean()):.0f} MW")
            series[col] = s
        else:
            logger.warning(f"    WARN {col}: sin datos")
        time.sleep(THROTTLE)

    if not series:
        logger.error("No se obtuvo ninguna serie Fingrid — abortando sin tocar el master")
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
    logger.info(f"\nOK Master enriquecido con Fingrid: +{len(added)} columnas")
    for col, cov in added:
        flag = "" if cov >= 80 else "  WARN cobertura parcial (dataset reciente)"
        logger.info(f"    {col:28s} cobertura {cov:5.1f}%{flag}")

    save_run_metadata(
        script_name="fingrid_fi",
        params={"datasets": DATASETS, "start": START, "end": END},
        stats={col: cov for col, cov in added},
    )


if __name__ == "__main__":
    main()
