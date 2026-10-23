"""
11_energinet_reserves.py — Mercados de reservas DK (Energi Data Service)
========================================================================
El equivalente danés de 10_fingrid_balancing.py: precios de los mercados de
reserva/balancing en los que participa una batería danesa. Crucial: DK1 y DK2
son sistemas DISTINTOS — DK1 está en el área síncrona continental (compra FCR
en la cooperación europea de regelleistung.net, junto a Alemania) y DK2 en la
nórdica (compra FCR-N/FCR-D como Finlandia/Suecia).

Datasets (api.energidataservice.dk, sin clave, CC-BY 4.0 — verificados 10-11
jun 2026 en la documentación de Energi Data Service):
  FcrNdDK2            → FCR-N y FCR-D up/down de DK2 (subastas D-1/D-2 nórdicas;
                        formato largo: ProductName × AuctionType). Usamos 'Total'.
  AfrrReservesNordic  → capacidad aFRR up/down por área (DK1, DK2 y FI — la
                        columna FI sirve de validación cruzada del dataset 52
                        de Fingrid que ya está en el master).
  mFRRCapacityMarket  → capacidad mFRR up/down DK1/DK2 (mercado nórdico desde
                        jun-2023 en DK).
  RegulatingBalancePowerdata → precios de energía de balancing (mFRR up/down)
                        e imbalance por hora — histórico hasta el cambio a
                        15 min (mar-2025).
  ImbalancePrice      → precio de desbalance 15-min (desde mar-2025) → se
                        agrega a horario (media).

ANTI-LEAKAGE: igual que las series Fingrid — como features solo laggeadas
≥24h; para el Battery Lab se usan como liquidación ex-post realizada.

NOTA DK1-FCR: el precio FCR de DK1 (cooperación alemana) NO está en EDS con
histórico utilizable (FcrReservesDK1 termina ~2021; FcrDK1 actual empieza
dic-2024). El FCR histórico DK1 vive en regelleistung.net (CSV) — pendiente
en roadmap; el Battery Lab DK cubre DK2 (nórdico) con stack completo.

Uso:  python src/extract/11_energinet_reserves.py
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PATH = ROOT / "data" / "master" / "dataset_master.parquet"

BASE_URL = "https://api.energidataservice.dk/dataset"
START = "2023-01-01T00:00"
END   = "2026-06-01T00:00"
HTTP_TIMEOUT = 60
THROTTLE = 2.0          # EDS rate-limita agresivo (429 con 'Try again in Ns')
MAX_RETRY = 6


def _get(url: str, params: dict, logger) -> dict | None:
    """GET con reintento ante 429 respetando el tiempo que pide la API."""
    for attempt in range(MAX_RETRY):
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            # El mensaje incluye 'Try again in N seconds'
            wait = 60
            try:
                import re
                m = re.search(r"(\d+) second", r.text)
                if m:
                    wait = min(int(m.group(1)) + 5, 180)
            except Exception:
                pass
            logger.warning(f"    429 rate-limit — esperando {wait}s (intento {attempt+1})")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            logger.warning(f"    HTTP {r.status_code}: {r.text[:200]}")
            return None
        return r.json()
    return None


def fetch_dataset(name: str, logger, extra_params: dict | None = None) -> pd.DataFrame:
    """Descarga completa por bloques de 6 meses (datasets de reservas son ligeros)."""
    frames = []
    cur = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=6), end)
        params = {"start": cur.strftime("%Y-%m-%dT%H:%M"),
                  "end": nxt.strftime("%Y-%m-%dT%H:%M"), "limit": 0}
        if extra_params:
            params.update(extra_params)
        j = _get(f"{BASE_URL}/{name}", params, logger)
        if j and j.get("records"):
            frames.append(pd.DataFrame(j["records"]))
        cur = nxt
        time.sleep(THROTTLE)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def _to_hourly_series(df: pd.DataFrame, time_col: str, value_col: str) -> pd.Series:
    s = pd.Series(df[value_col].values,
                  index=pd.to_datetime(df[time_col]).dt.tz_localize("UTC"))
    s = s[~s.index.duplicated(keep="first")].sort_index()
    if s.index.floor("h").has_duplicates:      # 15-min → horario (media)
        s = s.resample("h").mean()
    return s.astype(float)


def main():
    logger = get_logger("energinet_reserves")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"No existe {MASTER_PATH} — corre la integración primero")
    logger.info(f"=== Reservas DK (Energi Data Service) | {START[:10]} → {END[:10]} ===")

    series: dict[str, pd.Series] = {}

    # ── 1) FCR-N / FCR-D de DK2 (mercado nórdico) ───────────────────────────
    logger.info("  [FcrNdDK2] FCR-N + FCR-D up/down DK2...")
    fcr = fetch_dataset("FcrNdDK2", logger)
    if not fcr.empty:
        fcr.to_parquet(OUTPUT_DIR / "energinet_FcrNdDK2.parquet")
        tot = fcr[fcr["AuctionType"] == "Total"]
        product_map = {
            "FCR-N": "energinet_DK2_fcrn_price",
            "FCR-D upp": "energinet_DK2_fcrd_up_price",
            "FCR-D ned": "energinet_DK2_fcrd_down_price",
        }
        for product, col in product_map.items():
            sub = tot[tot["ProductName"] == product]
            if len(sub):
                series[col] = _to_hourly_series(sub, "HourUTC", "PriceTotalEUR")
                logger.info(f"    OK {col}: {len(series[col]):,} h | media {series[col].mean():.1f} €/MW,h")
            else:
                logger.warning(f"    WARN {product}: sin filas (¿histórico más corto?)")
    else:
        logger.warning("    WARN FcrNdDK2 vacío")

    # ── 2) aFRR capacidad (DK1, DK2 + FI como validación cruzada) ───────────
    logger.info("  [AfrrReservesNordic] aFRR capacidad por área...")
    afrr = fetch_dataset("AfrrReservesNordic", logger)
    if not afrr.empty:
        afrr.to_parquet(OUTPUT_DIR / "energinet_AfrrReservesNordic.parquet")
        for area in ("DK1", "DK2", "FI"):
            sub = afrr[afrr["PriceArea"] == area]
            if not len(sub):
                continue
            for side, col_v in (("Up", "UpPriceEUR"), ("Down", "DownPriceEUR")):
                col = f"energinet_{area}_afrr_{side.lower()}_price"
                series[col] = _to_hourly_series(sub, "TimeUTC", col_v)
                logger.info(f"    OK {col}: {len(series[col]):,} h | media {series[col].mean():.1f} €/MW,h")
    else:
        logger.warning("    WARN AfrrReservesNordic vacío")

    # ── 3) mFRR capacidad DK1/DK2 ────────────────────────────────────────────
    logger.info("  [mFRRCapacityMarket] mFRR capacidad DK1/DK2...")
    mfrr = fetch_dataset("mFRRCapacityMarket", logger)
    if not mfrr.empty:
        mfrr.to_parquet(OUTPUT_DIR / "energinet_mFRRCapacityMarket.parquet")
        for area in ("DK1", "DK2"):
            sub = mfrr[mfrr["PriceArea"] == area]
            if not len(sub):
                continue
            for side, col_v in (("Up", "UpPriceEUR"), ("Down", "DownPriceEUR")):
                col = f"energinet_{area}_mfrr_{side.lower()}_price"
                series[col] = _to_hourly_series(sub, "TimeUTC", col_v)
                logger.info(f"    OK {col}: {len(series[col]):,} h | media {series[col].mean():.1f} €/MW,h")
    else:
        logger.warning("    WARN mFRRCapacityMarket vacío")

    # ── 4) Energía de balancing + imbalance (histórico horario) ─────────────
    logger.info("  [RegulatingBalancePowerdata] energía balancing + imbalance DK...")
    reg = fetch_dataset("RegulatingBalancePowerdata", logger)
    if not reg.empty:
        reg.to_parquet(OUTPUT_DIR / "energinet_RegulatingBalancePowerdata.parquet")
        for area in ("DK1", "DK2"):
            sub = reg[reg["PriceArea"] == area]
            if not len(sub):
                continue
            for col, col_v in ((f"energinet_{area}_balup_price", "BalancingPowerPriceUpEUR"),
                               (f"energinet_{area}_baldown_price", "BalancingPowerPriceDownEUR"),
                               (f"energinet_{area}_imbalance_price", "ImbalancePriceEUR")):
                if col_v in sub.columns and sub[col_v].notna().any():
                    series[col] = _to_hourly_series(sub.dropna(subset=[col_v]), "HourUTC", col_v)
                    logger.info(f"    OK {col}: {len(series[col]):,} h")
    else:
        logger.warning("    WARN RegulatingBalancePowerdata vacío")

    # ── 5) ImbalancePrice 15-min (desde mar-2025) → completa el imbalance ───
    logger.info("  [ImbalancePrice] imbalance 15-min (2025+)...")
    imb = fetch_dataset("ImbalancePrice", logger)
    if not imb.empty:
        imb.to_parquet(OUTPUT_DIR / "energinet_ImbalancePrice.parquet")
        tcol = "TimeUTC" if "TimeUTC" in imb.columns else "Minutes5UTC"
        vcol = "ImbalancePriceEUR" if "ImbalancePriceEUR" in imb.columns else None
        if vcol:
            for area in ("DK1", "DK2"):
                sub = imb[imb["PriceArea"] == area].dropna(subset=[vcol])
                if not len(sub):
                    continue
                s_new = _to_hourly_series(sub, tcol, vcol)
                col = f"energinet_{area}_imbalance_price"
                if col in series:   # une histórico horario + 15-min agregado
                    series[col] = pd.concat([series[col][~series[col].index.isin(s_new.index)], s_new]).sort_index()
                else:
                    series[col] = s_new
                logger.info(f"    OK {col} (combinado): {len(series[col]):,} h")
    else:
        logger.warning("    WARN ImbalancePrice vacío")

    if not series:
        logger.error("Sin series de reservas DK — abortando sin tocar el master")
        return

    # ── Unión aditiva al master ──────────────────────────────────────────────
    master = pd.read_parquet(MASTER_PATH)
    master_utc = master.index.tz_convert("UTC")
    added = []
    for col, s in series.items():
        master[col] = s.reindex(master_utc).values
        added.append((col, round(master[col].notna().mean() * 100, 1)))
    master.to_parquet(MASTER_PATH)

    logger.info(f"\nOK Master enriquecido con reservas DK: +{len(added)} columnas")
    for col, cov in added:
        flag = "" if cov >= 60 else "  WARN cobertura parcial (mercado/dataset más reciente)"
        logger.info(f"    {col:38s} cobertura {cov:5.1f}%{flag}")

    save_run_metadata(
        script_name="energinet_reserves",
        params={"datasets": ["FcrNdDK2", "AfrrReservesNordic", "mFRRCapacityMarket",
                             "RegulatingBalancePowerdata", "ImbalancePrice"],
                "start": START, "end": END},
        stats={col: cov for col, cov in added},
    )


if __name__ == "__main__":
    main()
