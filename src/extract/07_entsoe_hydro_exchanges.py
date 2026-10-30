"""
07_entsoe_hydro_exchanges.py
============================
Extrae de ENTSO-E dos señales adicionales de alto valor para el precio nórdico:

  A. Nivel de embalses hidroeléctricos (water reservoirs & hydro storage)
     — semanal, por zona. Es EL driver estructural del precio nórdico: el
     sistema NO/SE es hidro-dependiente y el nivel de embalses fija el coste
     de oportunidad del agua (cuanto más bajos, más caro el MWh hidro).

  B. Intercambios comerciales PROGRAMADOS day-ahead (scheduled exchanges)
     — horario, por frontera. Es el programa nominado en la subasta, distinto
     de los flujos físicos reales.

ANTI-LEAKAGE (importante):
  - Embalses: dato SEMANAL publicado con retraso (el valor de la semana W lleva
    la marca de inicio de W, lunes 00:00 CET, y se publica como tarde el tercer
    día laborable después de W; Reglamento 543/2013, art. 16.2(d)). Se guarda
    tal cual (índice semanal). La integración lo reindexa a horario con
    forward-fill y features.py aplica un LAG de 13 días (312 h,
    features.RESERVOIR_LAG), de modo que en la subasta del día D solo se use un
    nivel ya PUBLICADO. Ver features.add_hydro_features.
  - Intercambios programados: aunque son "day-ahead", se determinan en el mismo
    clearing que el precio que queremos predecir. Para evitar circularidad se
    usan con LAG de 24h en features (el programa de ayer). Aquí solo se descargan.

Fuente: ENTSO-E Transparency (token gratuito). Misma cuenta que 01/02/05.

Uso:
    python src/extract/07_entsoe_hydro_exchanges.py
    # O desde el orquestador: python run_pipeline.py --step 7
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("ENTSOE_API_KEY")

OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2023-01-01", tz="UTC")
END   = pd.Timestamp("2026-06-01", tz="UTC")   # Exclusivo — cubre hasta may-2026

# Zonas con hidro relevante para el precio de DK1/DK2 (Noruega, Suecia, Finlandia).
# DK1/DK2 apenas tienen hidro → se omiten.
RESERVOIR_ZONES = ["NO_2", "SE_3", "SE_4", "FI"]

# Fronteras para intercambios programados day-ahead (las que mueven el precio DK).
# Ambas direcciones para poder calcular el neto.
SCHED_BORDERS = [
    ("DK_1", "NO_2",  "Skagerrak (hidro NO)"),
    ("NO_2", "DK_1",  "Skagerrak (inverso)"),
    ("DK_1", "DE_LU", "DK1 ↔ Alemania"),
    ("DE_LU", "DK_1", "Alemania → DK1"),
    ("DK_2", "SE_4",  "DK2 ↔ Suecia SE4"),
    ("SE_4", "DK_2",  "SE4 → DK2"),
    ("DK_2", "DE_LU", "DK2 ↔ Alemania"),
    ("DE_LU", "DK_2", "Alemania → DK2"),
    ("DK_1", "DK_2",  "Great Belt interno"),
    ("DK_2", "DK_1",  "Great Belt (inverso)"),
]

REQUEST_DELAY = 2.0   # Segundos entre llamadas (rate limit API gratuita)


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el índice a UTC explícito. Aplana MultiIndex si existe."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip("_")
                      for col in df.columns.values]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp_utc"
    # ROBUSTEZ 15-min (ENTSO-E 2025+): agregar a horario media. No-op si horario
    # o más grueso (embalses semanales NO se tocan).
    if df.index.floor("h").has_duplicates:   # resolución mixta/15-min
            df = df.select_dtypes("number").resample("h").mean()
            df.index.name = "timestamp_utc"
    return df


def save_parquet(df: pd.DataFrame, filename: str, logger) -> int:
    path = OUTPUT_DIR / filename
    df.to_parquet(path)
    logger.info(f"  OK Guardado: {filename} ({len(df)} filas)")
    return len(df)


# ── A. Embalses hidroeléctricos ────────────────────────────────────────────────

def download_reservoirs(client, logger, stats: dict):
    """A. Nivel de embalses (semanal) por zona hidro."""
    logger.info("\n--- A. Nivel de embalses hidroeléctricos (semanal) ---")
    for zone in RESERVOIR_ZONES:
        key = f"reservoir_{zone}"
        try:
            logger.info(f"  Embalses {zone}...")
            df = client.query_aggregate_water_reservoirs_and_hydro_storage(
                zone, start=START, end=END
            )
            if isinstance(df, pd.Series):
                df = df.to_frame(name="reservoir_mwh")
            df = to_utc(df)
            # Renombrar la primera columna a algo estable
            df.columns = [f"reservoir_{zone}_stored_mwh"] + list(df.columns[1:])
            stats[key] = save_parquet(df, f"entsoe_reservoir_{zone}.parquet", logger)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"  ERROR Embalses {zone}: {e}")
            stats[key] = "ERROR"


# ── B. Intercambios comerciales programados ────────────────────────────────────

def download_scheduled_exchanges(client, logger, stats: dict):
    """B. Intercambios comerciales programados day-ahead por frontera."""
    logger.info("\n--- B. Intercambios comerciales programados (day-ahead) ---")
    for origin, dest, desc in SCHED_BORDERS:
        key = f"sched_{origin}_{dest}"
        try:
            logger.info(f"  Programado {origin} → {dest} ({desc})...")
            s = client.query_scheduled_exchanges(
                origin, dest, start=START, end=END, dayahead=True
            )
            if isinstance(s, pd.Series):
                df = s.to_frame(name="sched_mw")
            else:
                df = s
            df = to_utc(df)
            fname = f"entsoe_sched_{origin}_{dest}.parquet".replace("/", "-")
            stats[key] = save_parquet(df, fname, logger)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"  ERROR Programado {origin}→{dest}: {e}")
            stats[key] = "ERROR"


# ── C. Alemania (DE/LU) — mercado acoplado dominante ───────────────────────────

def download_germany(client, logger, stats: dict):
    """
    C. Datos de Alemania (DE_LU): precio day-ahead + forecast renovable + carga.

    DECISIÓN TÉCNICA: el mercado alemán es, según la literatura EPF, uno de los
    drivers MÁS influyentes del precio nórdico (acoplamiento de mercados vía los
    interconectores DK1↔DE y DK2↔DE). Se descargan:
      - Precio DA alemán → se usará LAGGEADO 24h en features (su precio del día D
        se fija en la misma subasta acoplada que el danés → usar mismo timestamp
        sería leakage).
      - Forecast eólico+solar y de carga alemanes → SÍ disponibles antes de la
        subasta (son previsiones), se usan sin lag.
    """
    logger.info("\n--- C. Alemania (DE/LU): precio DA + renovables + carga ---")
    # C1. Precio day-ahead
    try:
        logger.info("  Precio DA Alemania...")
        s = client.query_day_ahead_prices("DE_LU", start=START, end=END)
        df = s.to_frame(name="de_price_eur") if isinstance(s, pd.Series) else s
        df = to_utc(df)
        stats["de_price"] = save_parquet(df, "entsoe_de_price.parquet", logger)
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        logger.error(f"  ERROR Precio DA DE: {e}"); stats["de_price"] = "ERROR"
    # C2. Forecast eólico + solar
    try:
        logger.info("  Forecast eólico/solar Alemania...")
        df = to_utc(client.query_wind_and_solar_forecast("DE_LU", start=START, end=END))
        stats["de_forecast"] = save_parquet(df, "entsoe_de_forecast.parquet", logger)
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        logger.error(f"  ERROR Forecast DE: {e}"); stats["de_forecast"] = "ERROR"
    # C3. Forecast de carga
    try:
        logger.info("  Forecast carga Alemania...")
        df = client.query_load_forecast("DE_LU", start=START, end=END)
        if isinstance(df, pd.Series): df = df.to_frame()
        df = to_utc(df)
        # Normalizar a un único nombre estable
        df = df.iloc[:, [0]]
        df.columns = ["de_load_forecast_mw"]
        stats["de_load_fc"] = save_parquet(df, "entsoe_de_load_fc.parquet", logger)
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        logger.error(f"  ERROR Load forecast DE: {e}"); stats["de_load_fc"] = "ERROR"


# ── Función principal ─────────────────────────────────────────────────────────

def extract_hydro_exchanges() -> dict:
    """
    Descarga embalses hidro e intercambios programados de ENTSO-E.

    Returns:
        Dict con stats {clave: n_filas | 'ERROR'}
    """
    logger = get_logger("entsoe_hydro_exchanges")

    if not API_KEY or API_KEY in ("PENDIENTE", "tu_token_aqui", "your_entsoe_token_here"):
        logger.error(
            "No se encontró un ENTSOE_API_KEY válido en .env\n"
            "  → Copia .env.example a .env y añade tu token gratuito."
        )
        raise ValueError("ENTSOE_API_KEY no configurada.")

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        logger.error("Módulo 'entsoe-py' no instalado. Ejecuta: pip install entsoe-py")
        raise

    client = EntsoePandasClient(api_key=API_KEY)
    stats  = {}

    logger.info(f"=== Embalses + Intercambios programados ENTSO-E | "
                f"{START.date()} → {END.date()} ===")
    logger.info(f"Zonas embalses: {RESERVOIR_ZONES}")
    logger.info(f"Fronteras programadas: {len(SCHED_BORDERS)}")

    download_reservoirs(client, logger, stats)
    download_scheduled_exchanges(client, logger, stats)
    download_germany(client, logger, stats)

    save_run_metadata(
        script_name="entsoe_hydro_exchanges",
        params={
            "start": str(START), "end": str(END),
            "reservoir_zones": RESERVOIR_ZONES,
            "sched_borders": [(o, d) for o, d, _ in SCHED_BORDERS],
        },
        stats=stats,
    )

    ok  = sum(1 for v in stats.values() if v != "ERROR")
    err = sum(1 for v in stats.values() if v == "ERROR")
    logger.info(f"\nResumen: {ok} descargas OK | {err} errores de {len(stats)} total")
    if err > 0:
        logger.warning(f"  Fallidos: {[k for k, v in stats.items() if v == 'ERROR']}")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_hydro_exchanges()
