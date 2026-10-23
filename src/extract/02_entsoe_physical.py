"""
02_entsoe_physical.py
=====================
Extrae datos físicos del sistema eléctrico desde ENTSO-E:

  A. Generación real por tipo de fuente (mix energético por zona)
  B. Previsión de viento y solar (day-ahead forecast — input del modelo)
  C. Demanda total real (load)
  D. Flujos físicos en interconexiones clave
  E. NTC - Net Transfer Capacity (capacidad disponible en fronteras)

DECISIÓN TÉCNICA: Se descarga tanto generación real como previsión DA.
La previsión es la que usará el modelo (evita data leakage); la real
sirve para calcular el error de previsión y como feature de contexto.

Uso:
    python src/extract/02_entsoe_physical.py
    # O desde el orquestador: python run_pipeline.py --step 2
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

# Mismo periodo que precios (ver decisión técnica en 01_entsoe_prices.py)
START = pd.Timestamp("2023-01-01", tz="UTC")
END   = pd.Timestamp("2026-06-01", tz="UTC")   # Exclusivo — cubre hasta may-2026

# Zonas objetivo
TARGET_ZONES = ["DK_1", "DK_2", "NO_2", "SE_3", "SE_4", "FI"]

# DECISIÓN TÉCNICA: Fronteras críticas para Dinamarca.
# Se descargan flujos en ambas direcciones (origen→destino y viceversa)
# para tener el flujo neto real, ya que la API devuelve flujos unidireccionales.
BORDERS = [
    # (origen, destino, descripción)
    ("DK_1", "DK_2",  "Great Belt (cable submarino interno DK)"),
    ("DK_2", "DK_1",  "Great Belt (inverso)"),
    ("DK_1", "NO_2",  "Skagerrak (cable hidro → Noruega)"),
    ("NO_2", "DK_1",  "Skagerrak (inverso)"),
    ("DK_1", "DE_LU", "DK1 → Alemania (export solar/eólico)"),
    ("DE_LU", "DK_1", "Alemania → DK1 (inverso)"),
    ("DK_2", "SE_4",  "DK2 → Suecia"),
    ("SE_4", "DK_2",  "Suecia → DK2 (inverso)"),
    ("DK_2", "DE_LU", "DK2 → Alemania"),
    ("DE_LU", "DK_2", "Alemania → DK2 (inverso)"),
]

# Fronteras para NTC (mismas, sin descripción)
NTC_BORDERS = [(o, d) for o, d, _ in BORDERS]

REQUEST_DELAY = 2.0   # Segundos entre llamadas — API gratuita tiene rate limit


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el índice a UTC explícito. Aplana MultiIndex si existe."""
    # Aplanar MultiIndex de columnas (frecuente en query_generation)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip("_")
                      for col in df.columns.values]

    # Normalizar timezone del índice
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp_utc"

    # ROBUSTEZ 15-min: desde 2025 ENTSO-E publica precios/forecasts en 15 min.
    # Agregamos a HORARIO (media) ya en extracción para que cleaning/integración
    # sean homogéneos. No-op para series ya horarias.
    if df.index.floor("h").has_duplicates:   # resolución mixta/15-min
            df = df.select_dtypes("number").resample("h").mean()
            df.index.name = "timestamp_utc"
    return df


def save_parquet(df: pd.DataFrame, filename: str, logger) -> int:
    """Guarda un DataFrame como Parquet y devuelve número de filas."""
    path = OUTPUT_DIR / filename
    df.to_parquet(path)
    logger.info(f"  OK Guardado: {filename} ({len(df)} filas)")
    return len(df)


# ── Secciones de descarga ─────────────────────────────────────────────────────

def download_generation(client, logger: object, stats: dict):
    """A. Generación real por tipo de fuente (mix energético)."""
    logger.info("\n--- A. Generación real por zona ---")
    for zone in TARGET_ZONES:
        key = f"gen_real_{zone}"
        try:
            logger.info(f"  Descargando generación {zone}...")
            df = client.query_generation(zone, start=START, end=END)
            df = to_utc(df)
            stats[key] = save_parquet(df, f"entsoe_gen_{zone}.parquet", logger)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"  ERROR Gen {zone}: {e}")
            stats[key] = "ERROR"


def download_wind_solar_forecast(client, logger: object, stats: dict):
    """
    B. Previsión day-ahead de viento y solar.

    DECISIÓN TÉCNICA: Esta es la variable más importante para el modelo.
    El precio day-ahead se publica ~12h antes del día de entrega, cuando
    ya se conoce la previsión de renovables. Usar el forecast (no el real)
    es fundamental para no introducir data leakage en el modelo predictivo.
    """
    logger.info("\n--- B. Previsión DA viento y solar ---")
    for zone in TARGET_ZONES:
        key = f"forecast_wind_solar_{zone}"
        try:
            logger.info(f"  Descargando forecast {zone}...")
            df = client.query_wind_and_solar_forecast(zone, start=START, end=END)
            df = to_utc(df)
            stats[key] = save_parquet(df, f"entsoe_forecast_{zone}.parquet", logger)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"  ERROR Forecast {zone}: {e}")
            stats[key] = "ERROR"


def download_load(client, logger: object, stats: dict):
    """
    C. Demanda total real y previsión de demanda.

    La demanda es un driver clave del precio (junto con la generación).
    Se descargan tanto el dato real como la previsión day-ahead.
    """
    logger.info("\n--- C. Demanda (load) ---")

    for zone in TARGET_ZONES:
        # C1. Demanda real
        key = f"load_real_{zone}"
        try:
            logger.info(f"  Demanda real {zone}...")
            df = client.query_load(zone, start=START, end=END)
            df = to_utc(df)
            # query_load puede devolver Serie o DataFrame
            if isinstance(df, pd.Series):
                df = df.to_frame(name="load_mw")
            stats[key] = save_parquet(df, f"entsoe_load_{zone}.parquet", logger)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"  ERROR Load real {zone}: {e}")
            stats[key] = "ERROR"

        # C2. Previsión de demanda day-ahead
        key_fc = f"load_forecast_{zone}"
        try:
            logger.info(f"  Previsión demanda {zone}...")
            df_fc = client.query_load_forecast(zone, start=START, end=END)
            df_fc = to_utc(df_fc)
            if isinstance(df_fc, pd.Series):
                df_fc = df_fc.to_frame(name="load_forecast_mw")
            stats[key_fc] = save_parquet(
                df_fc, f"entsoe_load_forecast_{zone}.parquet", logger
            )
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"  ERROR Load forecast {zone}: {e}")
            stats[key_fc] = "ERROR"


def download_crossborder_flows(client, logger: object, stats: dict):
    """D. Flujos físicos en interconexiones clave."""
    logger.info("\n--- D. Flujos físicos en interconexiones ---")

    for origin, dest, desc in BORDERS:
        key = f"flow_{origin}_{dest}"
        try:
            logger.info(f"  Flujo {origin} → {dest} ({desc})...")
            flows = client.query_crossborder_flows(origin, dest, start=START, end=END)

            if isinstance(flows, pd.Series):
                df = flows.to_frame(name="flow_mw")
            else:
                df = flows

            df = to_utc(df)
            fname = f"entsoe_flow_{origin}_{dest}.parquet".replace("/", "-")
            stats[key] = save_parquet(df, fname, logger)
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.error(f"  ERROR Flujo {origin}→{dest}: {e}")
            stats[key] = "ERROR"


def download_ntc(client, logger: object, stats: dict):
    """
    E. Net Transfer Capacity (NTC) — capacidad disponible en fronteras.

    DECISIÓN TÉCNICA: El NTC determina cuánta electricidad puede fluir
    entre zonas. Cuando el NTC se satura, el precio diverge entre zonas
    (spread DK1-DK2). Es una variable explicativa clave del spread.
    """
    logger.info("\n--- E. NTC (Net Transfer Capacity) ---")

    for origin, dest in NTC_BORDERS:
        key = f"ntc_{origin}_{dest}"
        try:
            logger.info(f"  NTC {origin} → {dest}...")
            # query_net_transfer_capacity_dayahead devuelve capacidades DA
            ntc = client.query_net_transfer_capacity_dayahead(
                origin, dest, start=START, end=END
            )

            if isinstance(ntc, pd.Series):
                df = ntc.to_frame(name="ntc_mw")
            else:
                df = ntc

            df = to_utc(df)
            fname = f"entsoe_ntc_{origin}_{dest}.parquet".replace("/", "-")
            stats[key] = save_parquet(df, fname, logger)
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.error(f"  ERROR NTC {origin}→{dest}: {e}")
            stats[key] = "ERROR"


# ── Función principal ─────────────────────────────────────────────────────────

def extract_physical() -> dict:
    """
    Orquesta la descarga de todos los datos físicos de ENTSO-E.

    Returns:
        Dict con stats {clave: n_filas | 'ERROR'}
    """
    logger = get_logger("entsoe_physical")

    if not API_KEY or API_KEY == "PENDIENTE":
        logger.error(
            "No se encontró ENTSOE_API_KEY en el archivo .env\n"
            "  → Copia .env.example a .env y añade tu clave."
        )
        raise ValueError("ENTSOE_API_KEY no configurada.")

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        logger.error("Módulo 'entsoe-py' no instalado. Ejecuta: pip install entsoe-py")
        raise

    client = EntsoePandasClient(api_key=API_KEY)
    stats  = {}

    logger.info(f"=== Datos Físicos ENTSO-E | {START.date()} → {END.date()} ===")
    logger.info(f"Zonas: {TARGET_ZONES}")
    logger.info(f"Fronteras: {len(BORDERS)} interconexiones")

    download_generation(client, logger, stats)
    download_wind_solar_forecast(client, logger, stats)
    download_load(client, logger, stats)
    download_crossborder_flows(client, logger, stats)
    download_ntc(client, logger, stats)

    # Guardar metadatos
    save_run_metadata(
        script_name="entsoe_physical",
        params={
            "start": str(START),
            "end":   str(END),
            "zones": TARGET_ZONES,
            "borders": [(o, d) for o, d, _ in BORDERS],
        },
        stats=stats,
    )

    # Resumen
    ok  = sum(1 for v in stats.values() if v != "ERROR")
    err = sum(1 for v in stats.values() if v == "ERROR")
    logger.info(f"\nResumen: {ok} descargas OK | {err} errores de {len(stats)} total")

    if err > 0:
        failed = [k for k, v in stats.items() if v == "ERROR"]
        logger.warning(f"  Fallidos: {failed}")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_physical()