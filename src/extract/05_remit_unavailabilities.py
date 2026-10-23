"""
05_remit_unavailabilities.py
============================
Descarga datos de indisponibilidad de unidades de generación (REMIT/UMM)
desde ENTSO-E Transparency Platform.

DECISIÓN TÉCNICA: Cuando una planta nuclear sueca de 1.4 GW sale de
servicio por mantenimiento, el precio nórdico sube 20-40 EUR/MWh durante
semanas. El modelo actual es completamente ciego a estos eventos.

Los datos REMIT (Regulation on Energy Market Integrity and Transparency)
incluyen:
  - Paradas planificadas de centrales (mantenimiento programado)
  - Paradas no planificadas (averías, trips)
  - Reducción de capacidad de interconexiones

Se agregan por zona y tipo de combustible para crear features:
  - MW nucleares no disponibles en SE3 (el más impactante)
  - MW térmicos no disponibles en FI
  - % de capacidad de interconexión reducida

Fuente: ENTSO-E Transparency Platform (usa entsoe-py)
Documentación: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

Dependencias:
    pip install entsoe-py

Uso:
    python src/extract/05_remit_unavailabilities.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
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

# Periodo
START = pd.Timestamp("2023-01-01", tz="UTC")
END   = pd.Timestamp("2026-06-01", tz="UTC")

# Zonas nórdicas relevantes para indisponibilidad
# Las que más impactan al precio DK1/DK2
ZONES = {
    "SE_3": "10Y1001A1001A46L",  # Suecia central — nuclear + hidro
    "SE_4": "10Y1001A1001A47J",  # Suecia sur — conecta con DK
    "FI":   "10YFI-1--------U",  # Finlandia — nuclear (Olkiluoto)
    "NO_2": "10YNO-2--------T",  # Noruega sur — hidro + cable a DK
    "DK_1": "10YDK-1--------W",  # Dinamarca oeste
    "DK_2": "10YDK-2--------M",  # Dinamarca este
}

# Pausa entre llamadas API (ENTSO-E rate limit: ~400 req/min)
REQUEST_DELAY = 1.5

# Chunk size para descarga (ENTSO-E limita a 1 año por petición para UMMs)
CHUNK_MONTHS = 3


# ── Descarga de indisponibilidad ──────────────────────────────────────────────

def download_unavailability_generation(client, zone_name: str,
                                        zone_eic: str, logger) -> pd.DataFrame:
    """
    Descarga indisponibilidad de unidades de generación para una zona.

    Usa entsoe-py: query_unavailability_of_generation_units()
    Devuelve DataFrame con columnas: start, end, plant, fuel_type, unavail_MW, status
    """
    logger.info(f"    Descargando generación {zone_name}...")

    all_records = []

    # Dividir en chunks (ENTSO-E limita respuestas grandes)
    current = START
    while current < END:
        chunk_end = min(current + pd.DateOffset(months=CHUNK_MONTHS), END)
        try:
            df = client.query_unavailability_of_generation_units(
                zone_eic,
                start=current,
                end=chunk_end,
            )

            if df is not None and not df.empty:
                # entsoe-py devuelve un DataFrame con multiindex o columnas variadas
                # Normalizar a formato plano
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()

                all_records.append(df)
                logger.info(f"      {current.strftime('%Y-%m')} → "
                            f"{chunk_end.strftime('%Y-%m')}: {len(df)} eventos")
            else:
                logger.info(f"      {current.strftime('%Y-%m')} → "
                            f"{chunk_end.strftime('%Y-%m')}: sin datos")

        except Exception as e:
            error_msg = str(e)
            if "No matching data" in error_msg or "no data" in error_msg.lower():
                logger.info(f"      {current.strftime('%Y-%m')} → "
                            f"{chunk_end.strftime('%Y-%m')}: sin datos disponibles")
            else:
                logger.warning(f"      WARN {current.strftime('%Y-%m')}: {error_msg[:120]}")

        current = chunk_end
        time.sleep(REQUEST_DELAY)

    if not all_records:
        return pd.DataFrame()

    result = pd.concat(all_records, ignore_index=True)
    return result


def download_unavailability_transmission(client, zone_eic_from: str,
                                          zone_eic_to: str, label: str,
                                          logger) -> pd.DataFrame:
    """
    Descarga indisponibilidad de interconexiones (NTC reducida).
    """
    logger.info(f"    Descargando transmisión {label}...")

    all_records = []
    current = START

    while current < END:
        chunk_end = min(current + pd.DateOffset(months=CHUNK_MONTHS), END)
        try:
            df = client.query_unavailability_transmission(
                zone_eic_from,
                zone_eic_to,
                start=current,
                end=chunk_end,
            )

            if df is not None and not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                all_records.append(df)
                logger.info(f"      {current.strftime('%Y-%m')} → "
                            f"{chunk_end.strftime('%Y-%m')}: {len(df)} eventos")
            else:
                logger.info(f"      {current.strftime('%Y-%m')} → "
                            f"{chunk_end.strftime('%Y-%m')}: sin datos")

        except Exception as e:
            error_msg = str(e)
            if "No matching data" in error_msg or "no data" in error_msg.lower():
                logger.info(f"      {current.strftime('%Y-%m')}: sin datos disponibles")
            else:
                logger.warning(f"      WARN {current.strftime('%Y-%m')}: {error_msg[:120]}")

        current = chunk_end
        time.sleep(REQUEST_DELAY)

    if not all_records:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)


# ── Agregación a serie horaria ────────────────────────────────────────────────

def aggregate_to_hourly(df_events: pd.DataFrame, zone_name: str,
                         logger) -> pd.DataFrame:
    """
    Convierte eventos de indisponibilidad (start, end, MW) en serie
    horaria de MW no disponibles por tipo de combustible.

    DECISIÓN TÉCNICA: Para cada hora del esqueleto 2023-2024, sumamos
    los MW no disponibles de todos los eventos activos en ese momento.
    Se agrupan por tipo de combustible (nuclear, fossil, hydro, wind, other).
    """
    if df_events.empty:
        logger.warning(f"    {zone_name}: sin eventos para agregar")
        return pd.DataFrame()

    # Crear esqueleto horario
    hourly_idx = pd.date_range(start=START, end=END - pd.Timedelta(hours=1),
                                freq="h", tz="UTC")

    # Detectar columnas de start/end del evento
    start_col = next((c for c in df_events.columns
                      if "start" in c.lower() or "created" in c.lower()), None)
    end_col = next((c for c in df_events.columns
                    if "end" in c.lower() and "update" not in c.lower()), None)

    # Detectar columna de MW
    mw_col = next((c for c in df_events.columns
                   if "power" in c.lower() or "mw" in c.lower()
                   or "capacity" in c.lower() or "nominal" in c.lower()), None)

    # Detectar tipo de combustible
    fuel_col = next((c for c in df_events.columns
                     if "fuel" in c.lower() or "type" in c.lower()
                     or "psrtype" in c.lower() or "production" in c.lower()), None)

    if not start_col or not end_col:
        logger.warning(f"    {zone_name}: columnas start/end no detectadas. "
                        f"Cols: {list(df_events.columns)[:10]}")
        # Guardar raw para inspección manual
        raw_path = OUTPUT_DIR / f"remit_raw_{zone_name}.parquet"
        df_events.to_parquet(raw_path)
        logger.info(f"    Guardado raw para inspección: {raw_path}")
        return pd.DataFrame()

    logger.info(f"    {zone_name}: agregando {len(df_events)} eventos...")
    logger.info(f"      start={start_col}, end={end_col}, mw={mw_col}, fuel={fuel_col}")

    # Asegurar que start/end son datetime
    df_events[start_col] = pd.to_datetime(df_events[start_col], utc=True)
    df_events[end_col] = pd.to_datetime(df_events[end_col], utc=True)

    # Clasificar tipo de combustible en categorías amplias
    fuel_categories = {
        "nuclear": ["B14", "Nuclear"],
        "fossil":  ["B01", "B02", "B03", "B04", "B05", "Fossil", "Gas", "Coal", "Oil", "Lignite"],
        "hydro":   ["B11", "B12", "Hydro", "Water"],
        "wind":    ["B18", "B19", "Wind"],
    }

    def classify_fuel(fuel_str):
        if pd.isna(fuel_str):
            return "other"
        fuel_str = str(fuel_str)
        for category, keywords in fuel_categories.items():
            if any(kw.lower() in fuel_str.lower() for kw in keywords):
                return category
        return "other"

    # Crear serie horaria de MW no disponibles por categoría
    result = pd.DataFrame(index=hourly_idx)
    result.index.name = "timestamp_utc"

    categories = ["nuclear", "fossil", "hydro", "wind", "other", "total"]
    for cat in categories:
        result[f"unavail_{zone_name}_{cat}_mw"] = 0.0

    # Para cada evento, sumar MW no disponibles a las horas afectadas
    for _, event in df_events.iterrows():
        ev_start = event[start_col]
        ev_end = event[end_col]

        # MW no disponibles
        mw = 0.0
        if mw_col and pd.notna(event.get(mw_col)):
            mw = float(event[mw_col])
        else:
            mw = 100.0  # Default si no hay dato de MW

        # Categoría de combustible
        fuel = "other"
        if fuel_col and pd.notna(event.get(fuel_col)):
            fuel = classify_fuel(event[fuel_col])

        # Seleccionar horas afectadas
        mask = (result.index >= ev_start) & (result.index < ev_end)
        if mask.any():
            result.loc[mask, f"unavail_{zone_name}_{fuel}_mw"] += mw
            result.loc[mask, f"unavail_{zone_name}_total_mw"] += mw

    # Estadísticas
    total_col = f"unavail_{zone_name}_total_mw"
    nonzero = (result[total_col] > 0).sum()
    logger.info(f"    {zone_name}: {nonzero:,} horas con indisponibilidad "
                f"({nonzero/len(result)*100:.1f}%)")
    if nonzero > 0:
        logger.info(f"      MW medio cuando hay indisponibilidad: "
                    f"{result.loc[result[total_col]>0, total_col].mean():.0f}")
        logger.info(f"      MW máximo: {result[total_col].max():.0f}")

    # Eliminar columnas que son todo ceros
    zero_cols = [c for c in result.columns if result[c].sum() == 0]
    if zero_cols:
        result = result.drop(columns=zero_cols)
        logger.info(f"      Eliminadas {len(zero_cols)} columnas vacías")

    return result


# ── Función principal ─────────────────────────────────────────────────────────

def extract_remit() -> dict:
    """
    Descarga datos REMIT de ENTSO-E y los guarda como parquet horario.

    Returns:
        Dict con stats {zone: n_events | 'ERROR' | 'NO_DATA'}
    """
    logger = get_logger("remit")
    stats = {}

    if not API_KEY or API_KEY == "PENDIENTE":
        logger.error(
            "No se encontró ENTSOE_API_KEY en .env\n"
            "  → Necesaria para descargar datos REMIT."
        )
        raise ValueError("ENTSOE_API_KEY no configurada.")

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        logger.error("Módulo 'entsoe-py' no instalado. Ejecuta: pip install entsoe-py")
        raise

    client = EntsoePandasClient(api_key=API_KEY)

    logger.info(f"=== REMIT Unavailabilities | {START.date()} → {END.date()} ===")
    logger.info(f"Zonas: {list(ZONES.keys())}")

    all_hourly = []

    # ── 1. Indisponibilidad de generación por zona ────────────────────────
    logger.info("\n[1] Indisponibilidad de unidades de generación...")

    for zone_name, zone_eic in ZONES.items():
        try:
            df_events = download_unavailability_generation(
                client, zone_name, zone_eic, logger
            )

            if df_events.empty:
                stats[f"gen_{zone_name}"] = "NO_DATA"
                continue

            # Guardar eventos raw
            raw_path = OUTPUT_DIR / f"remit_gen_{zone_name}.parquet"
            df_events.to_parquet(raw_path)
            stats[f"gen_{zone_name}"] = len(df_events)

            # Agregar a serie horaria
            df_hourly = aggregate_to_hourly(df_events, zone_name.replace("_", ""), logger)
            if not df_hourly.empty:
                all_hourly.append(df_hourly)

        except Exception as e:
            logger.error(f"  ERROR gen_{zone_name}: {e}")
            stats[f"gen_{zone_name}"] = "ERROR"

    # ── 2. Indisponibilidad de transmisión (interconexiones clave) ────────
    logger.info("\n[2] Indisponibilidad de interconexiones...")

    interconnectors = [
        ("DK_1", "DK_2", "Great Belt"),
        ("DK_1", "NO_2", "Skagerrak"),
        ("DK_2", "SE_4", "Øresund"),
    ]

    for zone_from, zone_to, name in interconnectors:
        eic_from = ZONES.get(zone_from, "")
        eic_to   = ZONES.get(zone_to, "")
        if not eic_from or not eic_to:
            continue

        label = f"{zone_from}_{zone_to}"
        try:
            df_tx = download_unavailability_transmission(
                client, eic_from, eic_to, f"{name} ({label})", logger
            )
            if not df_tx.empty:
                raw_path = OUTPUT_DIR / f"remit_tx_{label}.parquet"
                df_tx.to_parquet(raw_path)
                stats[f"tx_{label}"] = len(df_tx)
            else:
                stats[f"tx_{label}"] = "NO_DATA"

        except Exception as e:
            logger.error(f"  ERROR tx_{label}: {e}")
            stats[f"tx_{label}"] = "ERROR"

    # ── 3. Combinar todas las series horarias ─────────────────────────────
    if all_hourly:
        combined = pd.concat(all_hourly, axis=1)
        # Rellenar NaN con 0 (sin indisponibilidad = 0 MW)
        combined = combined.fillna(0)

        combined_path = OUTPUT_DIR / f"remit_unavailability_hourly.parquet"
        combined.to_parquet(combined_path)
        logger.info(f"\n  OK Combinado: {combined_path.name} | "
                    f"{combined.shape[0]:,} horas × {combined.shape[1]} columnas")
        logger.info(f"     Columnas: {list(combined.columns)}")
    else:
        logger.warning("\n  WARN Sin datos horarios combinados")

    # Guardar metadatos
    save_run_metadata(
        script_name="remit",
        params={
            "start":  str(START.date()),
            "end":    str(END.date()),
            "zones":  list(ZONES.keys()),
        },
        stats=stats,
    )

    ok  = sum(1 for v in stats.values() if isinstance(v, int))
    err = sum(1 for v in stats.values() if v == "ERROR")
    nd  = sum(1 for v in stats.values() if v == "NO_DATA")
    logger.info(f"\nResumen: {ok} OK | {nd} sin datos | {err} errores")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_remit()