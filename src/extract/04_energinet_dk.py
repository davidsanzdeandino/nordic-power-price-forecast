"""
04_energinet_dk.py
==================
Extrae datos de Energinet DataService — fuente oficial del TSO danés.

Sin API key — servicio público, sin registro.
URL base: https://api.energidataservice.dk/dataset/

Datasets descargados:
  1. Elspotprices          → Precios spot DK1/DK2 (validación cruzada vs ENTSO-E)
  2. ElectricityBalanceNonv → Balance físico horario (generación por tecnología,
                               consumo, intercambios) — alta granularidad Dinamarca
  3. ProductionConsumptionSettlement → Liquidación real (producción+consumo en MW)

DECISIÓN TÉCNICA: Energinet es la fuente de referencia para datos daneses.
Sus datos de generación eólica son más granulares y fiables que los de ENTSO-E
para DK1/DK2. Se usan como fuente primaria para features de Dinamarca y como
validación cruzada de los precios ENTSO-E.

PROBLEMA PAGINACIÓN: La API de Energinet limita a 100.000 registros por petición
con limit=0 (que en teoría devuelve todo). Para 2 años horarios x 2 zonas =
~17.520 registros, está dentro del límite. Pero para datasets más densos, la
función paginate_dataset() divide la descarga en bloques mensuales.

Uso:
    python src/extract/04_energinet_dk.py
    # O desde el orquestador: python run_pipeline.py --step 4
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL   = "https://api.energidataservice.dk/dataset"
START_DATE = "2023-01-01T00:00"
END_DATE   = "2026-06-01T00:00"   # Exclusivo — cubre hasta may-2026

# Tamaño del bloque de paginación (1 mes = safe para todos los datasets)
CHUNK_MONTHS = 1

# Límite máximo por petición (documentado por Energinet)
API_LIMIT = 100_000

REQUEST_DELAY = 1.0
HTTP_TIMEOUT  = 45

# ── Definición de datasets ────────────────────────────────────────────────────

DATASETS = {
    "Elspotprices": {
        "filter":      '{"PriceArea": ["DK1", "DK2"]}',
        "sort":        "HourUTC ASC",
        "desc":        "Precios spot DK1/DK2 — validación vs ENTSO-E",
        "date_col":    "HourUTC",
        "paginate":    False,   # Solo ~17k filas — cabe en una petición
    },
    "ElectricityBalanceNonv": {
        "filter":      '{"PriceArea": ["DK1", "DK2"]}',
        "sort":        "HourUTC ASC",
        "desc":        "Balance físico horario DK1/DK2 (generación por tech, consumo, intercambios)",
        "date_col":    "HourUTC",
        # DECISIÓN: Paginar por meses — este dataset tiene muchas columnas
        # y puede superar el límite con 2 zonas x 2 años
        "paginate":    True,
    },
    "ProductionConsumptionSettlement": {
        "filter":      '{"PriceArea": ["DK1", "DK2"]}',
        "sort":        "HourUTC ASC",
        "desc":        "Liquidación producción y consumo real (MW) — mayor precisión",
        "date_col":    "HourUTC",
        "paginate":    True,
    },
}


# ── Helpers de petición y paginación ─────────────────────────────────────────

def _single_request(dataset_name: str, config: dict,
                    start: str, end: str, logger) -> list:
    """
    Hace UNA petición a la API de Energinet para un rango dado.
    Devuelve lista de registros (dicts) o lanza excepción.
    """
    url    = f"{BASE_URL}/{dataset_name}"
    params = {
        "start":   start,
        "end":     end,
        "filter":  config["filter"],
        "sort":    config.get("sort", "HourUTC ASC"),
        "limit":   API_LIMIT,
        "timezone": "UTC",
    }

    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)

    # Log detallado si falla
    if not response.ok:
        logger.error(
            f"    HTTP {response.status_code} para {dataset_name} "
            f"[{start} → {end}]: {response.text[:300]}"
        )
        response.raise_for_status()

    data    = response.json()
    records = data.get("records", [])

    # Avisar si la respuesta está truncada (llegamos al límite)
    total = data.get("total", len(records))
    if len(records) >= API_LIMIT:
        logger.warning(
            f"    WARN Posible truncado: {len(records)} registros = límite API. "
            f"Total reportado: {total}. Considera reducir CHUNK_MONTHS."
        )

    return records


def paginate_dataset(dataset_name: str, config: dict, logger) -> pd.DataFrame:
    """
    Descarga un dataset en bloques mensuales y los concatena.
    Necesario cuando el volumen supera API_LIMIT en una sola petición.
    """
    all_records = []

    # Generar bloques mensuales
    current = datetime.strptime(START_DATE[:10], "%Y-%m-%d")
    end_dt  = datetime.strptime(END_DATE[:10],   "%Y-%m-%d")

    while current < end_dt:
        chunk_end = min(current + relativedelta(months=CHUNK_MONTHS), end_dt)
        start_str = current.strftime("%Y-%m-%dT%H:%M")
        end_str   = chunk_end.strftime("%Y-%m-%dT%H:%M")

        logger.info(f"    Bloque: {start_str} → {end_str}...")
        try:
            records = _single_request(dataset_name, config, start_str, end_str, logger)
            all_records.extend(records)
            logger.info(f"      → {len(records)} registros acumulados: {len(all_records)}")
        except Exception as e:
            logger.error(f"    ERROR en bloque {start_str}: {e}")

        current = chunk_end
        time.sleep(REQUEST_DELAY)

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def single_dataset(dataset_name: str, config: dict, logger) -> pd.DataFrame:
    """Descarga un dataset completo en una sola petición."""
    records = _single_request(
        dataset_name, config, START_DATE, END_DATE, logger
    )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ── Procesamiento post-descarga ───────────────────────────────────────────────

def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quita filas repetidas por (timestamp, PriceArea).

    Cada hora trae una fila por zona (DK1 y DK2): deduplicar solo por el
    timestamp dejaba una sola zona por hora. Si no hay columna PriceArea,
    se deduplica por el timestamp.
    """
    key = pd.DataFrame({"timestamp": df.index})
    if "PriceArea" in df.columns:
        key["PriceArea"] = df["PriceArea"].to_numpy()
    return df[~key.duplicated(keep="first").to_numpy()]


def process_and_save(df: pd.DataFrame, dataset_name: str,
                     config: dict, logger) -> int:
    """
    Limpieza mínima, normalización del índice y guardado en Parquet.
    (La limpieza profunda se hace en cleaning.py)

    Returns:
        Número de filas guardadas.
    """
    if df.empty:
        logger.warning(f"  WARN {dataset_name}: DataFrame vacío, no se guarda.")
        return 0

    date_col = config["date_col"]

    # Normalizar columna de fecha → índice UTC
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.rename(columns={date_col: "timestamp_utc"})
        df = df.set_index("timestamp_utc")
    else:
        logger.warning(f"  WARN Columna '{date_col}' no encontrada. Columnas: {list(df.columns)[:8]}")

    # Eliminar duplicados evidentes (algunos datasets tienen HourDK también)
    if "HourDK" in df.columns:
        df = df.drop(columns=["HourDK"])

    df = drop_duplicate_rows(df)
    df = df.sort_index(kind="stable")

    # Guardar
    filename = f"energinet_{dataset_name}.parquet"
    df.to_parquet(OUTPUT_DIR / filename)

    n = len(df)
    logger.info(f"  OK {filename} | {n} filas | cols: {list(df.columns)[:6]}...")
    return n


# ── Función principal ─────────────────────────────────────────────────────────

def extract_energinet() -> dict:
    """
    Descarga todos los datasets de Energinet y los guarda en data/raw/.

    Returns:
        Dict con stats {dataset: n_filas | 'ERROR'}
    """
    logger = get_logger("energinet_dk")
    stats  = {}

    logger.info(f"=== Energinet DataService | {START_DATE[:10]} → {END_DATE[:10]} ===")
    logger.info(f"Datasets: {list(DATASETS.keys())}")

    for dataset_name, config in DATASETS.items():
        logger.info(f"\nDataset: {dataset_name}")
        logger.info(f"   {config['desc']}")
        try:
            if config.get("paginate"):
                logger.info(f"   Modo: paginación mensual")
                df = paginate_dataset(dataset_name, config, logger)
            else:
                logger.info(f"   Modo: petición única")
                df = single_dataset(dataset_name, config, logger)

            n = process_and_save(df, dataset_name, config, logger)
            stats[dataset_name] = n

        except Exception as e:
            logger.error(f"  ERROR {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            stats[dataset_name] = "ERROR"

    # Guardar metadatos
    save_run_metadata(
        script_name="energinet_dk",
        params={
            "start":    START_DATE,
            "end":      END_DATE,
            "datasets": list(DATASETS.keys()),
            "base_url": BASE_URL,
        },
        stats=stats,
    )

    # Resumen
    ok  = sum(1 for v in stats.values() if v != "ERROR")
    err = sum(1 for v in stats.values() if v == "ERROR")
    logger.info(f"\nResumen: {ok}/{len(DATASETS)} datasets OK | {err} errores")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # dateutil es necesario para relativedelta (paginación mensual)
    try:
        from dateutil.relativedelta import relativedelta  # noqa: F401
    except ImportError:
        print("ERROR: falta 'python-dateutil'. Ejecuta: pip install python-dateutil")
        raise

    extract_energinet()