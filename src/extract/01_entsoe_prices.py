"""
01_entsoe_prices.py
===================
Extrae precios day-ahead (DA) de ENTSO-E para las zonas nórdicas objetivo.

DECISIÓN TÉCNICA: Periodo principal 2023-2024 (post-crisis energética).
Se descarga también 2022 por separado para análisis de eventos extremos,
pero NO se incluye en el dataset maestro de entrenamiento.

Zonas: DK1, DK2, NO2, SE3, SE4, FI
Granularidad: Horaria (UTC)
Formato salida: Parquet por zona en data/raw/

Uso:
    python src/extract/01_entsoe_prices.py
    # O desde el orquestador: python run_pipeline.py --step 1
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# --- Path setup (permite ejecutar desde raíz del proyecto) ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("ENTSOE_API_KEY")

OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DECISIÓN TÉCNICA: Periodo principal 2023-2024 (2 años, post-crisis).
# end = 2025-01-01 00:00 UTC porque la API usa rango [start, end) exclusivo
# al final — sin esto se pierden las últimas horas de 2024-12-31.
PERIODS = {
    "main": {
        "start": pd.Timestamp("2023-01-01", tz="UTC"),
        "end":   pd.Timestamp("2026-06-01", tz="UTC"),
        "label": "2023-2024",
    },
    # 2022 separado: útil para análisis de eventos extremos (crisis gas)
    # pero NO entra en el modelo de entrenamiento.
    "crisis": {
        "start": pd.Timestamp("2022-01-01", tz="UTC"),
        "end":   pd.Timestamp("2023-01-01", tz="UTC"),
        "label": "2022_crisis",
    },
}

# Zonas objetivo (nomenclatura ENTSO-E con guión bajo)
ZONES = ["DK_1", "DK_2", "NO_2", "SE_3", "SE_4", "FI"]

# Pausa entre llamadas (segundos) para no saturar la API gratuita
REQUEST_DELAY = 1.5


# ── Función principal ─────────────────────────────────────────────────────────

def extract_prices(include_crisis_year: bool = True) -> dict:
    """
    Descarga precios day-ahead de ENTSO-E y los guarda como Parquet.

    Args:
        include_crisis_year: Si True, descarga también 2022 por separado.

    Returns:
        Dict con stats de la ejecución {zona_periodo: n_filas | 'ERROR'}
    """
    logger = get_logger("entsoe_prices")

    # Validar API key antes de cualquier otra cosa
    if not API_KEY or API_KEY == "PENDIENTE":
        logger.error(
            "No se encontró ENTSOE_API_KEY en el archivo .env\n"
            "  → Copia .env.example a .env y añade tu clave.\n"
            "  → Solicita la clave en: https://transparency.entsoe.eu/"
        )
        raise ValueError("ENTSOE_API_KEY no configurada.")

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        logger.error("Módulo 'entsoe-py' no instalado. Ejecuta: pip install entsoe-py")
        raise

    client = EntsoePandasClient(api_key=API_KEY)

    periods_to_run = [PERIODS["main"]]
    if include_crisis_year:
        periods_to_run.append(PERIODS["crisis"])

    stats = {}

    for period in periods_to_run:
        label   = period["label"]
        p_start = period["start"]
        p_end   = period["end"]

        logger.info(f"=== Periodo: {label} ({p_start.date()} → {p_end.date()}) ===")

        for zone in ZONES:
            key = f"{zone}_{label}"
            try:
                logger.info(f"  Descargando {zone}...")

                prices = client.query_day_ahead_prices(
                    zone,
                    start=p_start,
                    end=p_end,
                )

                # Convertir a DataFrame limpio
                df = prices.to_frame(name="price_eur_mwh")
                df.index.name = "timestamp_utc"

                # DECISIÓN TÉCNICA: Convertir siempre a UTC explícito.
                # ENTSO-E puede devolver timestamps en hora local del país
                # — esto causa errores silenciosos en el merge posterior.
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")

                # ROBUSTEZ 15-min: desde 2025 los precios ENTSO-E son 15-min.
                # Agregar a HORARIO (media) para homogeneidad con 2023-2024.
                if df.index.floor("h").has_duplicates:   # resolución mixta/15-min
                    df = df[["price_eur_mwh"]].resample("h").mean()
                    df.index.name = "timestamp_utc"

                # Columna de zona (útil al concatenar múltiples zonas)
                df["zone"] = zone.replace("_", "")  # 'DK_1' → 'DK1'

                # Guardar
                filename = OUTPUT_DIR / f"entsoe_prices_{zone}_{label}.parquet"
                df.to_parquet(filename)

                n_rows = len(df)
                stats[key] = n_rows
                logger.info(f"  OK {zone}: {n_rows} filas → {filename.name}")

                # Verificación de cobertura: alertar si faltan muchas horas
                expected_hours = int((p_end - p_start).total_seconds() / 3600)
                coverage = n_rows / expected_hours * 100
                if coverage < 95:
                    logger.warning(
                        f"  WARN Cobertura baja en {zone}: {coverage:.1f}% "
                        f"({n_rows} de {expected_hours} horas esperadas)"
                    )

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.error(f"  ERROR en {zone} ({label}): {e}")
                stats[key] = "ERROR"

    # Guardar JSON de metadatos para trazabilidad
    save_run_metadata(
        script_name="entsoe_prices",
        params={
            "zones": ZONES,
            "periods": {
                p["label"]: {"start": str(p["start"]), "end": str(p["end"])}
                for p in periods_to_run
            },
            "include_crisis_year": include_crisis_year,
        },
        stats=stats,
    )

    # Resumen
    ok  = sum(1 for v in stats.values() if v != "ERROR")
    err = sum(1 for v in stats.values() if v == "ERROR")
    logger.info(f"\nResumen: {ok} descargas OK | {err} errores")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_prices(include_crisis_year=True)