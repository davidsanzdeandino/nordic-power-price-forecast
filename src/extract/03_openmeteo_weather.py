"""
03_openmeteo_weather.py
=======================
Extrae PREVISIONES meteorológicas pasadas de Open-Meteo Historical Forecast API.

Sin API key — servicio gratuito, sin registro.

DECISIÓN TÉCNICA (anti-leakage meteorológico): se usa la Historical Forecast
API (https://historical-forecast-api.open-meteo.com), NO la Archive API (ERA5
reanálisis). Motivo: en una subasta day-ahead, a las 12:00 del día D-1 solo se
dispone de la PREVISIÓN del tiempo para el día D, no del tiempo realmente
observado (que ERA5 reconstruye a posteriori). Entrenar y evaluar con la
previsión que realmente existía la víspera evita un sesgo optimista y refleja
las condiciones reales de explotación del modelo.

Variables descargadas por ubicación:
  - temperature_2m        → Temperatura a 2m (°C) — proxy de demanda calefacción
  - wind_speed_10m        → Viento a 10m (m/s)
  - wind_speed_100m       → Viento a 100m (m/s) — altura de turbinas modernas
  - wind_direction_100m   → Dirección del viento a 100m (°)
  - shortwave_radiation   → Radiación solar global (W/m²) — proxy generación solar
  - precipitation         → Precipitación (mm) — proxy estado embalses a largo plazo

DECISIÓN TÉCNICA: 7 ubicaciones estratégicas que representan:
  - Centros de generación eólica (Esbjerg, Aalborg, Kristiansand)
  - Centros de demanda (Copenhagen, Oslo, Stockholm, Helsinki)
El índice se convierte a UTC explícito para alinearse con el resto del pipeline.

Uso:
    python src/extract/03_openmeteo_weather.py
    # O desde el orquestador: python run_pipeline.py --step 3
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Open-Meteo Historical Forecast API — no requiere API key.
# Devuelve, para cada timestamp pasado, la PREVISIÓN que se emitió (no el
# reanálisis ERA5). Es lo que estaría disponible en una subasta day-ahead.
BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

START_DATE = "2023-01-01"
END_DATE   = "2026-05-31"

# DECISIÓN TÉCNICA: 7 ubicaciones cubren los drivers meteorológicos clave.
# - Esbjerg y Aalborg: representan generación eólica offshore y onshore de DK1
# - Kristiansand: hub de interconexión hidro con Noruega (NO2)
# - Copenhagen: centro de demanda DK2
# - Oslo: demanda noruega y proxy temperatura sistema
# - Stockholm: demanda sueca (SE3), afecta spreads
# - Helsinki: centro de demanda finlandés (FI), complementa los datos de Fingrid
LOCATIONS = {
    "Esbjerg_DK1":       {"lat": 55.47, "lon":  8.45, "zone": "DK1",
                          "desc": "Eólica offshore Mar del Norte"},
    "Aalborg_DK1":       {"lat": 57.05, "lon":  9.92, "zone": "DK1",
                          "desc": "Eólica onshore Jutlandia Norte"},
    "Copenhagen_DK2":    {"lat": 55.68, "lon": 12.57, "zone": "DK2",
                          "desc": "Centro de demanda DK2"},
    "Kristiansand_NO2":  {"lat": 58.15, "lon":  8.00, "zone": "NO2",
                          "desc": "Hub interconexión hidro DK1-NO2"},
    "Oslo_NO1":          {"lat": 59.91, "lon": 10.75, "zone": "NO1",
                          "desc": "Demanda noruega y temperatura sistema"},
    "Stockholm_SE3":     {"lat": 59.33, "lon": 18.07, "zone": "SE3",
                          "desc": "Demanda sueca SE3"},
    "Helsinki_FI":       {"lat": 60.17, "lon": 24.94, "zone": "FI",
                          "desc": "Demanda finlandesa"},
}

# Variables horarias a descargar
# DECISIÓN TÉCNICA: wind_speed_100m es el estándar para turbinas modernas
# (>100m de altura de buje). wind_speed_10m se mantiene como referencia
# para comparación y modelos más simples.
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "shortwave_radiation",   # Radiación solar global — renombrado a solar_radiation en clean
]

# Pausa entre llamadas (API gratuita — ser respetuoso con el servicio)
REQUEST_DELAY = 2.5

# Timeout de la petición HTTP (segundos).
# La Historical Forecast API es más lenta que el Archive para rangos largos
# (2 años horarios), así que damos margen amplio.
HTTP_TIMEOUT = 180


# ── Función principal ─────────────────────────────────────────────────────────

def extract_weather() -> dict:
    """
    Descarga datos meteorológicos históricos de Open-Meteo por ubicación.

    Returns:
        Dict con stats {ubicacion: n_filas | 'ERROR'}
    """
    logger = get_logger("openmeteo_weather")
    stats  = {}

    logger.info(f"=== Open-Meteo Historical Forecast | {START_DATE} → {END_DATE} ===")
    logger.info(f"Ubicaciones: {len(LOCATIONS)} | Variables: {len(HOURLY_VARS)}")

    for name, meta in LOCATIONS.items():
        try:
            logger.info(f"\n  Descargando: {name} ({meta['desc']})...")

            params = {
                "latitude":   meta["lat"],
                "longitude":  meta["lon"],
                "start_date": START_DATE,
                "end_date":   END_DATE,
                "hourly":     ",".join(HOURLY_VARS),
                # DECISIÓN TÉCNICA: pedir UTC directamente en la API.
                # Esto evita tener que hacer conversión DST posterior,
                # que es propensa a errores con el cambio de hora danés/nórdico.
                "timezone":   "UTC",
            }

            response = requests.get(BASE_URL, params=params, timeout=HTTP_TIMEOUT)
            response.raise_for_status()

            data_json = response.json()

            # Verificar que la respuesta contiene datos horarios
            hourly = data_json.get("hourly")
            if not hourly or "time" not in hourly:
                logger.error(f"  ERROR {name}: Respuesta sin datos horarios")
                stats[name] = "ERROR"
                continue

            # Construir DataFrame
            df = pd.DataFrame(hourly)

            # DECISIÓN TÉCNICA: Convertir a UTC explícito con tz-aware.
            # Al pedir timezone=UTC en la API, Open-Meteo devuelve timestamps
            # como strings "2023-01-01T00:00" sin sufijo. Los localizamos como UTC.
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
            df.index.name = "timestamp_utc"

            # Renombrar shortwave_radiation → solar_radiation (más descriptivo)
            if "shortwave_radiation" in df.columns:
                df = df.rename(columns={"shortwave_radiation": "solar_radiation"})

            # Añadir metadatos de ubicación como columnas (útil en integración)
            df["location"] = name
            df["zone"]     = meta["zone"]
            df["lat"]      = meta["lat"]
            df["lon"]      = meta["lon"]

            # Verificación de integridad
            expected = 8784 + 8760  # 2024 (bisiesto) + 2023
            coverage = len(df) / expected * 100
            if coverage < 99:
                logger.warning(
                    f"  WARN Cobertura: {coverage:.1f}% "
                    f"({len(df)}/{expected} horas esperadas)"
                )

            # Verificar valores nulos por variable
            null_report = df[HOURLY_VARS[:-1] + ["solar_radiation"]].isnull().sum()
            if null_report.any():
                logger.warning(f"  WARN Nulos detectados: {null_report[null_report > 0].to_dict()}")

            # Guardar
            filename = f"openmeteo_{name}.parquet"
            df.to_parquet(OUTPUT_DIR / filename)
            stats[name] = len(df)
            logger.info(f"  OK {filename} | {len(df)} filas | {df.columns.tolist()}")

            time.sleep(REQUEST_DELAY)

        except requests.exceptions.Timeout:
            logger.error(f"  ERROR {name}: Timeout ({HTTP_TIMEOUT}s) — reintenta más tarde")
            stats[name] = "ERROR"
        except requests.exceptions.HTTPError as e:
            logger.error(f"  ERROR {name}: HTTP {e.response.status_code} — {e}")
            stats[name] = "ERROR"
        except Exception as e:
            logger.error(f"  ERROR {name}: {e}")
            stats[name] = "ERROR"

    # Guardar metadatos
    save_run_metadata(
        script_name="openmeteo_weather",
        params={
            "start_date":   START_DATE,
            "end_date":     END_DATE,
            "locations":    {k: {"lat": v["lat"], "lon": v["lon"], "zone": v["zone"]}
                             for k, v in LOCATIONS.items()},
            "hourly_vars":  HOURLY_VARS,
            "timezone_api": "UTC",
        },
        stats=stats,
    )

    # Resumen
    ok  = sum(1 for v in stats.values() if v != "ERROR")
    err = sum(1 for v in stats.values() if v == "ERROR")
    logger.info(f"\nResumen: {ok}/{len(LOCATIONS)} ubicaciones OK | {err} errores")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_weather()