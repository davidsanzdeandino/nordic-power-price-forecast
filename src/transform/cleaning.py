"""
cleaning.py
===========
Pipeline de limpieza — Fase B del ETL.

Lee datos crudos de data/raw/, aplica tratamientos y guarda en data/clean/.
Cada transformación queda documentada en el log de limpieza.

Funciones por fuente:
  - clean_entsoe_prices()    → Precios DA por zona y periodo
  - clean_entsoe_physical()  → Generación, forecast, load, flujos, NTC
  - clean_weather()          → Datos Open-Meteo (ya en UTC, limpieza mínima)
  - clean_energinet()        → Precios y balance físico Dinamarca

DECISIONES TÉCNICAS aplicadas:
  - Duplicados: keep='first' (primer registro es el publicado originalmente)
  - Huecos 1-3h: interpolación lineal
  - Huecos 4-6h: forward fill
  - Huecos >6h:  se dejan como NaN (se documenta en el log)
  - Outliers/precios negativos: NO se eliminan, se marcan con flags binarias
  - Generación negativa: cero (artefacto de medición, no tiene significado físico)

Uso:
    python src/transform/cleaning.py
    # O desde el orquestador: python run_pipeline.py --step 8
"""

import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

RAW_DIR   = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Límites de interpolación (horas)
INTERP_LIMIT  = 3   # Interpolación lineal hasta 3h de hueco
FFILL_LIMIT   = 3   # Forward fill adicional hasta 6h total (3+3)

# Umbrales para flags de precio
SPIKE_HIGH_1  = 200   # EUR/MWh — spike moderado
SPIKE_HIGH_2  = 500   # EUR/MWh — spike extremo (crisis)
NEGATIVE_THRESHOLD = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que el índice es DatetimeIndex UTC tz-aware."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El DataFrame no tiene DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp_utc"
    return df


def remove_duplicates(df: pd.DataFrame,
                      area_col: str | None = None) -> tuple[pd.DataFrame, int]:
    """
    Elimina duplicados de índice, devuelve df limpio y n eliminados.

    Con area_col (p. ej. "PriceArea" en los datasets de Energinet, que traen
    una fila por zona y hora) la clave es (índice, area_col): deduplicar solo
    por el índice dejaría una sola zona por hora.
    """
    n_before = len(df)
    if area_col is not None and area_col in df.columns:
        key = pd.DataFrame({"timestamp": df.index, area_col: df[area_col].to_numpy()})
        df = df[~key.duplicated(keep="first").to_numpy()]
    else:
        df = df[~df.index.duplicated(keep="first")]
    return df, n_before - len(df)


def fill_gaps(df: pd.DataFrame, numeric_cols: list) -> tuple[pd.DataFrame, dict]:
    """
    Rellena huecos en serie temporal continua horaria.

    DECISIÓN TÉCNICA:
    - Reindexar a frecuencia horaria completa para detectar huecos implícitos
    - Interpolación lineal hasta 3h (apropiado para cambios graduales)
    - Forward fill hasta 3h adicionales (para huecos algo mayores)
    - Huecos >6h se dejan como NaN — son datos genuinamente faltantes
      y no deben inventarse; el modelo los manejará con dropna o imputación propia
    """
    # Detectar huecos antes de reindexar
    full_idx = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="h", tz="UTC"
    )
    n_gaps = len(full_idx) - len(df)

    df = df.reindex(full_idx)
    df.index.name = "timestamp_utc"

    null_before = df[numeric_cols].isnull().sum().sum()

    for col in numeric_cols:
        df[col] = df[col].interpolate(method="linear", limit=INTERP_LIMIT)
        df[col] = df[col].ffill(limit=FFILL_LIMIT)

    null_after = df[numeric_cols].isnull().sum().sum()

    stats = {
        "implicit_gaps_found":  n_gaps,
        "nulls_before_fill":    int(null_before),
        "nulls_after_fill":     int(null_after),
        "nulls_unfillable":     int(null_after),   # Huecos >6h dejados como NaN
    }
    return df, stats


def log_clean_stats(logger, source: str, zone: str, stats: dict):
    """Imprime resumen de limpieza de forma consistente."""
    logger.info(
        f"  [{source} | {zone}] "
        f"huecos={stats.get('implicit_gaps_found', '?')} | "
        f"dupes={stats.get('duplicates_removed', '?')} | "
        f"nulos sin rellenar={stats.get('nulls_unfillable', '?')}"
    )


# ── A. Precios ENTSO-E ────────────────────────────────────────────────────────

def clean_entsoe_prices(logger, all_stats: dict):
    """
    Limpia archivos entsoe_prices_*.parquet.

    Añade flags:
      - is_negative_price  (precio < 0)
      - is_spike_200       (precio > 200 EUR/MWh)
      - is_spike_500       (precio > 500 EUR/MWh — crisis extrema)

    DECISIÓN TÉCNICA: Los precios negativos y spikes son información,
    no errores. Se marcan con flags binarias para que el modelo pueda
    aprender cuándo ocurren, pero NO se eliminan.
    """
    logger.info("\n--- A. Limpieza precios ENTSO-E ---")
    files = sorted(RAW_DIR.glob("entsoe_prices_*.parquet"))

    if not files:
        logger.warning("  WARN No se encontraron archivos entsoe_prices_*.parquet")
        return

    for file in files:
        # Extraer zona y periodo del nombre: entsoe_prices_DK_1_2023-2024.parquet
        stem_parts = file.stem.replace("entsoe_prices_", "").rsplit("_", 1)
        tag = file.stem.replace("entsoe_prices_", "")
        try:
            df = pd.read_parquet(file)
            df = ensure_utc(df)
            df, n_dupes = remove_duplicates(df)

            numeric_cols = ["price_eur_mwh"]
            df, gap_stats = fill_gaps(df, numeric_cols)
            gap_stats["duplicates_removed"] = n_dupes

            # Flags (DECISIÓN TÉCNICA: mantener outliers, marcar con flags)
            df["is_negative_price"] = (df["price_eur_mwh"] < NEGATIVE_THRESHOLD).astype(int)
            df["is_spike_200"]      = (df["price_eur_mwh"] > SPIKE_HIGH_1).astype(int)
            df["is_spike_500"]      = (df["price_eur_mwh"] > SPIKE_HIGH_2).astype(int)

            # Estadísticas descriptivas del precio (útiles para el EDA)
            price_stats = df["price_eur_mwh"].describe().to_dict()
            n_negative  = int(df["is_negative_price"].sum())
            n_spike200  = int(df["is_spike_200"].sum())

            out_path = CLEAN_DIR / f"prices_{tag}.parquet"
            df.to_parquet(out_path)

            log_clean_stats(logger, "prices", tag, gap_stats)
            logger.info(
                f"    precio: min={price_stats['min']:.1f} | "
                f"mean={price_stats['mean']:.1f} | "
                f"max={price_stats['max']:.1f} EUR/MWh | "
                f"negativos={n_negative} | spikes>200={n_spike200}"
            )
            all_stats[f"prices_{tag}"] = {**gap_stats, **price_stats,
                                           "n_negative": n_negative,
                                           "n_spike_200": n_spike200}

        except Exception as e:
            logger.error(f"  ERROR {file.name}: {e}")
            import traceback; traceback.print_exc()


# ── B. Datos físicos ENTSO-E ──────────────────────────────────────────────────

def _clean_generic_entsoe(file: Path, logger) -> tuple[pd.DataFrame, dict] | None:
    """
    Limpieza genérica para archivos físicos de ENTSO-E.
    Devuelve (df_limpio, stats) o None si hay error.
    """
    try:
        df = pd.read_parquet(file)
        df = ensure_utc(df)
        df, n_dupes = remove_duplicates(df)

        # Columnas numéricas (excluir strings como 'zone')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            df, gap_stats = fill_gaps(df, numeric_cols)
        else:
            gap_stats = {"implicit_gaps_found": 0, "nulls_before_fill": 0,
                         "nulls_after_fill": 0, "nulls_unfillable": 0}

        gap_stats["duplicates_removed"] = n_dupes

        # DECISIÓN TÉCNICA: Generación negativa → cero.
        # Valores negativos en generación son artefactos de medición
        # (bombeo en hidro se reporta separado). Sin significado físico.
        for col in numeric_cols:
            if any(kw in col.lower() for kw in ["gen", "wind", "solar",
                                                  "nuclear", "hydro", "load"]):
                df[col] = df[col].clip(lower=0)

        return df, gap_stats

    except Exception as e:
        logger.error(f"  ERROR {file.name}: {e}")
        return None


def clean_entsoe_physical(logger, all_stats: dict):
    """
    Limpia todos los archivos físicos: generación, forecast, load, flujos, NTC.
    Aplica limpieza genérica + guardado en clean/.
    """
    logger.info("\n--- B. Limpieza datos físicos ENTSO-E ---")

    # Patrones de archivos físicos (todo excepto precios)
    patterns = [
        ("gen",          "entsoe_gen_*.parquet"),
        ("forecast",     "entsoe_forecast_*.parquet"),
        ("load",         "entsoe_load_*.parquet"),
        ("load_forecast","entsoe_load_forecast_*.parquet"),
        ("flow",         "entsoe_flow_*.parquet"),
        ("ntc",          "entsoe_ntc_*.parquet"),
    ]

    for prefix, pattern in patterns:
        files = sorted(RAW_DIR.glob(pattern))
        if not files:
            logger.info(f"  No hay archivos {pattern}")
            continue

        for file in files:
            tag = file.stem  # Nombre completo sin extensión
            result = _clean_generic_entsoe(file, logger)
            if result is None:
                all_stats[tag] = "ERROR"
                continue

            df, gap_stats = result
            out_path = CLEAN_DIR / f"{tag}.parquet"
            df.to_parquet(out_path)
            log_clean_stats(logger, prefix, tag, gap_stats)
            all_stats[tag] = gap_stats


# ── C. Meteorología Open-Meteo ────────────────────────────────────────────────

def clean_weather(logger, all_stats: dict):
    """
    Limpia archivos openmeteo_*.parquet.

    NOTA: Como 03_openmeteo_weather.py ya solicita UTC directamente a la API,
    no es necesaria conversión DST. La limpieza es principalmente:
    - Eliminar duplicados (raro pero posible en bordes de periodo)
    - Rellenar huecos puntuales (Open-Meteo es muy fiable, <0.1% nulos típico)
    - Verificar cobertura y columnas esperadas
    """
    logger.info("\n--- C. Limpieza datos meteorológicos Open-Meteo ---")
    files = sorted(RAW_DIR.glob("openmeteo_*.parquet"))

    EXPECTED_VARS = [
        "temperature_2m", "precipitation",
        "wind_speed_10m", "wind_speed_100m",
        "wind_direction_100m", "solar_radiation",
    ]

    if not files:
        logger.warning("  WARN No se encontraron archivos openmeteo_*.parquet")
        return

    for file in files:
        location = file.stem.replace("openmeteo_", "")
        try:
            df = pd.read_parquet(file)
            df = ensure_utc(df)
            df, n_dupes = remove_duplicates(df)

            # Verificar variables esperadas
            missing_vars = [v for v in EXPECTED_VARS if v not in df.columns]
            if missing_vars:
                logger.warning(f"  WARN {location}: Variables no encontradas: {missing_vars}")

            # Columnas meteorológicas numéricas (excluir metadatos de ubicación)
            meta_cols    = ["location", "zone", "lat", "lon"]
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                            if c not in meta_cols]

            df, gap_stats = fill_gaps(df, numeric_cols)
            gap_stats["duplicates_removed"] = n_dupes

            # Validaciones físicas mínimas
            if "wind_speed_100m" in df.columns:
                n_neg_wind = int((df["wind_speed_100m"] < 0).sum())
                if n_neg_wind > 0:
                    logger.warning(f"  WARN {location}: {n_neg_wind} valores negativos en wind_speed_100m → clip a 0")
                    df["wind_speed_100m"] = df["wind_speed_100m"].clip(lower=0)

            if "solar_radiation" in df.columns:
                df["solar_radiation"] = df["solar_radiation"].clip(lower=0)

            out_path = CLEAN_DIR / f"weather_{location}.parquet"
            df.to_parquet(out_path)
            log_clean_stats(logger, "weather", location, gap_stats)
            all_stats[f"weather_{location}"] = gap_stats

        except Exception as e:
            logger.error(f"  ERROR {location}: {e}")
            import traceback; traceback.print_exc()
            all_stats[f"weather_{location}"] = "ERROR"


# ── D. Energinet ──────────────────────────────────────────────────────────────

def clean_energinet(logger, all_stats: dict):
    """
    Limpia archivos energinet_*.parquet.

    Elspotprices → pivot por zona (DK1/DK2) → precios validados
    ElectricityBalanceNonv → split por zona → balance físico
    ProductionConsumptionSettlement → split por zona → liquidación
    """
    logger.info("\n--- D. Limpieza datos Energinet ---")

    # ── D1. Precios spot (validación cruzada)
    path = RAW_DIR / "energinet_Elspotprices.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df = ensure_utc(df)
            df, n_dupes = remove_duplicates(df, area_col="PriceArea")

            if "SpotPriceEUR" in df.columns and "PriceArea" in df.columns:
                df = df.reset_index()
                df = df.drop_duplicates(subset=["timestamp_utc", "PriceArea"])
                df_pivot = df.pivot(
                    index="timestamp_utc", columns="PriceArea", values="SpotPriceEUR"
                )
                df_pivot.columns = [f"price_energinet_{c.lower()}"
                                     for c in df_pivot.columns]
                df_pivot.index = pd.to_datetime(df_pivot.index, utc=True)
                df_pivot.index.name = "timestamp_utc"

                out = CLEAN_DIR / "energinet_prices_validated.parquet"
                df_pivot.to_parquet(out)
                logger.info(f"  OK Precios Energinet: {len(df_pivot)} filas | cols: {list(df_pivot.columns)}")
                all_stats["energinet_prices"] = {"rows": len(df_pivot), "duplicates_removed": n_dupes}
            else:
                logger.warning(f"  WARN Columnas no encontradas en Elspotprices: {list(df.columns)[:8]}")
        except Exception as e:
            logger.error(f"  ERROR Elspotprices: {e}")
            all_stats["energinet_prices"] = "ERROR"

    # ── D2. Balance físico por zona
    path_bal = RAW_DIR / "energinet_ElectricityBalanceNonv.parquet"
    if path_bal.exists():
        try:
            df_bal = pd.read_parquet(path_bal)
            df_bal = ensure_utc(df_bal)
            df_bal, n_dupes = remove_duplicates(df_bal, area_col="PriceArea")

            for area in ["DK1", "DK2"]:
                if "PriceArea" not in df_bal.columns:
                    break
                area_df = df_bal[df_bal["PriceArea"] == area].copy()
                if area_df.empty:
                    continue

                area_df = area_df.drop(columns=["PriceArea"], errors="ignore")
                numeric_cols = area_df.select_dtypes(include=[np.number]).columns.tolist()
                area_df, gap_stats = fill_gaps(area_df, numeric_cols)
                gap_stats["duplicates_removed"] = n_dupes

                # Renombrar columnas con prefijo de zona
                area_df.columns = [f"{area}_{c}" if c != "timestamp_utc" else c
                                    for c in area_df.columns]

                out = CLEAN_DIR / f"energinet_balance_{area}.parquet"
                area_df.to_parquet(out)
                log_clean_stats(logger, "balance", area, gap_stats)
                all_stats[f"energinet_balance_{area}"] = gap_stats

        except Exception as e:
            logger.error(f"  ERROR ElectricityBalanceNonv: {e}")
            all_stats["energinet_balance"] = "ERROR"

    # ── D3. Liquidación producción/consumo
    path_settlement = RAW_DIR / "energinet_ProductionConsumptionSettlement.parquet"
    if path_settlement.exists():
        try:
            df_set = pd.read_parquet(path_settlement)
            df_set = ensure_utc(df_set)
            df_set, n_dupes = remove_duplicates(df_set, area_col="PriceArea")

            for area in ["DK1", "DK2"]:
                if "PriceArea" not in df_set.columns:
                    break
                area_df = df_set[df_set["PriceArea"] == area].copy()
                if area_df.empty:
                    continue

                area_df = area_df.drop(columns=["PriceArea"], errors="ignore")
                numeric_cols = area_df.select_dtypes(include=[np.number]).columns.tolist()
                area_df, gap_stats = fill_gaps(area_df, numeric_cols)
                gap_stats["duplicates_removed"] = n_dupes

                area_df.columns = [f"{area}_settlement_{c}" if c != "timestamp_utc" else c
                                    for c in area_df.columns]

                out = CLEAN_DIR / f"energinet_settlement_{area}.parquet"
                area_df.to_parquet(out)
                log_clean_stats(logger, "settlement", area, gap_stats)
                all_stats[f"energinet_settlement_{area}"] = gap_stats

        except Exception as e:
            logger.error(f"  ERROR ProductionConsumptionSettlement: {e}")
            all_stats["energinet_settlement"] = "ERROR"


# ── Función principal ─────────────────────────────────────────────────────────

def run_cleaning() -> dict:
    """
    Ejecuta el pipeline completo de limpieza sobre todos los datos crudos.

    Returns:
        Dict con stats de limpieza por fuente/zona.
    """
    logger    = get_logger("cleaning")
    all_stats = {}

    logger.info("=== PIPELINE DE LIMPIEZA ===")
    logger.info(f"RAW_DIR:   {RAW_DIR}")
    logger.info(f"CLEAN_DIR: {CLEAN_DIR}")
    logger.info(f"Parámetros: interp_limit={INTERP_LIMIT}h | ffill_limit={FFILL_LIMIT}h")

    clean_entsoe_prices(logger, all_stats)
    clean_entsoe_physical(logger, all_stats)
    clean_weather(logger, all_stats)
    clean_energinet(logger, all_stats)

    # Guardar metadatos de limpieza
    save_run_metadata(
        script_name="cleaning",
        params={
            "interp_limit_h":       INTERP_LIMIT,
            "ffill_limit_h":        FFILL_LIMIT,
            "spike_threshold_1":    SPIKE_HIGH_1,
            "spike_threshold_2":    SPIKE_HIGH_2,
            "keep_outliers":        True,
            "clip_negative_gen":    True,
        },
        stats=all_stats,
    )

    # Resumen
    ok  = sum(1 for v in all_stats.values() if v != "ERROR")
    err = sum(1 for v in all_stats.values() if v == "ERROR")
    logger.info(f"\nLimpieza completada: {ok} fuentes OK | {err} errores")

    # Listar archivos generados en clean/
    clean_files = sorted(CLEAN_DIR.glob("*.parquet"))
    logger.info(f"\nArchivos en data/clean/ ({len(clean_files)}):")
    for f in clean_files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name:60s} {size_kb:8.1f} KB")

    return all_stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cleaning()