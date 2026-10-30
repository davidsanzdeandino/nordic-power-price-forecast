"""
integration.py
==============
Pipeline de integración — Fase C del ETL.

Lee todos los datasets limpios de data/clean/ y los combina en un único
DataFrame maestro indexado por timestamp_utc. Lo guarda en data/master/.

Orden de integración:
  1.  Esqueleto temporal horario (2023-2024, UTC)
  2.  Precios day-ahead — fuente primaria: Energinet, fallback: ENTSO-E
  3.  Generación real por zona (ENTSO-E)
  4.  Previsión DA viento y solar (ENTSO-E) — input del modelo
  5.  Demanda real y previsión (ENTSO-E)
  6.  Flujos físicos en interconexiones
  7.  NTC — Net Transfer Capacity
  8.  Meteorología (Open-Meteo, 7 ubicaciones)
  9.  Balance físico Energinet DK1/DK2
  10. Features de calendario (hora, día semana, mes, is_weekend...)
  11. Features derivadas (spread de precio, flag congestión)

DECISIÓN TÉCNICA: Join LEFT sobre el esqueleto temporal.
Cualquier hueco que quede tras la limpieza se mantiene como NaN visible
en lugar de eliminarse silenciosamente — el EDA y el modelo los ven.

Uso:
    python src/transform/integration.py
    # O desde el orquestador: python run_pipeline.py --step 9
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

CLEAN_DIR  = ROOT / "data" / "clean"
MASTER_DIR = ROOT / "data" / "master"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

# Periodo exacto del dataset maestro
START_DATE = "2023-01-01"
END_DATE   = "2026-05-31 23:00"   # Último timestamp incluido (horario)

# Zonas con precio objetivo — las que tendrán columna propia en el master
PRICE_ZONES = ["DK1", "DK2", "NO2", "SE3", "SE4", "FI"]

# Umbrales para feature de congestión
# DECISIÓN TÉCNICA: Una diferencia de precio >10 EUR/MWh entre zonas indica
# que la interconexión está saturada (congestionada). Valor de referencia
# habitual en literatura de mercados nórdicos.
CONGESTION_THRESHOLD = 10.0   # EUR/MWh


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el índice a DatetimeIndex UTC tz-aware."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp_utc"
    return df


def safe_join(master: pd.DataFrame, df: pd.DataFrame,
              label: str, logger) -> pd.DataFrame:
    """
    Hace left join del master con df tras normalizar el índice.
    Avisa si hay columnas duplicadas o si el merge produce nulos inesperados.
    """
    if df.empty:
        logger.warning(f"  WARN {label}: DataFrame vacío, se omite.")
        return master

    df = ensure_utc_index(df)

    # ROBUSTEZ DE ESQUEMA: desde 2025 ENTSO-E pasó precios y forecasts a
    # resolución de 15 min (cambio de MTU europeo). Si una serie llega
    # sub-horaria, se agrega a HORARIO (media) para mantener el dataset
    # consistente con 2023-2024. No-op para series ya horarias o más gruesas.
    if df.index.floor("h").has_duplicates:   # resolución mixta/15-min → horario
        df = df.select_dtypes("number").resample("h").mean()
        logger.info(f"    {label}: serie sub-horaria → agregada a horario (media)")

    # Detectar y renombrar columnas que ya existen en master (evitar _x/_y)
    overlap = set(df.columns) & set(master.columns)
    if overlap:
        logger.warning(f"  WARN {label}: columnas solapadas renombradas: {overlap}")
        df = df.rename(columns={c: f"{label}_{c}" for c in overlap})

    master = master.join(df, how="left")
    return master


# ── Pasos de integración ─────────────────────────────────────────────────────

def step_skeleton(logger) -> pd.DataFrame:
    """1. Esqueleto temporal horario — todas las horas de 2023-2024."""
    idx = pd.date_range(start=START_DATE, end=END_DATE, freq="h", tz="UTC")
    df  = pd.DataFrame(index=idx)
    df.index.name = "timestamp_utc"
    logger.info(f"  Esqueleto: {len(df):,} horas ({START_DATE} → {END_DATE})")
    return df


def step_prices(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    2. Precios day-ahead por zona.

    DECISIÓN TÉCNICA: Fuente primaria = Energinet (más granular para DK).
    Fallback = ENTSO-E (cubre NO2, SE3, SE4, FI que Energinet no tiene).
    Se construye una columna price_<zona> para cada zona objetivo.

    FIX (feb-2026): Tras integrar ambas fuentes, se rellenan los huecos
    en price_DK1/DK2 con datos ENTSO-E. Antes, si Energinet tenía cobertura
    parcial (~50% nulos), esos huecos se quedaban sin rellenar porque ENTSO-E
    solo se añadía como columna de validación.
    """
    logger.info("\n  [2] Integrando precios day-ahead...")
    added = []

    # 2a. Energinet — fuente primaria DK1/DK2
    path_en = CLEAN_DIR / "energinet_prices_validated.parquet"
    if path_en.exists():
        df_en = pd.read_parquet(path_en)
        # Columnas esperadas: price_energinet_dk1, price_energinet_dk2
        # Renombrar a esquema estándar: price_DK1, price_DK2
        rename_map = {}
        for col in df_en.columns:
            for zone in ["dk1", "dk2"]:
                if zone in col.lower():
                    rename_map[col] = f"price_{zone.upper()}"
        df_en = df_en.rename(columns=rename_map)
        master = safe_join(master, df_en, "energinet_prices", logger)
        added += list(rename_map.values())
        logger.info(f"    Energinet: {list(rename_map.values())}")

        # Log cobertura Energinet
        for zone_col in rename_map.values():
            if zone_col in master.columns:
                n_ok = master[zone_col].notna().sum()
                logger.info(f"      {zone_col} cobertura Energinet: "
                            f"{n_ok:,}/{len(master):,} ({n_ok/len(master)*100:.1f}%)")

    # 2b. ENTSO-E — fuente primaria para NO2, SE3, SE4, FI; backup para DK1/DK2
    entsoe_validation_cols = {}  # Mapeo zone → columna ENTSO-E en master

    for zone_raw in ["DK_1", "DK_2", "NO_2", "SE_3", "SE_4", "FI"]:
        zone_clean = zone_raw.replace("_", "")  # 'DK_1' → 'DK1'
        col_name   = f"price_{zone_clean}"
        path_entsoe = CLEAN_DIR / f"prices_{zone_raw}_2023-2024.parquet"

        if not path_entsoe.exists():
            logger.warning(f"    WARN No encontrado: prices_{zone_raw}_2023-2024.parquet")
            continue

        df_e = pd.read_parquet(path_entsoe)[["price_eur_mwh"]].copy()
        df_e = df_e.rename(columns={"price_eur_mwh": f"price_entsoe_{zone_clean}"})

        if col_name not in master.columns:
            # Esta zona no tiene precio Energinet → usar ENTSO-E como principal
            df_e = df_e.rename(columns={f"price_entsoe_{zone_clean}": col_name})
            master = safe_join(master, df_e, f"price_{zone_clean}", logger)
            added.append(col_name)
        else:
            # Ya existe de Energinet → añadir ENTSO-E como columna de validación
            master = safe_join(master, df_e, f"entsoe_{zone_clean}", logger)
            entsoe_validation_cols[zone_clean] = f"price_entsoe_{zone_clean}"

    # 2c. También integrar flags de precio (negativos, spikes) de DK1 y DK2
    for zone_raw in ["DK_1", "DK_2"]:
        zone_clean = zone_raw.replace("_", "")
        path_flags = CLEAN_DIR / f"prices_{zone_raw}_2023-2024.parquet"
        if path_flags.exists():
            df_f = pd.read_parquet(path_flags)
            flag_cols = [c for c in df_f.columns if c.startswith("is_")]
            if flag_cols:
                df_f = df_f[flag_cols].add_prefix(f"{zone_clean}_")
                master = safe_join(master, df_f, f"flags_{zone_clean}", logger)

    # ── 2d. FALLBACK: rellenar huecos en price_DK1/DK2 con ENTSO-E ──────
    #
    # DECISIÓN TÉCNICA: Si Energinet tiene cobertura parcial (e.g. solo 2023,
    # o perdió DK2 por dedup en limpieza), completamos con ENTSO-E.
    # Ambas fuentes reportan el mismo precio day-ahead de Nord Pool, así que
    # la diferencia entre ellas es mínima (típicamente <0.1 EUR/MWh por
    # redondeo o timezone edge cases).
    #
    # También buscamos columnas ENTSO-E que safe_join pueda haber renombrado
    # con prefijo si hubo solapamiento de nombres.

    logger.info("\n    [2d] Fallback ENTSO-E para huecos en precios DK...")

    for zone in ["DK1", "DK2"]:
        col_primary = f"price_{zone}"
        if col_primary not in master.columns:
            continue

        n_null_before = int(master[col_primary].isna().sum())
        if n_null_before == 0:
            logger.info(f"    {col_primary}: sin huecos — fallback no necesario")
            continue

        # Buscar columna ENTSO-E (puede tener varios nombres según safe_join)
        entsoe_col = None
        candidates = [
            f"price_entsoe_{zone}",                    # Nombre estándar
            f"entsoe_{zone}_price_entsoe_{zone}",      # Si safe_join renombró
        ]
        # También buscar por patrón en todas las columnas
        for c in master.columns:
            if "entsoe" in c.lower() and zone.lower() in c.lower() and "price" in c.lower():
                candidates.append(c)

        for cand in candidates:
            if cand in master.columns and master[cand].notna().any():
                entsoe_col = cand
                break

        if entsoe_col is None:
            logger.warning(f"    WARN {col_primary}: {n_null_before:,} huecos pero "
                           f"no se encontró columna ENTSO-E de fallback")
            continue

        # Rellenar
        master[col_primary] = master[col_primary].fillna(master[entsoe_col])
        n_null_after = int(master[col_primary].isna().sum())
        n_filled = n_null_before - n_null_after

        logger.info(f"    {col_primary}: {n_filled:,} huecos rellenados desde {entsoe_col} "
                    f"({n_null_after:,} restantes de {len(master):,})")

    logger.info(f"    Columnas de precio añadidas: {added}")
    return master


def step_generation(master: pd.DataFrame, logger) -> pd.DataFrame:
    """3. Generación real por tipo de fuente y zona."""
    logger.info("\n  [3] Integrando generación real (ENTSO-E)...")
    files = sorted(CLEAN_DIR.glob("entsoe_gen_*.parquet"))
    for file in files:
        zone = file.stem.replace("entsoe_gen_", "")  # Ej: 'DK_1'
        zone_clean = zone.replace("_", "")
        df = pd.read_parquet(file)
        # Prefijo por zona para evitar solapamiento de nombres de tecnología
        df = df.add_prefix(f"gen_{zone_clean}_")
        master = safe_join(master, df, f"gen_{zone_clean}", logger)
    logger.info(f"    Archivos integrados: {len(files)}")
    return master


def step_forecasts(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    4. Previsión DA de viento y solar.

    DECISIÓN TÉCNICA: Es el input más importante del modelo predictivo.
    Se trata como features de primera clase, con prefijo 'forecast_'.
    """
    logger.info("\n  [4] Integrando previsión DA viento/solar (ENTSO-E)...")
    files = sorted(CLEAN_DIR.glob("entsoe_forecast_*.parquet"))
    for file in files:
        zone = file.stem.replace("entsoe_forecast_", "")
        zone_clean = zone.replace("_", "")
        df = pd.read_parquet(file)
        df = df.add_prefix(f"forecast_{zone_clean}_")
        master = safe_join(master, df, f"forecast_{zone_clean}", logger)
    logger.info(f"    Archivos integrados: {len(files)}")
    return master


def step_load(master: pd.DataFrame, logger) -> pd.DataFrame:
    """5. Demanda real y previsión de demanda por zona."""
    logger.info("\n  [5] Integrando demanda (load)...")
    files_real = sorted(CLEAN_DIR.glob("entsoe_load_[!f]*.parquet"))
    files_fc   = sorted(CLEAN_DIR.glob("entsoe_load_forecast_*.parquet"))

    for file in files_real:
        zone = file.stem.replace("entsoe_load_", "")
        zone_clean = zone.replace("_", "")
        df = pd.read_parquet(file)
        df = df.add_prefix(f"load_{zone_clean}_")
        master = safe_join(master, df, f"load_{zone_clean}", logger)

    for file in files_fc:
        zone = file.stem.replace("entsoe_load_forecast_", "")
        zone_clean = zone.replace("_", "")
        df = pd.read_parquet(file)
        df = df.add_prefix(f"load_fc_{zone_clean}_")
        master = safe_join(master, df, f"load_fc_{zone_clean}", logger)

    logger.info(f"    Real: {len(files_real)} | Forecast: {len(files_fc)}")
    return master


def step_flows(master: pd.DataFrame, logger) -> pd.DataFrame:
    """6. Flujos físicos en interconexiones (ambas direcciones)."""
    logger.info("\n  [6] Integrando flujos físicos...")
    files = sorted(CLEAN_DIR.glob("entsoe_flow_*.parquet"))
    for file in files:
        tag = file.stem.replace("entsoe_flow_", "").replace("_", "")
        df  = pd.read_parquet(file)
        df  = df.add_prefix(f"flow_{tag}_")
        master = safe_join(master, df, f"flow_{tag}", logger)
    logger.info(f"    Archivos integrados: {len(files)}")
    return master


def step_ntc(master: pd.DataFrame, logger) -> pd.DataFrame:
    """7. Net Transfer Capacity — capacidad disponible en fronteras."""
    logger.info("\n  [7] Integrando NTC...")
    files = sorted(CLEAN_DIR.glob("entsoe_ntc_*.parquet"))
    for file in files:
        tag = file.stem.replace("entsoe_ntc_", "").replace("_", "")
        df  = pd.read_parquet(file)
        df  = df.add_prefix(f"ntc_{tag}_")
        master = safe_join(master, df, f"ntc_{tag}", logger)
    logger.info(f"    Archivos integrados: {len(files)}")
    return master


def step_weather(master: pd.DataFrame, logger) -> pd.DataFrame:
    """8. Meteorología — 7 ubicaciones, 6 variables cada una."""
    logger.info("\n  [8] Integrando meteorología (Open-Meteo)...")
    files = sorted(CLEAN_DIR.glob("weather_*.parquet"))

    # Columnas de metadatos que no son features (ya están en el nombre del archivo)
    META_COLS = {"location", "zone", "lat", "lon"}

    for file in files:
        location = file.stem.replace("weather_", "")
        df = pd.read_parquet(file)

        # Eliminar columnas de metadatos duplicadas
        drop_cols = [c for c in df.columns if c in META_COLS]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Prefijo con nombre de ubicación abreviado
        # Ej: 'Esbjerg_DK1' → 'wx_EsbjergDK1_wind_speed_100m'
        prefix = f"wx_{location.replace('_', '')}_"
        df = df.add_prefix(prefix)
        master = safe_join(master, df, f"wx_{location}", logger)

    logger.info(f"    Ubicaciones integradas: {len(files)}")
    return master


def step_energinet_balance(master: pd.DataFrame, logger) -> pd.DataFrame:
    """9. Balance físico Energinet — alta granularidad DK1/DK2."""
    logger.info("\n  [9] Integrando balance Energinet...")
    for area in ["DK1", "DK2"]:
        for suffix, label in [("balance", "balance"), ("settlement", "settlement")]:
            path = CLEAN_DIR / f"energinet_{suffix}_{area}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                master = safe_join(master, df, f"energinet_{suffix}_{area}", logger)
                logger.info(f"    energinet_{suffix}_{area}: {df.shape}")
    return master

def step_commodities(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    9b. Precios de commodities energéticos (gas TTF, CO2 EUA).

    DECISIÓN TÉCNICA: El precio del gas TTF y los derechos de emisión CO2
    determinan el coste marginal de generación térmica. Cuando la demanda
    residual es alta (poco viento), el precio eléctrico se fija por el
    coste de la central de gas marginal: TTF/eficiencia + CO2*emisiones.

    Se integra la serie horaria combinada que ya incluye:
    - ttf_eur_mwh: precio gas TTF (EUR/MWh)
    - co2_eua_eur: precio CO2 EUA del EU ETS vía CO2.L (EUR/tCO2, respaldo físico)
    - gas_marginal_cost_eur_mwh: TTF/eficiencia (55%)
    - thermal_marginal_cost: gas + CO2 (ambos en EUR → unidades coherentes)
    """
    logger.info("\n  [9b] Integrando commodities energéticos...")

    # Intentar archivo combinado primero
    combined_path = CLEAN_DIR.parent / "raw" / "commodity_combined_hourly.parquet"
    if not combined_path.exists():
        # Fallback: buscar archivos individuales
        for name in ["ttf_gas", "co2_eua"]:
            path = CLEAN_DIR.parent / "raw" / f"commodity_{name}_hourly.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                master = safe_join(master, df, f"commodity_{name}", logger)
                logger.info(f"    {name}: {df.shape}")
        return master

    df = pd.read_parquet(combined_path)
    master = safe_join(master, df, "commodities", logger)
    logger.info(f"    Commodities: {df.shape[1]} columnas, "
                f"{df.notna().all(axis=1).sum():,} horas completas")

    # Log cobertura
    for col in df.columns:
        n_ok = master[col].notna().sum() if col in master.columns else 0
        logger.info(f"      {col}: {n_ok:,} horas con dato")

    return master


def step_remit(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    9c. Datos REMIT de indisponibilidad de generación y transmisión.

    DECISIÓN TÉCNICA: Las paradas de centrales nucleares y térmicas
    tienen impacto directo en el precio (menos oferta → precio sube).
    Se integra la serie horaria de MW no disponibles por zona y tipo.
    """
    logger.info("\n  [9c] Integrando REMIT unavailabilities...")

    path = CLEAN_DIR.parent / "raw" / "remit_unavailability_hourly.parquet"
    if not path.exists():
        logger.warning("    WARN remit_unavailability_hourly.parquet no encontrado")
        return master

    df = pd.read_parquet(path)
    master = safe_join(master, df, "remit", logger)
    logger.info(f"    REMIT: {df.shape[1]} columnas, {len(df):,} horas")

    # Log resumen de indisponibilidad
    for col in df.columns:
        if "total" in col and col in master.columns:
            nonzero = (master[col] > 0).sum()
            max_mw = master[col].max()
            logger.info(f"      {col}: {nonzero:,} horas activas, max {max_mw:.0f} MW")

    return master


def step_reservoirs(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    9d. Nivel de embalses hidroeléctricos nórdicos (semanal → horario).

    DECISIÓN TÉCNICA: el nivel de embalses NO/SE/FI es el driver estructural
    del precio nórdico (sistema hidro-dependiente). El dato es SEMANAL, así que
    se reindexa al esqueleto horario con forward-fill: el valor de la semana W
    (marcado al inicio de W) ocupa las horas de W. El LAG anti-leakage (13 días,
    features.RESERVOIR_LAG) se aplica en features.py, no aquí: aquí solo dejamos
    la serie alineada y visible.
    """
    logger.info("\n  [9d] Integrando embalses hidroeléctricos (semanal→horario)...")
    raw_dir = CLEAN_DIR.parent / "raw"
    files = sorted(raw_dir.glob("entsoe_reservoir_*.parquet"))
    if not files:
        logger.warning("    WARN Sin ficheros entsoe_reservoir_*.parquet (paso 07)")
        return master

    for path in files:
        df = ensure_utc_index(pd.read_parquet(path))
        # Reindexar semanal → horario con forward-fill sobre el índice del master
        df_h = df.reindex(master.index.union(df.index)).ffill().reindex(master.index)
        master = safe_join(master, df_h, "reservoir", logger)
        for col in df_h.columns:
            n_ok = master[col].notna().sum() if col in master.columns else 0
            logger.info(f"      {col}: {n_ok:,} horas con dato")
    return master


def step_scheduled_exchanges(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    9e. Intercambios comerciales PROGRAMADOS day-ahead por frontera (horario).

    Es el programa nominado en la subasta (distinto del flujo físico real).
    Se renombra cada serie a sched_<O><D>_mw. El LAG anti-leakage (24h) se aplica
    en features.py para evitar circularidad con el precio que se predice.
    """
    logger.info("\n  [9e] Integrando intercambios programados day-ahead...")
    raw_dir = CLEAN_DIR.parent / "raw"
    files = sorted(raw_dir.glob("entsoe_sched_*.parquet"))
    if not files:
        logger.warning("    WARN Sin ficheros entsoe_sched_*.parquet (paso 07)")
        return master

    for path in files:
        # entsoe_sched_DK_1_NO_2.parquet → tag "DK1NO2"
        tag = path.stem.replace("entsoe_sched_", "").replace("_", "")
        df = ensure_utc_index(pd.read_parquet(path))
        col0 = df.columns[0]
        df = df.rename(columns={col0: f"sched_{tag}_mw"})[[f"sched_{tag}_mw"]]
        master = safe_join(master, df, f"sched_{tag}", logger)
        n_ok = master[f"sched_{tag}_mw"].notna().sum()
        logger.info(f"      sched_{tag}_mw: {n_ok:,} horas con dato")
    return master


def step_germany(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    9f. Alemania (DE/LU): precio DA + forecast renovable + carga (mercado acoplado).

    Driver dominante del precio nórdico vía interconectores. El precio se usará
    laggeado en features (anti-leakage); los forecasts, sin lag (disponibles
    antes de la subasta).
    """
    logger.info("\n  [9f] Integrando Alemania (DE/LU)...")
    raw = CLEAN_DIR.parent / "raw"
    files = {
        "entsoe_de_price.parquet":    None,                 # ya viene como de_price_eur
        "entsoe_de_forecast.parquet": "de_forecast",        # prefijo a sus columnas
        "entsoe_de_load_fc.parquet":  None,                 # de_load_forecast_mw
    }
    for fname, prefix in files.items():
        path = raw / fname
        if not path.exists():
            logger.warning(f"    WARN {fname} no encontrado (paso 07)")
            continue
        df = ensure_utc_index(pd.read_parquet(path))
        # El forecast/carga de DE vienen en 15-min → resamplear a horario (media).
        # El precio ya es horario (resample no lo altera).
        df = df.select_dtypes("number").resample("h").mean()
        if prefix:
            df = df.add_prefix(f"{prefix}_")
        master = safe_join(master, df, "de", logger)
        for c in df.columns:
            logger.info(f"      {c}: {master[c].notna().sum():,} horas con dato")
    return master


def step_calendar_features(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    10. Features de calendario derivadas del índice temporal.

    Variables generadas (según especificación del doc Fase 1):
      - hour_of_day    (0-23)
      - day_of_week    (0=lunes … 6=domingo)
      - month          (1-12)
      - quarter        (1-4)
      - is_weekend     (bool) — fin de semana
      - is_holiday_dk  (bool) — festivos daneses aproximados (solo día)
      - week_of_year   (1-53)

    NOTA: Se usan ciclos seno/coseno en features.py (Fase 2) para el modelo.
    Aquí se dejan los valores enteros que son más interpretables para el EDA.
    """
    logger.info("\n  [10] Generando features de calendario...")

    idx = pd.to_datetime(master.index, utc=True)

    master["hour_of_day"]  = idx.hour
    master["day_of_week"]  = idx.dayofweek      # 0 = lunes
    master["month"]        = idx.month
    master["quarter"]      = idx.quarter
    master["week_of_year"] = idx.isocalendar().week.astype(int)
    master["is_weekend"]   = (idx.dayofweek >= 5).astype(int)

    # Festivos daneses aproximados (días fijos — no incluye movibles como Pascua)
    # Los festivos móviles tienen impacto menor y se pueden añadir con 'holidays' lib
    DK_FIXED_HOLIDAYS = {
        (1, 1):   "Año Nuevo",
        (6, 5):   "Día Constitución DK",
        (12, 24): "Nochebuena",
        (12, 25): "Navidad",
        (12, 26): "2º día Navidad",
        (12, 31): "Nochevieja",
    }
    holiday_mask = pd.Series(False, index=master.index)
    for (m, d), _ in DK_FIXED_HOLIDAYS.items():
        holiday_mask |= (idx.month == m) & (idx.day == d)
    master["is_holiday_dk"] = holiday_mask.astype(int)

    logger.info(
        f"    Features: hour_of_day, day_of_week, month, quarter, "
        f"week_of_year, is_weekend, is_holiday_dk"
    )
    logger.info(f"    Fines de semana: {master['is_weekend'].sum():,} horas "
                f"({master['is_weekend'].mean()*100:.1f}%)")
    return master


def step_derived_features(master: pd.DataFrame, logger) -> pd.DataFrame:
    """
    11. Features derivadas de dominio — spreads y flags de congestión.

    Variables generadas:
      - spread_DK1_DK2       → Diferencia de precio DK1-DK2 (EUR/MWh)
      - spread_DK1_NO2       → Spread DK1-NO2 (indica saturación Skagerrak)
      - spread_DK2_SE4       → Spread DK2-SE4 (indica saturación Great Belt Norte)
      - is_congested_DK1_DK2 → Flag: |spread| > umbral (congestión interna)
      - is_negative_price_*  → (ya integrados desde precios, paso 2)

    DECISIÓN TÉCNICA: El spread de precio entre zonas vecinas es el indicador
    más directo de congestión en la interconexión. Cuando DK1≠DK2, el Great Belt
    está saturado. Cuando DK1≠NO2, el Skagerrak está saturado.
    Umbral de 10 EUR/MWh basado en práctica habitual de análisis Nord Pool.
    """
    logger.info("\n  [11] Generando features derivadas (spreads, congestión)...")

    # Spreads de precio entre zonas (solo si ambas columnas existen)
    spread_pairs = [
        ("DK1", "DK2",  "spread_DK1_DK2",  "is_congested_DK1_DK2"),
        ("DK1", "NO2",  "spread_DK1_NO2",  "is_congested_DK1_NO2"),
        ("DK2", "SE4",  "spread_DK2_SE4",  "is_congested_DK2_SE4"),
        ("SE3", "SE4",  "spread_SE3_SE4",  "is_congested_SE3_SE4"),
    ]

    for z1, z2, spread_col, cong_col in spread_pairs:
        c1 = f"price_{z1}"
        c2 = f"price_{z2}"
        if c1 in master.columns and c2 in master.columns:
            master[spread_col] = master[c1] - master[c2]
            master[cong_col]   = (master[spread_col].abs() > CONGESTION_THRESHOLD).astype(int)
            n_cong = int(master[cong_col].sum())
            pct    = master[cong_col].mean() * 100
            logger.info(
                f"    {spread_col}: congestión {n_cong:,} horas "
                f"({pct:.1f}% del tiempo)"
            )
        else:
            missing = [c for c in [c1, c2] if c not in master.columns]
            logger.warning(f"    WARN {spread_col}: faltan columnas {missing}")

    return master


# ── Validación final ──────────────────────────────────────────────────────────

def validate_master(master: pd.DataFrame, logger) -> dict:
    """
    Comprobaciones básicas de integridad del dataset maestro.
    Devuelve dict con métricas de calidad.
    """
    logger.info("\n--- Validación del dataset maestro ---")

    total_cells  = master.shape[0] * master.shape[1]
    null_cells   = master.isnull().sum().sum()
    null_pct     = null_cells / total_cells * 100

    # Cobertura por columna — detecta columnas muy vacías
    col_coverage = (1 - master.isnull().mean()) * 100
    low_coverage = col_coverage[col_coverage < 90].sort_values()

    logger.info(f"  Dimensiones:      {master.shape[0]:,} filas × {master.shape[1]} columnas")
    logger.info(f"  Nulos totales:    {null_cells:,} ({null_pct:.2f}% del total)")
    logger.info(f"  Rango temporal:   {master.index.min()} → {master.index.max()}")
    logger.info(f"  Frecuencia:       {pd.infer_freq(master.index[:100])}")

    if len(low_coverage) > 0:
        logger.warning(f"  WARN Columnas con cobertura <90%:")
        for col, cov in low_coverage.items():
            logger.warning(f"     {col:60s} {cov:.1f}%")

    # Verificar precios objetivo presentes
    for zone in ["DK1", "DK2"]:
        col = f"price_{zone}"
        if col in master.columns:
            n_null  = master[col].isnull().sum()
            pct_null = n_null / len(master) * 100
            logger.info(f"  {col}: {n_null} nulos ({pct_null:.2f}%)")
        else:
            logger.warning(f"  WARN Columna {col} NO encontrada en master")

    return {
        "rows":                int(master.shape[0]),
        "cols":                int(master.shape[1]),
        "null_pct":            round(null_pct, 4),
        "low_coverage_cols":   list(low_coverage.index),
    }


# ── Función principal ─────────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de integración.

    Returns:
        DataFrame maestro integrado.
    """
    logger = get_logger("integration")
    logger.info("=== PIPELINE DE INTEGRACIÓN ===")
    logger.info(f"CLEAN_DIR:  {CLEAN_DIR}")
    logger.info(f"MASTER_DIR: {MASTER_DIR}")
    logger.info(f"Periodo:    {START_DATE} → {END_DATE}")

    # Ejecutar pasos en orden
    master = step_skeleton(logger)
    master = step_prices(master, logger)
    master = step_generation(master, logger)
    master = step_forecasts(master, logger)
    master = step_load(master, logger)
    master = step_flows(master, logger)
    master = step_ntc(master, logger)
    master = step_weather(master, logger)
    master = step_energinet_balance(master, logger)
    master = step_commodities(master, logger)
    master = step_remit(master, logger)
    master = step_reservoirs(master, logger)
    master = step_scheduled_exchanges(master, logger)
    master = step_germany(master, logger)
    master = step_calendar_features(master, logger)
    master = step_derived_features(master, logger)

    # Validación
    val_stats = validate_master(master, logger)

    # ── Guardar ───────────────────────────────────────────────────────────
    out_parquet = MASTER_DIR / "dataset_master.parquet"
    out_csv     = MASTER_DIR / "dataset_master_sample.csv"

    master.to_parquet(out_parquet)
    # CSV solo primeras 2000 filas — para inspección rápida sin abrir el Parquet
    master.head(2000).to_csv(out_csv)

    logger.info(f"\nOK INTEGRACIÓN COMPLETADA")
    logger.info(f"   Parquet: {out_parquet}  ({out_parquet.stat().st_size/1e6:.1f} MB)")
    logger.info(f"   CSV muestra: {out_csv}")
    logger.info(f"   Columnas totales: {master.shape[1]}")

    # Lista de columnas por categoría (resumen para el diccionario de datos)
    categories = {
        "Precio":      [c for c in master.columns if c.startswith("price_")],
        "Generación":  [c for c in master.columns if c.startswith("gen_")],
        "Forecast":    [c for c in master.columns if c.startswith("forecast_")],
        "Demanda":     [c for c in master.columns if c.startswith("load_")],
        "Flujos":      [c for c in master.columns if c.startswith("flow_")],
        "NTC":         [c for c in master.columns if c.startswith("ntc_")],
        "Meteo":       [c for c in master.columns if c.startswith("wx_")],
        "Energinet":   [c for c in master.columns if "energinet" in c],
        "Embalses":    [c for c in master.columns if c.startswith("reservoir_")],
        "Programados": [c for c in master.columns if c.startswith("sched_")],
        "Alemania":    [c for c in master.columns if c.startswith("de_")],
        "Calendario":  ["hour_of_day", "day_of_week", "month", "quarter",
                        "week_of_year", "is_weekend", "is_holiday_dk"],
        "Derivadas":   [c for c in master.columns if "spread" in c or "congested" in c],
    }
    logger.info("\n  Columnas por categoría:")
    for cat, cols in categories.items():
        if cols:
            logger.info(f"    {cat:12s}: {len(cols):3d}  → {cols[:3]}{'...' if len(cols) > 3 else ''}")

    # Guardar metadatos
    save_run_metadata(
        script_name="integration",
        params={
            "start_date":            START_DATE,
            "end_date":              END_DATE,
            "price_zones":           PRICE_ZONES,
            "congestion_threshold":  CONGESTION_THRESHOLD,
        },
        stats={**val_stats, "columns_by_category": {k: len(v) for k, v in categories.items()}},
    )

    return master


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_and_merge()