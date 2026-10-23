"""
features.py
===========
Feature engineering completo para el modelo predictivo — Fase 2.

Lee el dataset maestro (data/master/dataset_master.parquet) y genera
el dataset de features listo para entrenar el modelo.

Features generadas:
  A. Temporales cíclicas      — seno/coseno de hora, día semana, mes
  B. Autoregresivas (lags)    — precio lag 24h, 48h, 168h + rolling stats
  C. De dominio físico        — residual load, wind penetration
  D. Meteorológicas agregadas — viento promedio zona, temperatura
  E. De mercado               — spreads, flag congestión
  F. Interacción              — wind × temperatura (demanda fría + bajo viento)
  G. Calendario nórdico       — festivos completos DK/SE/NO/FI + Pascua + Navidad

DECISIÓN TÉCNICA: Codificación cíclica (sin/cos) para variables temporales.
One-hot encoding crearía 24 columnas para hora, 7 para día semana, 12 para mes.
La codificación cíclica preserva la continuidad temporal (hora 23 ≈ hora 0)
y solo necesita 2 columnas por variable.

DECISIÓN TÉCNICA: Los lags se calculan con shift() estricto.
Nunca se usan valores del mismo timestamp o posteriores — data leakage cero.
El shift(24) garantiza que solo vemos el precio de 24h atrás o más.

Uso:
    python src/model/features.py
    # O importando: from src.model.features import generate_features
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

MASTER_PATH  = ROOT / "data" / "master" / "dataset_master.parquet"
FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"

# Zonas objetivo del modelo
# FI añadido como track separado (mercado finlandés): el backbone de datos FI
# (precio, generación nuclear/eólica, forecasts, meteo, REMIT, embalses) ya existe;
# su acoplamiento dominante es Suecia (SE3/SE4) + Estonia, no Alemania (ver add_sweden_features).
TARGET_ZONES = ["DK1", "DK2", "FI"]

# Columnas target (lo que predecimos)
TARGET_COLS = {
    "DK1": "price_DK1",
    "DK2": "price_DK2",
    "FI":  "price_FI",
}

# Fecha de operación comercial de Olkiluoto 3 (1.6 GW) — quiebre estructural del
# mercado FI (precio medio cayó ~75%). Se usa como flag de régimen para el modelo FI.
OL3_COMMERCIAL = "2023-04-16"

# Lags de precio (horas) — reflejan los ciclos naturales del mercado
PRICE_LAGS = [24, 48, 168]  # 1 día, 2 días, 1 semana

# Ventanas para rolling statistics (horas, aplicadas sobre datos con shift 24h)
ROLLING_WINDOWS = [24, 48, 168]

# Umbral de congestión para flag (EUR/MWh)
CONGESTION_THRESHOLD = 10.0


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variables temporales: sin/cos + enteros + bloques de mercado.

    DECISIÓN TÉCNICA: Sin/cos se mantienen (útiles para modelos lineales
    y como regularización suave), pero se añaden hora/dow/mes como enteros
    porque LightGBM los usa mucho mejor: un solo split "hour >= 16" captura
    lo que sin/cos necesita combinar en múltiples niveles de profundidad.

    Se añaden bloques horarios del mercado nórdico — los traders piensan
    en bloques (base, peak, off-peak), no en horas individuales.
    """
    idx = pd.to_datetime(df.index, utc=True)
    # Hora local CET — el mercado nórdico opera en CET, no UTC
    idx_cet = idx.tz_convert("Europe/Copenhagen")

    # ── Sin/cos (mantener para Ridge y como señal suave) ──
    df["feat_hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    df["feat_hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    df["feat_dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["feat_dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["feat_month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["feat_month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["feat_week_sin"]  = np.sin(2 * np.pi * idx.isocalendar().week.astype(int) / 53)
    df["feat_week_cos"]  = np.cos(2 * np.pi * idx.isocalendar().week.astype(int) / 53)

    # ── Enteros directos (LightGBM los usa con splits simples) ──
    df["feat_hour_int"]  = idx_cet.hour
    df["feat_dow_int"]   = idx_cet.dayofweek        # 0=lunes, 6=domingo
    df["feat_month_int"] = idx_cet.month

    # ── Bloques horarios del mercado nórdico (hora CET) ──
    # Los contratos futuros y la operación real usan estos bloques:
    #   Base:     00-24 (todo el día — referencia)
    #   Off-peak: 00-08 (noche, demanda baja, eólica domina)
    #   Peak:     08-20 (actividad económica, demanda alta)
    #   Super-peak: 16-20 (pico tarde, el spread más alto del día)
    hour_cet = idx_cet.hour
    df["feat_block_offpeak"]   = ((hour_cet < 8)).astype(int)
    df["feat_block_peak"]      = ((hour_cet >= 8) & (hour_cet < 20)).astype(int)
    df["feat_block_superpeak"] = ((hour_cet >= 16) & (hour_cet < 20)).astype(int)

    # Transiciones — las horas de cambio de bloque son las más volátiles
    df["feat_block_morning_ramp"]  = ((hour_cet >= 6) & (hour_cet < 9)).astype(int)
    df["feat_block_evening_ramp"]  = ((hour_cet >= 17) & (hour_cet < 21)).astype(int)

    return df

# ── B. Features autoregresivas (lags de precio) ──────────────────────────────

def add_lag_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Lags de precio y estadísticas rolling sobre datos pasados.

    DECISIÓN TÉCNICA: shift(24) mínimo — garantiza que en una predicción
    day-ahead (predicimos T+1 a T+24) nunca usamos datos del periodo
    de predicción. Los rolling stats también aplican shift(24) antes
    de la ventana para no filtrar información futura.
    """
    for zone, target_col in TARGET_COLS.items():
        if target_col not in df.columns:
            logger.warning(f"  WARN: {target_col} no encontrada — lags de {zone} omitidos")
            continue

        series = df[target_col]

        # Lags simples
        for lag in PRICE_LAGS:
            df[f"feat_{zone}_price_lag{lag}h"] = series.shift(lag)

        # Rolling statistics — siempre con shift(24) previo para no ver el futuro
        shifted = series.shift(24)
        for window in ROLLING_WINDOWS:
            df[f"feat_{zone}_price_roll_mean_{window}h"] = shifted.rolling(window).mean()
            df[f"feat_{zone}_price_roll_std_{window}h"]  = shifted.rolling(window).std()
            df[f"feat_{zone}_price_roll_max_{window}h"]  = shifted.rolling(window).max()
            df[f"feat_{zone}_price_roll_min_{window}h"]  = shifted.rolling(window).min()

        # Diferencias de precio (momentum)
        df[f"feat_{zone}_price_diff_24h"]  = series.shift(24) - series.shift(48)
        df[f"feat_{zone}_price_diff_168h"] = series.shift(24) - series.shift(192)

        logger.info(f"    {zone}: lags {PRICE_LAGS}h + rolling {ROLLING_WINDOWS}h")

    return df


# ── C. Features de dominio físico ────────────────────────────────────────────

def add_physical_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features derivadas del conocimiento del mercado eléctrico.

    Residual Load = Demanda - Viento - Solar
    Es el driver principal del precio marginal: cuando la demanda residual
    es alta (poca renovable, mucha demanda), el precio sube porque entran
    en el despacho centrales de gas/carbón más caras.

    Wind Penetration = Viento / Demanda
    Proporción de la demanda cubierta por eólica. Alta penetración → precio bajo
    o negativo. Es el mejor predictor individual del precio en Dinamarca.
    """
    for zone in TARGET_ZONES:
        # Buscar columnas de forecast (preferidas) o reales como fallback
        # El prefijo sigue la convención de integration.py
        all_cols = df.columns.tolist()

        # Viento forecast (day-ahead) — lo que se sabe en el momento de la subasta
        wind_cols = [c for c in all_cols
                     if f"forecast_{zone}_" in c and "Wind" in c]
        wind_onshore  = next((c for c in wind_cols if "Onshore" in c), None)
        wind_offshore = next((c for c in wind_cols if "Offshore" in c), None)

        # Solar forecast
        solar_cols = [c for c in all_cols
                      if f"forecast_{zone}_" in c and "Solar" in c]
        solar_col = solar_cols[0] if solar_cols else None

        # Demanda forecast
        load_fc_col = next(
            (c for c in all_cols if f"load_fc_{zone}_" in c), None
        )
        # Fallback a demanda real
        if not load_fc_col:
            load_fc_col = next(
                (c for c in all_cols if f"load_{zone}_" in c), None
            )

        # Combinar onshore + offshore en total eólico
        if wind_onshore and wind_offshore:
            wind_total = df[wind_onshore].fillna(0) + df[wind_offshore].fillna(0)
        elif wind_onshore:
            wind_total = df[wind_onshore].fillna(0)
        elif wind_offshore:
            wind_total = df[wind_offshore].fillna(0)
        else:
            logger.warning(f"    WARN: {zone}: sin columnas de viento forecast — features físicas omitidas")
            continue

        solar_val = df[solar_col].fillna(0) if solar_col else pd.Series(0, index=df.index)

        # C1. Residual Load
        if load_fc_col:
            load_val = df[load_fc_col].fillna(df[load_fc_col].median())
            df[f"feat_{zone}_residual_load"] = load_val - wind_total - solar_val
            df[f"feat_{zone}_wind_penetration"] = wind_total / load_val.replace(0, np.nan)
            logger.info(f"    {zone}: residual_load + wind_penetration OK")
        else:
            logger.warning(f"    WARN: {zone}: sin columna de demanda — residual_load omitido")

        # C2. Total wind forecast (suma onshore + offshore)
        df[f"feat_{zone}_wind_total_fc"] = wind_total

        # C3. Wind forecast lag (comparar previsión de hoy con la de ayer mismo slot)
        df[f"feat_{zone}_wind_fc_lag24h"] = wind_total.shift(24)
        df[f"feat_{zone}_wind_fc_change"]  = wind_total - wind_total.shift(24)

        # C4. Solar forecast
        if solar_col:
            df[f"feat_{zone}_solar_fc"] = df[solar_col].fillna(0)

    return df


# ── D. Features meteorológicas agregadas ─────────────────────────────────────

def add_weather_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Agrega variables meteorológicas de múltiples ubicaciones en features
    representativas por zona.

    En lugar de usar las 42 columnas meteorológicas individuales directamente
    (que añaden mucho ruido y multicolinealidad), calculamos:
    - Viento promedio de las ubicaciones representativas de cada zona
    - Temperatura media del área
    """
    all_cols = df.columns.tolist()

    # Ubicaciones relevantes por zona
    zone_wx_map = {
        "DK1": ["EsbjergDK1", "AalborgDK1"],
        "DK2": ["CopenhagenDK2"],
        "NO2": ["KristiansandNO2"],
        "SE3": ["StockholmSE3"],
        # FI: demanda (Helsinki) + grandes hubs eólicos del oeste (Vaasa/Olkiluoto/Pori).
        # Los sitios que no existan en el master se ignoran automáticamente (filtro abajo).
        "FI":  ["HelsinkiFI", "VaasaFI", "OlkiluotoFI", "PoriFI"],
    }

    for zone, locations in zone_wx_map.items():
        # Viento 100m promedio de las ubicaciones de la zona
        wind_cols = [
            f"wx_{loc}_wind_speed_100m"
            for loc in locations
            if f"wx_{loc}_wind_speed_100m" in all_cols
        ]
        if wind_cols:
            df[f"feat_wx_{zone}_wind100m_avg"] = df[wind_cols].mean(axis=1)

        # Temperatura promedio
        temp_cols = [
            f"wx_{loc}_temperature_2m"
            for loc in locations
            if f"wx_{loc}_temperature_2m" in all_cols
        ]
        if temp_cols:
            df[f"feat_wx_{zone}_temp_avg"] = df[temp_cols].mean(axis=1)
            # Temperatura al cuadrado: captura efecto no lineal (demanda crece más
            # rápido en frío extremo y calor extremo que en temperaturas medias)
            df[f"feat_wx_{zone}_temp_sq"]  = df[f"feat_wx_{zone}_temp_avg"] ** 2

        # Radiación solar promedio de la zona
        solar_cols = [
            f"wx_{loc}_solar_radiation"
            for loc in locations
            if f"wx_{loc}_solar_radiation" in all_cols
        ]
        if solar_cols:
            df[f"feat_wx_{zone}_solar_avg"] = df[solar_cols].mean(axis=1)

    # Gradiente de temperatura (diferencia entre zonas norte y sur)
    # Indica si hay presión de calefacción diferencial → afecta spreads
    t_fi   = df.get("feat_wx_FI_temp_avg")
    t_dk1  = df.get("feat_wx_DK1_temp_avg")
    if t_fi is not None and t_dk1 is not None:
        df["feat_wx_north_south_temp_gradient"] = t_fi - t_dk1

    n_wx_features = len([c for c in df.columns if c.startswith("feat_wx_")])
    logger.info(f"    {n_wx_features} features meteorológicas agregadas")
    return df


# ── E. Features de mercado ────────────────────────────────────────────────────

def add_market_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features derivadas de la estructura del mercado.

    Spreads entre zonas: indican congestión y presión de arbitraje.
    Capacidad NTC utilizada: indica cuánto margen hay en las interconexiones.
    """
    # Spreads de precio entre zonas vecinas
    spread_pairs = [
        ("DK1", "DK2"), ("DK1", "NO2"), ("DK2", "SE4"), ("SE3", "SE4")
    ]
    for z1, z2 in spread_pairs:
        c1, c2 = f"price_{z1}", f"price_{z2}"
        if c1 in df.columns and c2 in df.columns:
            spread = df[c1] - df[c2]
            df[f"feat_spread_{z1}_{z2}"]           = spread.shift(24)  # lag 24h
            df[f"feat_congested_{z1}_{z2}"]        = (spread.shift(24).abs() > CONGESTION_THRESHOLD).astype(int)

    # Ratio de utilización NTC en interconexiones clave
    ntc_flow_pairs = [
        ("flow_DK1DK2", "ntc_DK1DK2"),
        ("flow_DK1NO2", "ntc_DK1NO2"),
    ]
    for flow_prefix, ntc_prefix in ntc_flow_pairs:
        flow_col = next((c for c in df.columns if flow_prefix in c), None)
        ntc_col  = next((c for c in df.columns if ntc_prefix in c), None)
        if flow_col and ntc_col:
            utilization = (df[flow_col].abs() / df[ntc_col].replace(0, np.nan)).clip(0, 1)
            tag = flow_prefix.replace("flow_", "")
            df[f"feat_ntc_utilization_{tag}"] = utilization.shift(24)

    n_mkt = len([c for c in df.columns if c.startswith("feat_spread_") or
                 c.startswith("feat_congested_") or c.startswith("feat_ntc_")])
    logger.info(f"    {n_mkt} features de mercado generadas")
    return df


# ── F. Features de interacción ────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Interacciones entre features que el modelo no captura bien por separado.

    La más importante: viento × temperatura.
    Cuando hace frío Y hay poco viento → doble presión alcista en el precio.
    Cuando hace calor Y hay mucho viento → precio muy bajo o negativo.
    Esta interacción captura el escenario más extremo de cada dirección.
    """
    for zone in TARGET_ZONES:
        wind_col = f"feat_{zone}_wind_total_fc"
        temp_col = f"feat_wx_{zone}_temp_avg"

        if wind_col in df.columns and temp_col in df.columns:
            # Cold & low wind: situación de precio alto (calefacción + poca renovable)
            # Normalizamos: viento ÷ 1000 MW aprox para que la escala sea comparable
            wind_norm = df[wind_col] / 1000.0
            # Temperatura negativa → más fría → más demanda
            cold_pressure = (-df[temp_col]).clip(lower=0)  # Solo temperaturas <0°C

            df[f"feat_{zone}_cold_low_wind"] = cold_pressure * (1 - wind_norm.clip(0, 1))
            logger.info(f"    {zone}: interacción cold×low_wind OK")

        # Hora × is_weekend: el pico de la mañana es más suave los fines de semana
        if "feat_hour_sin" in df.columns and "is_weekend" in df.columns:
            df[f"feat_{zone}_morning_peak_wd"] = (
                df["feat_hour_sin"] * (1 - df["is_weekend"])
            )

    return df


# ── G. Calendario nórdico completo ───────────────────────────────────────────

def _compute_easter(year: int) -> pd.Timestamp:
    """Calcula domingo de Pascua con algoritmo de Butcher (Gauss mejorado)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year, month, day, tz="UTC")


def _get_nordic_holidays(year: int) -> dict:
    """
    Devuelve dict {Timestamp: nombre} con festivos de DK, SE, NO, FI.

    DECISIÓN TÉCNICA: Los festivos nórdicos causan caídas de demanda del
    15-25% y el modelo anterior solo tenía 6 festivos fijos daneses.
    Faltaban completamente:
      - Semana Santa (Jueves-Lunes): el evento calendario MÁS impactante
        en Nord Pool, 4-5 días consecutivos con -20% demanda
      - Ascensión, Pentecostés: puentes nórdicos largos
      - Midsommar (SE/FI): industria sueca para por completo
      - 17 de mayo (NO): día nacional noruego, hidro impactada
      - Periodo navideño extendido (23 dic - 2 ene)
    """
    easter = _compute_easter(year)
    td = pd.Timedelta

    holidays = {}

    # ── Festivos fijos comunes ───────────────────────────────────────────
    holidays[pd.Timestamp(year, 1, 1, tz="UTC")]   = "new_year"
    holidays[pd.Timestamp(year, 5, 1, tz="UTC")]   = "may_day"         # DK/SE/NO/FI
    holidays[pd.Timestamp(year, 12, 24, tz="UTC")]  = "christmas_eve"   # Nórdico
    holidays[pd.Timestamp(year, 12, 25, tz="UTC")]  = "christmas_day"
    holidays[pd.Timestamp(year, 12, 26, tz="UTC")]  = "christmas_2nd"
    holidays[pd.Timestamp(year, 12, 31, tz="UTC")]  = "new_years_eve"

    # ── Festivos nacionales específicos ──────────────────────────────────
    holidays[pd.Timestamp(year, 6, 5, tz="UTC")]   = "dk_constitution"  # DK
    holidays[pd.Timestamp(year, 5, 17, tz="UTC")]   = "no_national"     # NO
    holidays[pd.Timestamp(year, 6, 6, tz="UTC")]    = "se_national"     # SE
    holidays[pd.Timestamp(year, 12, 6, tz="UTC")]   = "fi_independence" # FI

    # Midsommar: viernes entre 19-25 junio (SE/FI) — industria para
    for d in range(19, 26):
        dt = pd.Timestamp(year, 6, d, tz="UTC")
        if dt.dayofweek == 4:  # Viernes
            holidays[dt] = "midsommar"
            break

    # ── Festivos móviles basados en Pascua ───────────────────────────────
    holidays[easter - td(days=3)]  = "maundy_thursday"    # Jueves Santo
    holidays[easter - td(days=2)]  = "good_friday"        # Viernes Santo
    holidays[easter - td(days=1)]  = "holy_saturday"      # Sábado Santo
    holidays[easter]               = "easter_sunday"
    holidays[easter + td(days=1)]  = "easter_monday"
    holidays[easter + td(days=39)] = "ascension"          # Ascensión (jueves)
    holidays[easter + td(days=49)] = "whit_sunday"        # Pentecostés
    holidays[easter + td(days=50)] = "whit_monday"        # Lunes Pentecostés

    # DK: Store Bededag eliminado en 2024, pero existía en 2023
    if year <= 2023:
        holidays[easter + td(days=26)] = "dk_prayer_day"  # Gran día de oración

    return holidays


def add_holiday_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Genera features de calendario nórdico completas.

    Features generadas:
      - feat_is_holiday_nordic  : 1 si es festivo en >=1 país nórdico
      - feat_is_easter_period   : 1 si estamos en Jueves-Lunes Santo
      - feat_is_christmas_period: 1 si estamos en 23 dic - 2 ene
      - feat_is_bridge_day      : 1 si es laborable entre festivo y fin de semana
      - feat_holiday_proximity  : distancia al festivo más cercano (-2..+2 días)
      - feat_holiday_demand_impact: estimación del impacto en demanda (0.0 - 1.0)

    DECISIÓN TÉCNICA: Estas features capturan los patrones de demanda
    reducida durante festivos nórdicos. El efecto no es binario:
    - Semana Santa: -20% demanda durante 5 días consecutivos
    - Navidad: -15% durante ~10 días (cierre industrial)
    - Festivos nacionales: -10% un solo día
    - Puentes: -5-10% (efecto parcial)
    """
    idx = pd.to_datetime(df.index, utc=True)
    years = sorted(set(idx.year))

    # Recopilar todos los festivos del rango
    all_holidays = {}
    for year in years:
        all_holidays.update(_get_nordic_holidays(year))

    holiday_dates = set(ts.normalize() for ts in all_holidays.keys())
    n_holidays = len(holiday_dates)
    logger.info(f"    Festivos nordicos: {n_holidays} dias en {years}")

    # feat_is_holiday_nordic
    day_normalized = idx.normalize()
    df["feat_is_holiday_nordic"] = day_normalized.isin(holiday_dates).astype(int)

    # feat_is_easter_period (Jueves Santo -> Lunes de Pascua, 5 dias)
    easter_dates = set()
    for year in years:
        easter = _compute_easter(year)
        for offset in range(-3, 2):  # Thu-Mon
            easter_dates.add((easter + pd.Timedelta(days=offset)).normalize())
    df["feat_is_easter_period"] = day_normalized.isin(easter_dates).astype(int)

    # feat_is_christmas_period (23 dic -> 2 ene, ~10 dias de cierre industrial)
    christmas_dates = set()
    for year in years:
        for m, d in [(12, 23), (12, 24), (12, 25), (12, 26), (12, 27),
                     (12, 28), (12, 29), (12, 30), (12, 31)]:
            christmas_dates.add(pd.Timestamp(year, m, d, tz="UTC"))
        christmas_dates.add(pd.Timestamp(year + 1, 1, 1, tz="UTC"))
        christmas_dates.add(pd.Timestamp(year + 1, 1, 2, tz="UTC"))
    # Filtrar solo las que caen dentro del rango del dataset
    christmas_dates = {d for d in christmas_dates
                       if d >= idx.min().normalize() and d <= idx.max().normalize()}
    df["feat_is_christmas_period"] = day_normalized.isin(christmas_dates).astype(int)

    # feat_is_bridge_day: laborable entre festivo y fin de semana
    # Ej: festivo jueves + viernes puente, o festivo martes + lunes puente
    bridge_days = set()
    for hd in holiday_dates:
        dow = hd.dayofweek
        if dow == 3:  # Festivo jueves -> viernes puente
            bridge_days.add(hd + pd.Timedelta(days=1))
        elif dow == 1:  # Festivo martes -> lunes puente
            bridge_days.add(hd - pd.Timedelta(days=1))
    df["feat_is_bridge_day"] = day_normalized.isin(bridge_days).astype(int)

    # feat_holiday_proximity: -2, -1, 0, +1, +2 dias al festivo mas cercano
    # Procesamos de mayor a menor distancia para que el mas cercano gane
    holiday_series = pd.Series(0, index=df.index, dtype="int8")
    for abs_dist in [2, 1, 0]:  # Lejano primero, cercano sobreescribe
        for hd in holiday_dates:
            for sign in ([-1, 1] if abs_dist > 0 else [1]):
                offset = abs_dist * sign
                value = offset
                target = hd + pd.Timedelta(days=offset)
                mask = day_normalized == target
                if mask.any():
                    holiday_series.loc[mask] = value
    df["feat_holiday_proximity"] = holiday_series

    # feat_holiday_demand_impact: impacto estimado en demanda (0 = normal, 1 = maximo)
    # DECISION TECNICA: Valores calibrados a partir de literatura Nord Pool
    impact_map = {
        "new_year": 0.8, "maundy_thursday": 0.7, "good_friday": 0.9,
        "holy_saturday": 0.8, "easter_sunday": 0.9, "easter_monday": 0.9,
        "may_day": 0.6, "ascension": 0.7, "whit_sunday": 0.6,
        "whit_monday": 0.7, "dk_constitution": 0.4, "no_national": 0.5,
        "se_national": 0.4, "fi_independence": 0.4, "midsommar": 0.8,
        "dk_prayer_day": 0.6, "christmas_eve": 0.9, "christmas_day": 1.0,
        "christmas_2nd": 0.9, "new_years_eve": 0.7,
    }
    impact_series = pd.Series(0.0, index=df.index)
    for ts, name in all_holidays.items():
        mask = day_normalized == ts.normalize()
        impact_series.loc[mask] = impact_map.get(name, 0.5)
    df["feat_holiday_demand_impact"] = impact_series

    # Logging
    n_easter = int(df["feat_is_easter_period"].sum())
    n_xmas   = int(df["feat_is_christmas_period"].sum())
    n_bridge = int(df["feat_is_bridge_day"].sum())
    n_hol    = int(df["feat_is_holiday_nordic"].sum())
    logger.info(f"    Horas festivas:    {n_hol:,} ({n_hol/len(df)*100:.1f}%)")
    logger.info(f"    Horas Pascua:      {n_easter:,}")
    logger.info(f"    Horas Navidad ext: {n_xmas:,}")
    logger.info(f"    Horas puente:      {n_bridge:,}")

    return df


# ── H. Features de commodities (gas TTF, CO2 EUA) ────────────────────────────

def add_commodity_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features derivadas de precios de commodities energeticos.

    DECISION TECNICA: El gas TTF y CO2 EUA determinan el "suelo" del
    precio electrico en horas sin renovable. El modelo necesita saber:
    1. El coste marginal termico actual (TTF/eficiencia + CO2*emisiones)
    2. La tendencia reciente (gas subiendo = precios electricos subiendo)
    3. El spark spread (precio electrico - coste gas = margen de la central)

    Todos los valores se aplican con lag 24h para respetar el contexto
    day-ahead (no vemos el precio de gas de hoy al predecir manana).
    """
    n_before = len([c for c in df.columns if c.startswith("feat_")])

    # Buscar columnas de commodities en el master
    ttf_col = next((c for c in df.columns if "ttf" in c.lower() and "mwh" in c.lower()), None)
    co2_col = next((c for c in df.columns if "co2" in c.lower()), None)
    gas_margin_col = next((c for c in df.columns if "gas_marginal" in c.lower()), None)
    thermal_col = next((c for c in df.columns if "thermal_marginal" in c.lower()), None)

    if not ttf_col:
        logger.warning("    WARN: No se encontraron datos de gas TTF en el master")
        logger.warning("    Ejecuta primero: python src/extract/06_commodities.py")
        logger.warning("    y luego: python src/transform/integration.py")
        return df

    logger.info(f"    TTF: {ttf_col}")
    if co2_col:
        logger.info(f"    CO2: {co2_col}")
    if gas_margin_col:
        logger.info(f"    Gas marginal: {gas_margin_col}")

    # H1. Gas TTF con lag 24h (day-ahead context)
    df["feat_ttf_lag24h"] = df[ttf_col].shift(24)

    # H2. Gas TTF momentum (cambio 7 dias)
    df["feat_ttf_7d_change"] = df[ttf_col].shift(24) - df[ttf_col].shift(24 + 168)

    # H3. Gas TTF rolling mean 7d (tendencia)
    df["feat_ttf_roll_mean_7d"] = df[ttf_col].shift(24).rolling(168).mean()

    # H4. Coste marginal termico lag 24h
    if gas_margin_col:
        df["feat_gas_marginal_lag24h"] = df[gas_margin_col].shift(24)
    elif ttf_col:
        # Calcular directamente si no existe la columna
        df["feat_gas_marginal_lag24h"] = df[ttf_col].shift(24) / 0.55  # Eficiencia CCGT

    # H5. Coste marginal termico completo (gas + CO2)
    if thermal_col:
        df["feat_thermal_marginal_lag24h"] = df[thermal_col].shift(24)

    # H6. CO2 lag 24h
    if co2_col:
        df["feat_co2_lag24h"] = df[co2_col].shift(24)

    # H7. Spark spread estimado = precio electrico lag24h - coste gas lag24h
    # Indica si las centrales de gas estan ganando dinero (spread positivo)
    # o perdiendo (spread negativo = probablemente apagadas)
    for zone in ["DK1", "DK2"]:
        price_col = f"price_{zone}"
        margin_feat = "feat_gas_marginal_lag24h"
        if price_col in df.columns and margin_feat in df.columns:
            df[f"feat_{zone}_spark_spread"] = (
                df[price_col].shift(24) - df[margin_feat]
            )

    # H8. Ratio precio/gas — cuantas veces el precio electrico supera al gas
    # Valores > 2 indican que el gas fija precio con margen alto
    # Valores < 1 indican renovable dominando o precios negativos
    if "feat_ttf_lag24h" in df.columns:
        for zone in ["DK1", "DK2"]:
            price_col = f"price_{zone}"
            if price_col in df.columns:
                ttf_safe = df["feat_ttf_lag24h"].replace(0, np.nan)
                df[f"feat_{zone}_price_gas_ratio"] = df[price_col].shift(24) / ttf_safe

    n_after = len([c for c in df.columns if c.startswith("feat_")])
    n_new = n_after - n_before
    logger.info(f"    {n_new} features de commodities generadas")

    return df


# ── I. Features REMIT (indisponibilidad de generacion) ───────────────────────

def add_remit_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features derivadas de datos REMIT de indisponibilidad.

    DECISION TECNICA: Las paradas de centrales (especialmente nucleares
    en Suecia y Finlandia) causan subidas sostenidas de precio durante
    dias o semanas. El modelo necesita saber:
    1. MW totales no disponibles por zona (especialmente SE3, FI)
    2. Cambios bruscos en indisponibilidad (nueva parada = shock de precio)
    3. Indisponibilidad total nordica agregada

    Los valores NO se aplican con lag (la indisponibilidad es publica
    en tiempo real via REMIT y ya conocida antes de la subasta day-ahead).
    """
    n_before = len([c for c in df.columns if c.startswith("feat_")])

    # Buscar columnas REMIT en el master
    unavail_cols = [c for c in df.columns if c.startswith("unavail_")]
    total_cols = [c for c in unavail_cols if "total" in c]

    if not unavail_cols:
        logger.warning("    WARN: No se encontraron datos REMIT en el master")
        logger.warning("    Ejecuta primero: python src/extract/05_remit_unavailabilities.py")
        logger.warning("    y luego: python src/transform/integration.py")
        return df

    logger.info(f"    Columnas REMIT encontradas: {len(unavail_cols)}")

    # I1. Copiar columnas total como features directas
    # (ya son features utiles tal cual — MW no disponibles por zona)
    for col in total_cols:
        zone = col.replace("unavail_", "").replace("_total_mw", "")
        df[f"feat_unavail_{zone}_total"] = df[col]

    # I2. Indisponibilidad total nordica (suma de todas las zonas)
    if total_cols:
        df["feat_unavail_nordic_total"] = df[total_cols].sum(axis=1)
        logger.info(f"    Nordic total: max = {df['feat_unavail_nordic_total'].max():.0f} MW")

    # I3. Cambio en indisponibilidad (delta 24h) — detecta nuevas paradas
    for col in total_cols:
        zone = col.replace("unavail_", "").replace("_total_mw", "")
        feat_name = f"feat_unavail_{zone}_delta24h"
        df[feat_name] = df[col] - df[col].shift(24)

    # I4. Cambio en indisponibilidad nordica total
    if "feat_unavail_nordic_total" in df.columns:
        df["feat_unavail_nordic_delta24h"] = (
            df["feat_unavail_nordic_total"] -
            df["feat_unavail_nordic_total"].shift(24)
        )

    # I5. Flag: indisponibilidad alta (> percentil 90)
    if "feat_unavail_nordic_total" in df.columns:
        threshold = df["feat_unavail_nordic_total"].quantile(0.90)
        df["feat_unavail_high"] = (
            df["feat_unavail_nordic_total"] > threshold
        ).astype(int)
        logger.info(f"    Threshold P90: {threshold:.0f} MW | "
                    f"Horas alta indisponibilidad: {df['feat_unavail_high'].sum():,}")

    # I6. Indisponibilidad por tipo (si hay categorias disponibles)
    # Especialmente nuclear — la mas impactante
    nuclear_cols = [c for c in unavail_cols if "nuclear" in c]
    if nuclear_cols:
        df["feat_unavail_nuclear_total"] = df[nuclear_cols].sum(axis=1)
        logger.info(f"    Nuclear: max = {df['feat_unavail_nuclear_total'].max():.0f} MW")
    else:
        logger.info("    (Sin desglose nuclear — todo clasificado como 'other')")

    # I7. Interaccion: indisponibilidad alta × bajo viento = precio extremo
    # (para FI esto captura Olkiluoto/OL3 fuera + poco viento = spike de precio)
    for zone in TARGET_ZONES:
        wind_col = f"feat_{zone}_wind_total_fc"
        if wind_col in df.columns and "feat_unavail_nordic_total" in df.columns:
            # Normalizar ambos a [0, 1] approx
            wind_q95 = max(df[wind_col].quantile(0.95), 1)
            wind_norm = df[wind_col] / wind_q95
            unavail_q95 = max(df["feat_unavail_nordic_total"].quantile(0.95), 1)
            unavail_norm = df["feat_unavail_nordic_total"] / unavail_q95
            # Alto cuando hay MUCHA indisponibilidad Y POCO viento
            df[f"feat_{zone}_unavail_low_wind"] = unavail_norm * (1 - wind_norm.clip(0, 1))

    n_after = len([c for c in df.columns if c.startswith("feat_")])
    n_new = n_after - n_before
    logger.info(f"    {n_new} features REMIT generadas")

    return df

def add_hydro_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features de embalses hidro e intercambios programados (fuentes ENTSO-E nuevas).

    DECISIÓN TÉCNICA (anti-leakage):
      - Embalses (semanal): se usan con LAG de 168h (1 semana). El nivel de la
        semana W se publica al cierre de W, así que en la subasta del día D solo
        es seguro usar el último nivel ya publicado (≈ semana anterior). Además
        se añade la tendencia a 4 semanas (¿se están vaciando o llenando?).
      - Intercambios programados (horario): se determinan en el mismo clearing
        day-ahead que el precio, así que se usan con LAG de 24h (el programa de
        ayer) para evitar circularidad.
    """
    n_before = len([c for c in df.columns if c.startswith("feat_")])

    # ── Embalses ──
    res_cols = [c for c in df.columns if c.startswith("reservoir_") and c.endswith("_stored_mwh")]
    res_lagged = []
    for col in res_cols:
        zone = col.replace("reservoir_", "").replace("_stored_mwh", "")
        lagged = df[col].shift(168)                       # 1 semana
        df[f"feat_reservoir_{zone}_lag1w"]  = lagged
        df[f"feat_reservoir_{zone}_chg4w"]  = lagged - df[col].shift(168 + 24 * 7 * 4)
        res_lagged.append(f"feat_reservoir_{zone}_lag1w")
    # Total nórdico agregado (suma de zonas con dato), ya laggeado
    if res_lagged:
        df["feat_reservoir_nordic_total_lag1w"] = df[res_lagged].sum(axis=1, min_count=1)
        logger.info(f"    Embalses: {len(res_cols)} zonas (lag 1 semana) + total nórdico OK")

        # ── HYDRO BALANCE: anomalía vs normal estacional (semana-del-año) ──
        # Es EL indicador nórdico: no importa tanto el nivel absoluto como cuánto
        # se desvía del nivel típico para esa época del año. Normal estacional =
        # media por semana-ISO (determinista, calendario; no usa el target).
        # NOTA: con 2 años de datos la "normal" es una estimación gruesa.
        week = pd.Series(
            pd.to_datetime(df.index, utc=True).isocalendar().week.astype(int).values,
            index=df.index,
        )
        for col in res_cols:
            zone = col.replace("reservoir_", "").replace("_stored_mwh", "")
            lagged = df[col].shift(168)
            normal = lagged.groupby(week).transform("mean")
            df[f"feat_reservoir_{zone}_anomaly"] = lagged - normal
        # Anomalía agregada nórdica
        anom_cols = [f"feat_reservoir_{c.replace('reservoir_','').replace('_stored_mwh','')}_anomaly"
                     for c in res_cols]
        df["feat_reservoir_nordic_anomaly"] = df[anom_cols].sum(axis=1, min_count=1)
        logger.info(f"    Hydro balance: anomalía vs normal estacional ({len(res_cols)} zonas) OK")
    else:
        logger.warning("    WARN: Sin columnas de embalses (reservoir_*) — features omitidas")

    # ── Intercambios programados (neto por frontera, lag 24h) ──
    sched_pairs = [
        ("DK1", "NO2"), ("DK1", "DELU"), ("DK2", "SE4"),
        ("DK2", "DELU"), ("DK1", "DK2"),
    ]
    n_sched = 0
    for a, b in sched_pairs:
        c_ab = f"sched_{a}{b}_mw"
        c_ba = f"sched_{b}{a}_mw"
        if c_ab in df.columns and c_ba in df.columns:
            net = df[c_ab].fillna(0) - df[c_ba].fillna(0)
            df[f"feat_sched_net_{a}_{b}"] = net.shift(24)
            n_sched += 1
        elif c_ab in df.columns:
            df[f"feat_sched_{a}_{b}"] = df[c_ab].shift(24)
            n_sched += 1
    if n_sched:
        logger.info(f"    Intercambios programados: {n_sched} fronteras (neto, lag 24h) OK")
    else:
        logger.warning("    WARN: Sin columnas de intercambios programados (sched_*) — omitidas")

    n_after = len([c for c in df.columns if c.startswith("feat_")])
    logger.info(f"    {n_after - n_before} features hidro/intercambios generadas")
    return df


def add_germany_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features del mercado acoplado alemán (DE/LU) — driver dominante del precio
    nórdico según la literatura EPF.

    ANTI-LEAKAGE:
      - El PRECIO DA alemán se fija en la misma subasta acoplada (EUPHEMIA,
        12:00 D-1) que el danés → se usa LAGGEADO (24/48/168h), nunca al mismo
        timestamp.
      - Los FORECASTS alemanes (eólico+solar, carga) sí están disponibles antes
        de la subasta → se usan sin lag (igual que los forecasts nórdicos).
    """
    n_before = len([c for c in df.columns if c.startswith("feat_")])
    if "de_price_eur" not in df.columns:
        logger.warning("    WARN: Sin datos de Alemania (de_price_eur) — features omitidas")
        return df

    # Precio alemán laggeado (acoplamiento de mercados)
    for lag in [24, 48, 168]:
        df[f"feat_de_price_lag{lag}h"] = df["de_price_eur"].shift(lag)
    df["feat_de_price_roll_mean_7d"] = df["de_price_eur"].shift(24).rolling(168).mean()

    # Spread DK-DE laggeado (indica dirección/saturación del interconector)
    for z in ["DK1", "DK2"]:
        if f"price_{z}" in df.columns:
            df[f"feat_{z}_de_spread_lag24h"] = (df[f"price_{z}"] - df["de_price_eur"]).shift(24)

    # Forecast eólico+solar alemán (sin lag — disponible day-ahead)
    de_wind  = [c for c in df.columns if c.startswith("de_forecast_") and "Wind" in c]
    de_solar = [c for c in df.columns if c.startswith("de_forecast_") and "Solar" in c]
    if de_wind:
        df["feat_de_wind_fc"]  = df[de_wind].sum(axis=1, min_count=1)
    if de_solar:
        df["feat_de_solar_fc"] = df[de_solar].sum(axis=1, min_count=1)

    # Residual load alemana forecast = carga_fc - eólica_fc - solar_fc
    if "de_load_forecast_mw" in df.columns:
        load = df["de_load_forecast_mw"]
        wind = df["feat_de_wind_fc"]  if "feat_de_wind_fc"  in df.columns else 0
        solar = df["feat_de_solar_fc"] if "feat_de_solar_fc" in df.columns else 0
        df["feat_de_residual_load_fc"] = load - wind - solar

    n_new = len([c for c in df.columns if c.startswith("feat_")]) - n_before
    logger.info(f"    {n_new} features de Alemania (precio lag + spreads + renovables) OK")
    return df


def add_sweden_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features del mercado acoplado de Finlandia: Suecia (SE3 primario, SE4/SE1) y
    Estonia (EE vía EstLink). Es el análogo finlandés del acoplamiento alemán que
    domina el precio danés. FI iguala su precio con SE3/SE1 la mayoría de horas y
    solo se desacopla (spike o negativo) cuando un interconector satura.

    ANTI-LEAKAGE:
      - Los precios DA suecos/estonios se fijan en la MISMA subasta acoplada
        (EUPHEMIA, 12:00 D-1) que el finlandés → se usan LAGGEADOS (24/48/168h),
        nunca al mismo timestamp.
      - feat_post_ol3: flag de régimen tras la entrada de Olkiluoto 3 (no es un
        precio, es calendario determinista → sin riesgo de leakage).

    Solo aplica si FI es zona objetivo y existe price_SE3.
    """
    if "FI" not in TARGET_COLS or "price_FI" not in df.columns:
        return df
    n_before = len([c for c in df.columns if c.startswith("feat_")])

    # Flag de régimen estructural OL3 (Olkiluoto 3 comercial) — determinista
    idx = pd.to_datetime(df.index, utc=True)
    df["feat_post_ol3"] = (idx >= pd.Timestamp(OL3_COMMERCIAL, tz="UTC")).astype(int)

    # Precios de acoplamiento laggeados + spread FI vs cada zona (lag 24h)
    coupling = [z for z in ["SE3", "SE1", "SE4", "EE"] if f"price_{z}" in df.columns]
    if not coupling:
        logger.warning("    WARN: Sin precios de acoplamiento (price_SE3/SE1/EE) — solo flag OL3")
    for z in coupling:
        col = f"price_{z}"
        for lag in [24, 48, 168]:
            df[f"feat_{z}_price_lag{lag}h"] = df[col].shift(lag)
        df[f"feat_FI_{z}_spread_lag24h"] = (df["price_FI"] - df[col]).shift(24)
    # Media móvil 7d del precio SE3 (referencia lenta de nivel nórdico), laggeada
    if "price_SE3" in df.columns:
        df["feat_SE3_price_roll_mean_7d"] = df["price_SE3"].shift(24).rolling(168).mean()

    # Flujos programados FI (EstLink FI-EE y FI-SE), neto por frontera, lag 24h.
    # El estado del interconector importa más que el nivel de precio EE: FI se va
    # a negativo cuando hay mucho viento y la exportación por EstLink está limitada.
    fi_pairs = [("FI", "EE"), ("FI", "SE_3"), ("FI", "SE_1")]
    for a, b in fi_pairs:
        c_ab, c_ba = f"sched_{a}{b}_mw", f"sched_{b}{a}_mw"
        if c_ab in df.columns and c_ba in df.columns:
            net = df[c_ab].fillna(0) - df[c_ba].fillna(0)
            df[f"feat_sched_net_{a}_{b}"] = net.shift(24)
        elif c_ab in df.columns:
            df[f"feat_sched_{a}_{b}"] = df[c_ab].shift(24)

    n_new = len([c for c in df.columns if c.startswith("feat_")]) - n_before
    logger.info(f"    {n_new} features FI acopladas (SE/EE lag + spreads + OL3 flag) "
                f"| zonas acoplamiento: {coupling or 'ninguna'} OK")
    return df


def add_price_profile_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Perfil típico de precio por (hora, tipo_día, estación).

    DECISIÓN TÉCNICA: En vez de que el modelo aprenda desde cero que
    "a las 17h de un miércoles de invierno el precio suele ser ~85 EUR",
    le damos esa información directamente. El modelo solo tiene que
    predecir la DESVIACIÓN respecto al perfil típico.

    Esto es práctica estándar en EPF profesional (Lago et al., 2021;
    equipos de EPEX/Nord Pool). Los traders llaman a esto el "shape"
    del día y lo usan como referencia mental constante.

    El perfil se calcula con expanding window: solo usa datos PASADOS
    (shift 24h + expanding mean). No hay data leakage.
    """
    n_before = len([c for c in df.columns if c.startswith("feat_")])

    idx_cet = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Copenhagen")

    # Claves de agrupación
    hour_cet = idx_cet.hour
    is_weekend = (idx_cet.dayofweek >= 5).astype(int)
    # Estaciones: 1=invierno, 2=primavera, 3=verano, 4=otoño
    season = idx_cet.month.map({
        12: 1, 1: 1, 2: 1,    # Invierno
        3: 2, 4: 2, 5: 2,     # Primavera
        6: 3, 7: 3, 8: 3,     # Verano
        9: 4, 10: 4, 11: 4,   # Otoño
    })

    for zone, target_col in TARGET_COLS.items():
        if target_col not in df.columns:
            continue

        price = df[target_col]

        # ── K1. Perfil por (hora, weekend, estación) — expanding mean ──
        # Usamos shift(24) + expanding para que sea estrictamente causal
        shifted_price = price.shift(24)

        # Crear clave de grupo
        group_key = hour_cet.astype(str) + "_" + is_weekend.astype(str) + "_" + season.astype(str)
        profile = shifted_price.groupby(group_key).expanding().mean()
        # Realinear al índice original
        profile = profile.droplevel(0).sort_index()
        # Si hay duplicados por el groupby, tomar el último
        profile = profile[~profile.index.duplicated(keep='last')]
        df[f"feat_{zone}_profile_hws"] = profile.reindex(df.index)

        # ── K2. Desviación respecto al perfil ──
        # Esto es lo que el modelo realmente predice: ¿cuánto se desvía hoy?
        df[f"feat_{zone}_deviation_from_profile"] = shifted_price - df[f"feat_{zone}_profile_hws"]

        # ── K3. Perfil simplificado por hora sola (más robusto, menos granular) ──
        profile_hour = shifted_price.groupby(hour_cet).expanding().mean()
        profile_hour = profile_hour.droplevel(0).sort_index()
        profile_hour = profile_hour[~profile_hour.index.duplicated(keep='last')]
        df[f"feat_{zone}_profile_hour"] = profile_hour.reindex(df.index)

        # ── K4. Ratio precio ayer vs perfil de ayer ──
        # Si ayer el precio estuvo 1.5x por encima del perfil, hoy probablemente también
        ratio = shifted_price / df[f"feat_{zone}_profile_hws"].replace(0, np.nan)
        df[f"feat_{zone}_profile_ratio_lag"] = ratio.clip(-5, 5)  # Clip extremos

        logger.info(f"    {zone}: perfil típico (hora×weekend×estación) + desviación OK")

    n_after = len([c for c in df.columns if c.startswith("feat_")])
    logger.info(f"    {n_after - n_before} features de perfil generadas")

    return df


def add_fingrid_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Features de FI a partir de las PREVISIONES day-ahead propias de Fingrid (el TSO
    finlandés), potencialmente más precisas que las de ENTSO-E. Todas son forecasts
    "updated once a day" (conocidas antes de la subasta) → leakage-safe, igual que
    las forecasts de ENTSO-E que ya usamos.

    Fuente: src/extract/09_fingrid_fi.py. Si las columnas no existen (no se corrió el
    extractor o no hay key Fingrid), la función no hace nada — degradación elegante.

    NOTA honestidad: estos datasets de Fingrid empiezan en fechas recientes, así que en
    el periodo de entrenamiento antiguo quedan NaN (train.py los rellena con la mediana).
    Su valor se mide después comparando MAE con/sin Fingrid; solo se conservan si ayudan.
    """
    load_fc  = df.get("fingrid_FI_load_fc_mw")
    wind_fc  = df.get("fingrid_FI_wind_fc_mw")
    solar_fc = df.get("fingrid_FI_solar_fc_mw")
    ol3_fc   = df.get("fingrid_FI_ol3_fc_mw")

    if load_fc is None and wind_fc is None and ol3_fc is None:
        logger.info("    (sin columnas Fingrid en el master — features Fingrid omitidas)")
        return df

    n_before = len([c for c in df.columns if c.startswith("feat_")])

    # F1. Residual load según Fingrid = carga − eólica − solar (driver del precio marginal)
    if load_fc is not None and wind_fc is not None:
        solar_v = solar_fc.fillna(0) if solar_fc is not None else 0
        df["feat_FI_fingrid_residual_load"] = load_fc - wind_fc.fillna(0) - solar_v
        df["feat_FI_fingrid_wind_penetration"] = wind_fc / load_fc.replace(0, np.nan)
        # Lag 24h + cambio respecto a ayer (mismo slot)
        df["feat_FI_fingrid_resid_load_lag24h"] = df["feat_FI_fingrid_residual_load"].shift(24)

    # F2. Previsiones crudas (la propia señal del TSO)
    if wind_fc is not None:
        df["feat_FI_fingrid_wind_fc"] = wind_fc
    if load_fc is not None:
        df["feat_FI_fingrid_load_fc"] = load_fc

    # F3. DESACUERDO de previsión: Fingrid vs ENTSO-E (carga). Señal de incertidumbre
    #     del sistema — cuando los dos TSO/fuentes no coinciden, suele haber más riesgo de
    #     desvío de precio. Ambas son day-ahead → sin leakage.
    entsoe_load_fc = next((c for c in df.columns if c.startswith("load_fc_FI_")), None)
    if load_fc is not None and entsoe_load_fc:
        df["feat_FI_loadfc_disagreement"] = load_fc - df[entsoe_load_fc]

    # F4. Olkiluoto-3 protection scheme forecast (señal específica del régimen nuclear)
    if ol3_fc is not None:
        df["feat_FI_fingrid_ol3_fc"] = ol3_fc

    n_after = len([c for c in df.columns if c.startswith("feat_")])
    cov = (load_fc.notna().mean() * 100) if load_fc is not None else float("nan")
    logger.info(f"    {n_after - n_before} features Fingrid FI generadas "
                f"(cobertura load_fc {cov:.0f}%)")
    return df


# ── Funcion principal ─────────────────────────────────────────────────────────

def generate_features(save: bool = True) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de feature engineering.

    Args:
        save: Si True, guarda el resultado en data/master/dataset_features.parquet

    Returns:
        DataFrame con todas las features generadas.
    """
    logger = get_logger("features")

    if not MASTER_PATH.exists():
        logger.error(
            f"Dataset maestro no encontrado: {MASTER_PATH}\n"
            "  -> Ejecuta primero el pipeline de Fase 1: python run_pipeline.py"
        )
        raise FileNotFoundError(str(MASTER_PATH))

    logger.info("=== FEATURE ENGINEERING — Fase 2 ===")
    logger.info(f"Input:  {MASTER_PATH}")
    logger.info(f"Output: {FEATURES_PATH}")

    df = pd.read_parquet(MASTER_PATH)
    df = df.sort_index()
    n_orig_cols = len(df.columns)
    logger.info(f"Dataset maestro: {df.shape[0]:,} filas x {n_orig_cols} columnas")

    # Ejecutar secciones en orden
    logger.info("\n[A] Features temporales ciclicas...")
    df = add_cyclical_features(df)
    logger.info(f"    8 features generadas (hour/dow/month/week sin+cos)")

    logger.info("\n[B] Features autoregresivas (lags de precio)...")
    df = add_lag_features(df, logger)

    logger.info("\n[C] Features de dominio fisico...")
    df = add_physical_features(df, logger)

    logger.info("\n[D] Features meteorologicas agregadas...")
    df = add_weather_features(df, logger)

    logger.info("\n[E] Features de mercado (spreads, NTC)...")
    df = add_market_features(df, logger)

    logger.info("\n[F] Features de interaccion...")
    df = add_interaction_features(df, logger)

    logger.info("\n[G] Calendario nordico completo...")
    df = add_holiday_features(df, logger)

    logger.info("\n[H] Features de commodities (gas TTF, CO2)...")
    df = add_commodity_features(df, logger)

    logger.info("\n[I] Features REMIT (indisponibilidad)...")
    df = add_remit_features(df, logger)

    logger.info("\n[J] Features hidro (embalses + hydro balance) + intercambios...")
    df = add_hydro_features(df, logger)

    logger.info("\n[J2] Features mercado acoplado Alemania (DE/LU)...")
    df = add_germany_features(df, logger)

    logger.info("\n[J3] Features mercado acoplado Finlandia (SE/EE + OL3)...")
    df = add_sweden_features(df, logger)

    logger.info("\n[J4] Features físicas Fingrid FI (previsiones propias del TSO)...")
    df = add_fingrid_features(df, logger)

    logger.info("\n[K] Perfil tipico de precio (hora x dia x estacion)...")
    df = add_price_profile_features(df, logger)

    # ── Resumen de features generadas ────────────────────────────────────
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    n_new     = len(df.columns) - n_orig_cols
    logger.info(f"\nTotal features generadas: {len(feat_cols)} (nuevas: {n_new})")
    logger.info(f"   Dataset final: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Diagnostico de nulos en features (los lags generan NaN en las primeras filas)
    feat_nulls = df[feat_cols].isnull().sum()
    max_nulls  = feat_nulls.max()
    logger.info(f"   Max nulos en features: {max_nulls} filas "
                f"(primeras ~{max_nulls} horas — perdida esperada por lags)")

    # Categorias de features
    categories = {
        "Temporales ciclicas": [c for c in feat_cols if any(x in c for x in
                                  ["hour", "dow", "month", "week", "sin", "cos"])],
        "Lags de precio":      [c for c in feat_cols if "lag" in c or "roll" in c or "diff" in c],
        "Dominio fisico":      [c for c in feat_cols if any(x in c for x in
                                  ["residual", "penetration", "wind_total", "wind_fc", "solar_fc"])],
        "Meteorologicas":      [c for c in feat_cols if c.startswith("feat_wx_")],
        "Mercado":             [c for c in feat_cols if any(x in c for x in
                                  ["spread", "congested", "ntc_util"])],
        "Interaccion":         [c for c in feat_cols if any(x in c for x in
                                  ["cold_low_wind", "morning_peak"])],
        "Calendario nordico":  [c for c in feat_cols if any(x in c for x in
                                  ["holiday", "easter", "christmas", "bridge"])],
        "Commodities":         [c for c in feat_cols if any(x in c for x in
                                  ["ttf", "co2", "gas_margin", "spark", "thermal"])],
        "REMIT":               [c for c in feat_cols if "unavail" in c],
        "Hidro/Embalses":      [c for c in feat_cols if "reservoir" in c],
        "Intercambios prog.":  [c for c in feat_cols if "sched" in c],
        "Alemania (acoplado)": [c for c in feat_cols if c.startswith("feat_de_") or "_de_spread" in c],
        "Finlandia (acoplado)": [c for c in feat_cols if "post_ol3" in c
                                  or any(f"feat_{z}_price_lag" in c for z in ("SE3","SE1","SE4","EE"))
                                  or "_spread_lag24h" in c and "FI_" in c
                                  or "sched_net_FI" in c],
        "Perfil tipico":       [c for c in feat_cols if any(x in c for x in
                                  ["profile_", "deviation_from_profile"])],
    }
    logger.info("\n  Features por categoria:")
    for cat, cols in categories.items():
        if cols:
            logger.info(f"    {cat:25s}: {len(cols):3d}")

    if save:
        df.to_parquet(FEATURES_PATH)
        size_mb = FEATURES_PATH.stat().st_size / 1e6
        logger.info(f"\n  Guardado: {FEATURES_PATH} ({size_mb:.1f} MB)")

    save_run_metadata(
        script_name="features",
        params={
            "target_zones":          TARGET_ZONES,
            "price_lags":            PRICE_LAGS,
            "rolling_windows":       ROLLING_WINDOWS,
            "congestion_threshold":  CONGESTION_THRESHOLD,
        },
        stats={
            "input_cols":   n_orig_cols,
            "feat_cols":    len(feat_cols),
            "total_cols":   len(df.columns),
            "rows":         len(df),
            "categories":   {k: len(v) for k, v in categories.items()},
        },
    )

    return df


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_features(save=True)