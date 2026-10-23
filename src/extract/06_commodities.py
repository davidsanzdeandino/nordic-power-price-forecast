"""
06_commodities.py
=================
Descarga precios de commodities energéticos — gas TTF y CO2 EUA.

DECISIÓN TÉCNICA: El precio del gas natural (TTF) y los derechos de
emisión de CO2 (EUA) determinan el coste marginal de generación térmica.
Cuando las renovables no cubren la demanda, el precio eléctrico se fija
por el coste de la central marginal (típicamente gas), que depende
directamente de TTF y CO2:
    Precio_marginal ≈ TTF / eficiencia_gas + CO2 * factor_emisión

Sin estas variables, el modelo es ciego al "suelo" del precio eléctrico
en horas de baja renovable. Impacto estimado: -5-10% MAE.

Fuentes (todas gratuitas vía Yahoo Finance):
  - Gas TTF (Dutch Title Transfer Facility): Yahoo Finance (TTF=F) — EUR/MWh
  - CO2 EUA (EU Emission Trading System): Yahoo Finance (CO2.L) — EUR/tCO2
      CO2.L = "SparkChange Physical Carbon ETC", un ETC respaldado FÍSICAMENTE
      por derechos de emisión EUA del EU ETS y cotizado en EUR. Su precio sigue
      de cerca al EUA spot/futuro, por lo que es un proxy directo (en EUR/tCO2),
      a diferencia de un ETF de carbono global en USD.
    + fallback a descarga manual (Investing.com / Ember) si Yahoo falla.

Granularidad: Diaria (se resamplea a horario en integración)
Periodo: 2023-01-01 → 2024-12-31

USO DE LOS DATOS: los precios de Yahoo Finance se descargan con yfinance solo
para uso personal de investigación. Los ficheros generados (data/raw/commodity_*)
y las columnas derivadas NO se incluyen en este repositorio ni se redistribuyen;
cada usuario ejecuta este script en local.

USO DE LOS DATOS: los precios de Yahoo Finance se descargan con yfinance solo
para uso personal de investigación. Los ficheros generados (data/raw/commodity_*)
y las columnas derivadas NO se incluyen en este repositorio ni se redistribuyen;
cada usuario ejecuta este script en local.

NOTA: yfinance es gratuito pero puede fallar ocasionalmente.
Si falla, el script genera instrucciones para descarga manual.

Dependencias:
    pip install yfinance

Uso:
    python src/extract/06_commodities.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.logger import get_logger, save_run_metadata

# ── Configuración ─────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2023-01-01"
END_DATE   = "2026-06-01"  # Exclusivo — cubre hasta may-2026

# Tickers de Yahoo Finance
TICKERS = {
    "ttf_gas": {
        "symbol":      "TTF=F",
        "description": "Dutch TTF Natural Gas Front-Month Futures (EUR/MWh)",
        "column":      "ttf_eur_mwh",
        "unit":        "EUR/MWh",
    },
    "co2_eua": {
        "symbol":      "CO2.L",
        "description": "SparkChange Physical Carbon ETC (EUA del EU ETS, respaldo físico, EUR/tCO2)",
        "column":      "co2_eua_eur",
        "unit":        "EUR/tCO2",
        # CO2.L está respaldado FÍSICAMENTE por derechos EUA y cotiza en EUR,
        # así que su precio es un proxy directo del EUA (no un ETF global en USD).
    },
}

# Tickers alternativos si los principales fallan.
# OJO: se evitan fallbacks con unidades/mercado distintos (p.ej. NG=F Henry Hub
# en USD/MMBtu o materias primas descorrelacionadas), que contaminarían el
# dataset. Si el ticker principal falla, mejor usar el CSV manual (ver abajo).
FALLBACK_TICKERS = {
    "ttf_gas": [],
    "co2_eua": ["KEUA"],   # KraneShares European Carbon (sigue EUA; en USD — se avisa)
}


# ── Descarga ──────────────────────────────────────────────────────────────────

def download_commodity(name: str, config: dict, logger) -> pd.DataFrame:
    """
    Descarga serie diaria de un commodity desde Yahoo Finance.

    Returns:
        DataFrame con índice DatetimeIndex UTC y columna de precio.
        DataFrame vacío si falla.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "Módulo 'yfinance' no instalado.\n"
            "  → pip install yfinance"
        )
        return pd.DataFrame()

    symbol = config["symbol"]
    logger.info(f"    Descargando {symbol} ({config['description']})...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")

        if df.empty:
            logger.warning(f"    WARN {symbol}: sin datos. Intentando fallback...")
            # Intentar tickers alternativos
            for alt_symbol in FALLBACK_TICKERS.get(name, []):
                logger.info(f"    Probando {alt_symbol}...")
                ticker = yf.Ticker(alt_symbol)
                df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")
                if not df.empty:
                    logger.info(f"    OK Usando fallback {alt_symbol}")
                    break

        if df.empty:
            logger.error(f"    ERROR {name}: no se pudieron obtener datos")
            return pd.DataFrame()

        # Normalizar
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp_utc"

        # Usar precio de cierre ajustado
        col_name = config["column"]
        result = df[["Close"]].rename(columns={"Close": col_name})

        # Eliminar fines de semana y festivos (mercados cerrados) → se interpolan después
        result = result[result[col_name].notna()]

        logger.info(f"    OK {name}: {len(result)} días | "
                    f"{result[col_name].min():.2f} – {result[col_name].max():.2f} {config['unit']}")

        return result

    except Exception as e:
        logger.error(f"    ERROR {name}: error descargando {symbol}: {e}")
        return pd.DataFrame()


def create_manual_instructions(logger):
    """
    Si yfinance falla, genera instrucciones para descarga manual.
    """
    logger.info("\n" + "="*70)
    logger.info("INSTRUCCIONES PARA DESCARGA MANUAL (si yfinance falla)")
    logger.info("="*70)
    logger.info("""
    GAS TTF:
    1. Ir a: https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data
    2. Seleccionar periodo: 01/01/2023 → 31/12/2024
    3. Descargar CSV → renombrar a: data/raw/manual_ttf_gas.csv
    4. El CSV debe tener columnas: Date, Price (o Close)

    CO2 EUA:
    1. Ir a: https://www.investing.com/commodities/carbon-emissions-historical-data
    2. Seleccionar periodo: 01/01/2023 → 31/12/2024
    3. Descargar CSV → renombrar a: data/raw/manual_co2_eua.csv
    4. El CSV debe tener columnas: Date, Price (o Close)

    Alternativa CO2 (más precisa):
    1. Ir a: https://ember-climate.org/data/data-tools/carbon-price-viewer/
    2. Descargar datos EU ETS
    """)


def load_manual_csv(name: str, config: dict, logger) -> pd.DataFrame:
    """
    Carga datos de un CSV descargado manualmente como fallback.
    """
    csv_path = OUTPUT_DIR / f"manual_{name}.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    logger.info(f"    Cargando CSV manual: {csv_path}")
    try:
        df = pd.read_csv(csv_path)

        # Detectar columna de fecha
        date_col = next((c for c in df.columns if c.lower() in
                         ["date", "fecha", "timestamp", "time"]), df.columns[0])
        # Detectar columna de precio
        price_col = next((c for c in df.columns if c.lower() in
                          ["close", "price", "precio", "adj close"]), df.columns[1])

        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index(date_col)
        df.index.name = "timestamp_utc"

        col_name = config["column"]
        result = df[[price_col]].rename(columns={price_col: col_name})

        # Limpiar: algunos CSVs de Investing.com usan comas como decimal
        if result[col_name].dtype == object:
            result[col_name] = result[col_name].str.replace(",", "").astype(float)

        result = result[result[col_name].notna()]

        logger.info(f"    OK {name} (manual): {len(result)} días")
        return result

    except Exception as e:
        logger.error(f"    ERROR leyendo CSV manual {name}: {e}")
        return pd.DataFrame()


# ── Procesamiento ─────────────────────────────────────────────────────────────

def resample_to_hourly(df: pd.DataFrame, col_name: str, logger) -> pd.DataFrame:
    """
    Convierte serie diaria a horaria con interpolación.

    DECISIÓN TÉCNICA: Los precios de commodities son diarios (mercados
    financieros), pero el modelo eléctrico es horario. Usamos forward-fill
    (el precio del día aplica a todas sus horas) y luego interpolamos
    los huecos de fines de semana/festivos.
    """
    if df.empty:
        return df

    # Crear índice horario completo
    hourly_idx = pd.date_range(
        start=df.index.min().normalize(),
        end=df.index.max().normalize() + pd.Timedelta(hours=23),
        freq="h",
        tz="UTC"
    )

    # Reindexar a horario y forward-fill (precio del día aplica todo el día)
    df_hourly = df.reindex(hourly_idx)
    df_hourly[col_name] = df_hourly[col_name].ffill()

    # Interpolar huecos restantes (fines de semana: viernes → lunes)
    df_hourly[col_name] = df_hourly[col_name].interpolate(method="linear", limit=96)

    df_hourly.index.name = "timestamp_utc"

    n_null = df_hourly[col_name].isna().sum()
    if n_null > 0:
        logger.warning(f"    WARN {col_name}: {n_null} horas sin dato tras interpolación")

    return df_hourly


# ── Función principal ─────────────────────────────────────────────────────────

def extract_commodities() -> dict:
    """
    Descarga commodities energéticos y los guarda en data/raw/.

    Returns:
        Dict con stats {commodity: n_filas | 'ERROR' | 'MANUAL_NEEDED'}
    """
    logger = get_logger("commodities")
    stats = {}

    logger.info(f"=== Commodities Energéticos | {START_DATE} → {END_DATE} ===")
    logger.info(f"Commodities: {list(TICKERS.keys())}")

    all_dfs = {}

    for name, config in TICKERS.items():
        logger.info(f"\n{name}: {config['description']}")

        # Intentar yfinance
        df = download_commodity(name, config, logger)

        # Fallback a CSV manual
        if df.empty:
            df = load_manual_csv(name, config, logger)

        if df.empty:
            logger.warning(f"  WARN {name}: sin datos disponibles")
            stats[name] = "MANUAL_NEEDED"
            continue

        # Resamplear a horario
        col_name = config["column"]
        df_hourly = resample_to_hourly(df, col_name, logger)

        # Guardar diario (raw) y horario (para integración)
        daily_path = OUTPUT_DIR / f"commodity_{name}_daily.parquet"
        hourly_path = OUTPUT_DIR / f"commodity_{name}_hourly.parquet"

        df.to_parquet(daily_path)
        df_hourly.to_parquet(hourly_path)

        stats[name] = len(df_hourly)
        all_dfs[name] = df_hourly

        logger.info(f"  OK {daily_path.name} | {len(df)} días")
        logger.info(f"  OK {hourly_path.name} | {len(df_hourly)} horas")

    # ── Crear features derivadas de commodities ──────────────────────────
    # Si tenemos ambos, calcular spark spread y dark spread
    if "ttf_gas" in all_dfs and not all_dfs["ttf_gas"].empty:
        ttf = all_dfs["ttf_gas"]

        # Eficiencia típica de planta de gas CCGT: ~55%
        # Factor emisión gas: ~0.37 tCO2/MWh térmico
        GAS_EFFICIENCY = 0.55
        GAS_EMISSION_FACTOR = 0.37  # tCO2/MWh

        # Coste marginal del gas (EUR/MWh eléctrico)
        ttf_col = TICKERS["ttf_gas"]["column"]
        ttf["gas_marginal_cost_eur_mwh"] = ttf[ttf_col] / GAS_EFFICIENCY

        # Si tenemos CO2, añadir el coste de emisiones
        if "co2_eua" in all_dfs and not all_dfs["co2_eua"].empty:
            co2 = all_dfs["co2_eua"]
            co2_col = TICKERS["co2_eua"]["column"]

            # Merge TTF + CO2
            combined = ttf.join(co2, how="outer").ffill()

            # Coste total marginal = gas/eficiencia + CO2 * emisiones/eficiencia
            combined["thermal_marginal_cost"] = (
                combined[ttf_col] / GAS_EFFICIENCY +
                combined[co2_col] * GAS_EMISSION_FACTOR / GAS_EFFICIENCY
            )

            combined_path = OUTPUT_DIR / f"commodity_combined_hourly.parquet"
            combined.to_parquet(combined_path)
            logger.info(f"\n  OK {combined_path.name} | "
                        f"{len(combined)} horas | cols: {list(combined.columns)}")
        else:
            combined_path = OUTPUT_DIR / f"commodity_combined_hourly.parquet"
            ttf.to_parquet(combined_path)

    # Instrucciones si falta algo
    if any(v == "MANUAL_NEEDED" for v in stats.values()):
        create_manual_instructions(logger)

    # Guardar metadatos
    save_run_metadata(
        script_name="commodities",
        params={
            "start": START_DATE,
            "end":   END_DATE,
            "tickers": {k: v["symbol"] for k, v in TICKERS.items()},
        },
        stats=stats,
    )

    ok  = sum(1 for v in stats.values() if isinstance(v, int))
    err = sum(1 for v in stats.values() if v in ("ERROR", "MANUAL_NEEDED"))
    logger.info(f"\nResumen: {ok}/{len(TICKERS)} commodities OK | {err} pendientes")

    return stats


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("ERROR: falta 'yfinance'. Ejecuta: pip install yfinance")
        print("   (Alternativamente, descarga CSVs manualmente — ver instrucciones)")

    extract_commodities()