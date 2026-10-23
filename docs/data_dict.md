# Diccionario de Datos — Dataset Maestro

**Archivo:** `data/master/dataset_master.parquet`  
**Índice:** `timestamp_utc` — DatetimeIndex horario, UTC tz-aware  
**Periodo:** 2023-01-01 00:00 UTC → 2026-05-31 23:00 UTC  
**Volumen real:** 29.928 filas × ~270 columnas (incluye precios, generación,
forecast, demanda, flujos, NTC, meteo, Energinet, commodities, REMIT y derivadas).  
**Generado por:** `src/transform/integration.py`

---

## Índice

1. [Precio day-ahead](#1-precio-day-ahead)
2. [Generación real por tecnología](#2-generación-real-por-tecnología)
3. [Previsión DA viento y solar](#3-previsión-da-viento-y-solar)
4. [Demanda](#4-demanda)
5. [Flujos físicos en interconexiones](#5-flujos-físicos-en-interconexiones)
6. [Net Transfer Capacity (NTC)](#6-net-transfer-capacity-ntc)
7. [Meteorología](#7-meteorología)
8. [Datos Energinet](#8-datos-energinet)
9. [Features de calendario](#9-features-de-calendario)
10. [Features derivadas](#10-features-derivadas)
11. [Flags de precio](#11-flags-de-precio)
12. [Track Finlandia — acoplamiento FI + Fingrid](#12-track-finlandia--acoplamiento-fi--fingrid)
13. [Precios de mercados de reservas/balancing (Battery Lab)](#13-precios-de-mercados-de-reservasbalancing-battery-lab)

---

## 1. Precio day-ahead

Fuente primaria DK1/DK2: **Energinet**. Fuente NO2/SE3/SE4/FI: **ENTSO-E**.  
Columnas de validación cruzada ENTSO-E para DK disponibles bajo prefijo `price_entsoe_*`.
Nota (23-oct-2026): en la ejecución de junio de 2026 la deduplicación por timestamp dejó solo DK2 en los datos de Energinet, así que `price_DK1` sale entero de ENTSO-E y `price_DK2` de Energinet hasta el 30-sep-2025 y de ENTSO-E después (arreglado; ver DT-23 en `docs/decisions.md`).

| Columna | Tipo | Unidad | Fuente | Descripción |
|---------|------|--------|--------|-------------|
| `price_DK1` | float | EUR/MWh | Energinet / ENTSO-E | Precio spot horario zona DK1 (Jutlandia + Funen) |
| `price_DK2` | float | EUR/MWh | Energinet / ENTSO-E | Precio spot horario zona DK2 (Zelanda + islas) |
| `price_NO2` | float | EUR/MWh | ENTSO-E | Precio spot zona Noruega Sur |
| `price_SE3` | float | EUR/MWh | ENTSO-E | Precio spot zona Suecia Central |
| `price_SE4` | float | EUR/MWh | ENTSO-E | Precio spot zona Suecia Sur |
| `price_FI`  | float | EUR/MWh | ENTSO-E | Precio spot zona Finlandia |
| `price_SE1` | float | EUR/MWh | ENTSO-E | Precio spot Suecia Norte (acoplamiento FI — excedente hidro/eólico barato) |
| `price_EE`  | float | EUR/MWh | ENTSO-E | Precio spot Estonia (acoplamiento FI vía EstLink) |
| `price_entsoe_DK1` | float | EUR/MWh | ENTSO-E | Precio DK1 según ENTSO-E (validación cruzada) |
| `price_entsoe_DK2` | float | EUR/MWh | ENTSO-E | Precio DK2 según ENTSO-E (validación cruzada) |

**Valores esperados:** Rango típico -50 a 500 EUR/MWh en 2023-2024. Valores negativos posibles (exceso renovables). Ver flags en sección 11.  
**Nulos:** Huecos rellenos hasta 6h (ver DT-05). Huecos >6h quedan como NaN.

---

## 2. Generación real por tecnología

Fuente: **ENTSO-E** (`query_generation`). Prefijo: `gen_<ZONA>_`.  
Las columnas exactas dependen de las tecnologías reportadas por cada TSO — varían por zona.

| Patrón de columna | Tipo | Unidad | Descripción |
|-------------------|------|--------|-------------|
| `gen_DK1_Wind Onshore_Actual Aggregated` | float | MW | Generación eólica onshore DK1 real |
| `gen_DK1_Wind Offshore_Actual Aggregated` | float | MW | Generación eólica offshore DK1 real |
| `gen_DK1_Solar_Actual Aggregated` | float | MW | Generación solar DK1 real |
| `gen_DK2_*` | float | MW | Mismo esquema para DK2 |
| `gen_NO2_*` | float | MW | Generación NO2 (principalmente hidro) |
| `gen_SE3_*` | float | MW | Generación SE3 (mix nuclear + hidro + eólico) |
| `gen_SE4_*` | float | MW | Generación SE4 |
| `gen_FI_*`  | float | MW | Generación FI (nuclear Olkiluoto relevante) |

**Nota:** Los nombres exactos de columna incluyen el tipo de tecnología tal como lo reporta ENTSO-E. Consultar las columnas reales con `df.filter(like='gen_').columns` tras generar el dataset.  
**Tratamiento:** Valores negativos llevados a 0 (artefactos de medición, ver DT-09).

---

## 3. Previsión DA viento y solar

Fuente: **ENTSO-E** (`query_wind_and_solar_forecast`). Prefijo: `forecast_<ZONA>_`.  
**Estas son las variables más importantes para el modelo predictivo** — son las que se conocen en el momento de la subasta del mercado day-ahead.

| Patrón de columna | Tipo | Unidad | Descripción |
|-------------------|------|--------|-------------|
| `forecast_DK1_Wind Onshore` | float | MW | Previsión eólica onshore DK1 (day-ahead) |
| `forecast_DK1_Wind Offshore` | float | MW | Previsión eólica offshore DK1 (day-ahead) |
| `forecast_DK1_Solar` | float | MW | Previsión solar DK1 (day-ahead) |
| `forecast_DK2_*` | float | MW | Mismo esquema para DK2 |
| `forecast_NO2_*` | float | MW | Previsión NO2 |
| `forecast_SE3_*` | float | MW | Previsión SE3 |
| `forecast_SE4_*` | float | MW | Previsión SE4 |
| `forecast_FI_*`  | float | MW | Previsión FI |

**Nota sobre data leakage:** Usar siempre el forecast (no la generación real) como input del modelo. La generación real no se conoce en el momento de la subasta day-ahead.

---

## 4. Demanda

Fuente: **ENTSO-E** (`query_load`, `query_load_forecast`).

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `load_DK1_load_mw` | float | MW | Demanda real DK1 |
| `load_DK2_load_mw` | float | MW | Demanda real DK2 |
| `load_NO2_load_mw` | float | MW | Demanda real NO2 |
| `load_SE3_load_mw` | float | MW | Demanda real SE3 |
| `load_SE4_load_mw` | float | MW | Demanda real SE4 |
| `load_FI_load_mw`  | float | MW | Demanda real FI |
| `load_fc_DK1_load_forecast_mw` | float | MW | Previsión DA demanda DK1 |
| `load_fc_DK2_load_forecast_mw` | float | MW | Previsión DA demanda DK2 |
| `load_fc_*` | float | MW | Previsión DA demanda por zona |

---

## 5. Flujos físicos en interconexiones

Fuente: **ENTSO-E** (`query_crossborder_flows`). Prefijo: `flow_<ORIGEN><DESTINO>_`.  
Descargados en ambas direcciones — el flujo neto se calcula como diferencia.

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `flow_DK1DK2_flow_mw` | float | MW | Flujo DK1 → DK2 (Great Belt) |
| `flow_DK2DK1_flow_mw` | float | MW | Flujo DK2 → DK1 (inverso) |
| `flow_DK1NO2_flow_mw` | float | MW | Flujo DK1 → NO2 (Skagerrak) |
| `flow_NO2DK1_flow_mw` | float | MW | Flujo NO2 → DK1 (inverso) |
| `flow_DK1DELU_flow_mw` | float | MW | Flujo DK1 → Alemania |
| `flow_DELU DK1_flow_mw` | float | MW | Flujo Alemania → DK1 |
| `flow_DK2SE4_flow_mw` | float | MW | Flujo DK2 → SE4 |
| `flow_SE4DK2_flow_mw` | float | MW | Flujo SE4 → DK2 |
| `flow_DK2DELU_flow_mw` | float | MW | Flujo DK2 → Alemania |
| `flow_DELUDK2_flow_mw` | float | MW | Flujo Alemania → DK2 |

**Valores positivos:** Flujo en la dirección indicada. **Flujo neto:** `flow_ABA - flow_BAB`.

---

## 6. Net Transfer Capacity (NTC)

Fuente: **ENTSO-E** (`query_net_transfer_capacity_dayahead`). Prefijo: `ntc_<ORIGEN><DESTINO>_`.

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `ntc_DK1DK2_ntc_mw` | float | MW | Capacidad disponible DK1→DK2 |
| `ntc_DK2DK1_ntc_mw` | float | MW | Capacidad disponible DK2→DK1 |
| `ntc_DK1NO2_ntc_mw` | float | MW | Capacidad disponible DK1→NO2 (Skagerrak) |
| `ntc_NO2DK1_ntc_mw` | float | MW | Capacidad disponible NO2→DK1 |
| `ntc_DK1DELU_ntc_mw` | float | MW | Capacidad disponible DK1→DE |
| `ntc_DK2SE4_ntc_mw` | float | MW | Capacidad disponible DK2→SE4 |

**Interpretación:** Cuando `flow / ntc → 1`, la interconexión está saturada y el precio diverge entre zonas.

---

## 7. Meteorología

Fuente: **Open-Meteo Historical Forecast API** (la *previsión* que existía la
víspera, no el reanálisis ERA5 — evita leakage meteorológico). Prefijo: `wx_<Ubicacion>_`.  
7 ubicaciones × 6 variables = 42 columnas meteorológicas.

### Ubicaciones

| Prefijo | Zona | Descripción |
|---------|------|-------------|
| `wx_EsbjergDK1_` | DK1 | Eólica offshore Mar del Norte |
| `wx_AalborgDK1_` | DK1 | Eólica onshore Jutlandia Norte |
| `wx_CopenhagenDK2_` | DK2 | Centro de demanda DK2 |
| `wx_KristiansandNO2_` | NO2 | Hub interconexión hidro |
| `wx_OsloNO1_` | NO1 | Demanda noruega |
| `wx_StockholmSE3_` | SE3 | Demanda sueca |
| `wx_HelsinkiFI_` | FI | Demanda finlandesa |

### Variables por ubicación

| Sufijo | Tipo | Unidad | Descripción |
|--------|------|--------|-------------|
| `*_temperature_2m` | float | °C | Temperatura a 2m — proxy demanda calefacción |
| `*_precipitation` | float | mm | Precipitación — proxy estado embalses (largo plazo) |
| `*_wind_speed_10m` | float | km/h | Velocidad viento a 10m (referencia) |
| `*_wind_speed_100m` | float | km/h | Velocidad viento a 100m — altura turbinas modernas |
| `*_wind_direction_100m` | float | ° | Dirección viento a 100m (0=Norte, 90=Este) |
| `*_solar_radiation` | float | W/m² | Radiación solar global — proxy generación solar |

**Unidad de viento:** km/h (unidad por defecto de Open-Meteo). Para m/s, dividir
entre 3.6. El modelo es invariante a la escala (árboles), así que se mantiene km/h.  
**Valores esperados:** wind_speed_100m: 0-110 km/h (≈0-30 m/s) | temperature_2m: -20 a 35°C | solar_radiation: 0 a 900 W/m²  
**Nulos:** Open-Meteo es muy fiable (<0.1% nulos típico). Rellenados por interpolación en cleaning.

---

## 8. Datos Energinet

Fuente: **Energinet DataService** (TSO danés).

### 8.1 Precios validados

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `price_energinet_dk1` | float | EUR/MWh | Precio spot DK1 según Energinet (fuente primaria) |
| `price_energinet_dk2` | float | EUR/MWh | Precio spot DK2 según Energinet (fuente primaria) |

### 8.2 Balance físico (ElectricityBalanceNonv)

Prefijo: `DK1_` o `DK2_`. Columnas exactas dependen del dataset de Energinet.  
Incluyen: producción eólica, solar, térmica, consumo total, intercambios con vecinos.

### 8.3 Liquidación (ProductionConsumptionSettlement)

Prefijo: `DK1_settlement_` o `DK2_settlement_`.  
Datos de liquidación final — mayor precisión que el balance provisional.

---

## 9. Features de calendario

Derivadas del índice `timestamp_utc`. No requieren fuente externa.

| Columna | Tipo | Rango | Descripción |
|---------|------|-------|-------------|
| `hour_of_day` | int | 0–23 | Hora del día (UTC) |
| `day_of_week` | int | 0–6 | Día de la semana (0=lunes, 6=domingo) |
| `month` | int | 1–12 | Mes del año |
| `quarter` | int | 1–4 | Trimestre |
| `week_of_year` | int | 1–53 | Semana ISO del año |
| `is_weekend` | int | 0/1 | 1 si sábado o domingo |
| `is_holiday_dk` | int | 0/1 | 1 si festivo danés fijo (Navidad, Año Nuevo, etc.) |

**Nota:** Las versiones cíclicas seno/coseno (`feat_hour_sin`, `feat_hour_cos`, etc.) se generan en `features.py` (Fase 2) para uso en el modelo. Aquí se mantienen los enteros para interpretabilidad en el EDA.

---

## 10. Features derivadas

Calculadas en el paso 11 de `integration.py`.

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `spread_DK1_DK2` | float | EUR/MWh | Diferencia precio DK1 - DK2. Positivo → DK1 más caro |
| `spread_DK1_NO2` | float | EUR/MWh | Diferencia precio DK1 - NO2. Indica saturación Skagerrak |
| `spread_DK2_SE4` | float | EUR/MWh | Diferencia precio DK2 - SE4. Indica saturación Great Belt Norte |
| `spread_SE3_SE4` | float | EUR/MWh | Diferencia precio SE3 - SE4. Referencia sistema sueco |
| `is_congested_DK1_DK2` | int | 0/1 | 1 si \|spread_DK1_DK2\| > 10 EUR/MWh (ver DT-12) |
| `is_congested_DK1_NO2` | int | 0/1 | 1 si \|spread_DK1_NO2\| > 10 EUR/MWh |
| `is_congested_DK2_SE4` | int | 0/1 | 1 si \|spread_DK2_SE4\| > 10 EUR/MWh |
| `is_congested_SE3_SE4` | int | 0/1 | 1 si \|spread_SE3_SE4\| > 10 EUR/MWh |

---

## 11. Flags de precio

Generadas en `cleaning.py` a partir de los precios DK1/DK2. Prefijo de zona.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `DK1_is_negative_price` | int | 1 si precio DK1 < 0 EUR/MWh |
| `DK1_is_spike_200` | int | 1 si precio DK1 > 200 EUR/MWh |
| `DK1_is_spike_500` | int | 1 si precio DK1 > 500 EUR/MWh (crisis extrema) |
| `DK2_is_negative_price` | int | 1 si precio DK2 < 0 EUR/MWh |
| `DK2_is_spike_200` | int | 1 si precio DK2 > 200 EUR/MWh |
| `DK2_is_spike_500` | int | 1 si precio DK2 > 500 EUR/MWh |

**Frecuencia esperada (2023-2024):** negativos ~5-15% de horas | spikes>200 ~1-3% | spikes>500 <0.5%.

---

## 12. Track Finlandia — acoplamiento FI + Fingrid

Columnas añadidas para el producto finlandés (DK queda intacto). Generadas por
`src/extract/08_finland_coupling.py` (acoplamiento) y `src/extract/09_fingrid_fi.py`
(Fingrid). Todos los precios/forecasts de acoplamiento se usan **laggeados ≥24h**
en features (misma subasta EUPHEMIA → anti-leakage). Ver `docs/decisions.md` DT-17/DT-18.

### 12a. Acoplamiento de mercado (extractor 08)

| Columna | Tipo | Unidad | Fuente | Descripción |
|---------|------|--------|--------|-------------|
| `price_SE1`, `price_EE` | float | EUR/MWh | ENTSO-E | Precios de acoplamiento (ver sección 1) |
| `sched_FIEE_mw` / `sched_EEFI_mw` | float | MW | ENTSO-E | Intercambio programado FI→EE / EE→FI (EstLink) |
| `sched_FISE_3_mw` / `sched_SE_3FI_mw` | float | MW | ENTSO-E | Intercambio programado FI↔SE3 |
| `sched_FISE_1_mw` / `sched_SE_1FI_mw` | float | MW | ENTSO-E | Intercambio programado FI↔SE1 |
| `wx_VaasaFI_*` | float | varias | Open-Meteo | Meteo Vaasa (nodo eólico onshore de la costa oeste) — mismas variables que sección 7 |
| `wx_OlkiluotoFI_*` | float | varias | Open-Meteo | Meteo Olkiluoto (nuclear + eólica costera) |
| `wx_PoriFI_*` | float | varias | Open-Meteo | Meteo Pori (eólica oeste) |

### 12b. Fingrid — previsiones day-ahead del TSO finlandés (extractor 09)

Previsiones "updated once a day" (publicadas antes de la subasta → leakage-safe).
Cobertura 99,9-100% desde 2023, salvo OL3 (~40%, dataset reciente).

| Columna | Tipo | Unidad | Fuente | Descripción |
|---------|------|--------|--------|-------------|
| `fingrid_FI_load_fc_mw` | float | MW | Fingrid (165) | Previsión DA de consumo/carga FI |
| `fingrid_FI_wind_fc_mw` | float | MW | Fingrid (246) | Previsión DA de eólica FI |
| `fingrid_FI_solar_fc_mw` | float | MW | Fingrid (247) | Previsión DA de solar FI |
| `fingrid_FI_ol3_fc_mw` | float | MW | Fingrid (372) | Forecast del esquema de protección de Olkiluoto-3 |

### 12c. Features FI derivadas (en `dataset_features.parquet`)

| Feature | Descripción |
|---------|-------------|
| `feat_FI_fingrid_residual_load` | Residual load según Fingrid (carga−eólica−solar). **Feature #1 del modelo FI** |
| `feat_FI_loadfc_disagreement` | Desacuerdo de previsión de carga Fingrid vs ENTSO-E (incertidumbre del sistema) |
| `feat_FI_SE3_spread_lag24h`, `feat_FI_EE_spread_lag24h` | Spreads de acoplamiento laggeados FI-SE3 / FI-EE |
| `feat_sched_net_FI_EE`, `feat_sched_net_FI_SE_3` | Flujo neto programado FI↔EE / FI↔SE3 |
| `feat_post_ol3` | Flag de régimen post-Olkiluoto-3 (quiebre estructural 2023-04-16) |

---

## 13. Precios de mercados de reservas/balancing (Battery Lab)

Liquidación ex-post realizada; como features SOLO laggeados ≥24h.
Fuente FI: Fingrid Open Data (mercado horario, precio marginal, €/MW,h).
Fuente DK: Energi Data Service (sin clave, CC-BY 4.0). Cobertura 2023-01 → 2026-05.

| Columna | Descripción |
|---|---|
| `fingrid_FI_fcrn_price` | FCR-N FI, mercado horario (dataset 317) |
| `fingrid_FI_fcrd_up_price` / `fingrid_FI_fcrd_down_price` | FCR-D up/down FI (318 / 283) |
| `fingrid_FI_afrr_up_price` | aFRR capacidad up FI (52) |
| `fingrid_FI_mfrr_up_price` | mFRR capacidad up FI (329) |
| `fingrid_FI_imbalance_price` | Precio de desbalance FI (319; 15-min desde 19-mar-2025, agregado a hora) |
| `fingrid_FI_balup_price` / `fingrid_FI_baldown_price` | Energía de balancing up/down FI (244 / 106) |
| `energinet_DK2_fcrn_price` | FCR-N DK2, subasta nórdica, tipo 'Total' (FcrNdDK2) |
| `energinet_DK2_fcrd_up_price` / `energinet_DK2_fcrd_down_price` | FCR-D up/down DK2 (FcrNdDK2) |
| `energinet_{DK1,DK2,FI}_afrr_{up,down}_price` | aFRR capacidad (AfrrReservesNordic). La columna FI valida en cruzado el dataset 52 de Fingrid (corr 1,000; mediana de diferencia 0,00 €; se comprueba con `checks/test_dk2_stack.py`) |
| `energinet_{DK1,DK2}_mfrr_{up,down}_price` | mFRR capacidad DK (mFRRCapacityMarket) |
| `energinet_{DK1,DK2}_balup_price` / `..._baldown_price` | Energía de balancing DK (RegulatingBalancePowerdata, hasta mar-2025) |
| `energinet_{DK1,DK2}_imbalance_price` | Desbalance DK (histórico horario + 15-min 2025+ combinados) |

Consumidos por `src/model/revenue_stacking.py` (Battery Lab,
`pages/5_Battery_Lab.py`) como precios de liquidación realizados — nunca como
inputs contemporáneos del forecast.

---

## Notas generales

- **Índice:** Siempre UTC tz-aware. Para convertir a hora local danesa: `df.index.tz_convert('Europe/Copenhagen')`.
- **Nulos residuales:** Huecos >6h no rellenados quedan como `NaN`. Antes de entrenar el modelo, aplicar `dropna(subset=['price_DK1', 'price_DK2'])` como mínimo.
- **Acceso rápido a columnas por categoría:**
  ```python
  price_cols    = [c for c in df.columns if c.startswith('price_')]
  forecast_cols = [c for c in df.columns if c.startswith('forecast_')]
  wx_cols       = [c for c in df.columns if c.startswith('wx_')]
  calendar_cols = ['hour_of_day','day_of_week','month','is_weekend','is_holiday_dk']
  spread_cols   = [c for c in df.columns if 'spread' in c]
  ```
