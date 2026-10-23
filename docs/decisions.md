# Registro de Decisiones Técnicas

**Proyecto:** Nordic day-ahead price forecast — Pipeline de predicción de precios Nord Pool  
**Última actualización:** junio de 2026  
**Formato:** Cada decisión documenta qué se decidió, qué alternativas había y por qué se eligió esa opción.

---

## DT-01 · Periodo de datos históricos

**Decisión:** Usar 2 años completos: enero 2023 – diciembre 2024 como dataset de entrenamiento principal.

**Alternativas consideradas:**
- **(A) 5 años (2020-2024):** Más datos, pero incluye la crisis energética de 2022 que distorsiona los patrones normales del mercado. Un modelo entrenado con esos precios extremos generalizaría mal hacia condiciones típicas.
- **(B) 1 año (2024):** Insuficiente para capturar dos ciclos estacionales completos. El modelo no aprendería la estacionalidad invierno/verano correctamente.
- **(C) 3 años (2022-2024):** Incluye 2022, que es anómalo por la crisis del gas ruso.

**Razonamiento:** 2023-2024 captura dos ciclos estacionales completos en régimen post-crisis (precios normalizados). El 2022 se descarga por separado (`*_2022_crisis.parquet`) para análisis de eventos extremos, pero no entra en el modelo principal.

---

## DT-02 · Granularidad temporal

**Decisión:** Resolución horaria como base. Opción de agregar a bloques de 4h o diarios para ciertos análisis del EDA.

**Alternativas consideradas:**
- **(A) 15 minutos:** Disponible en Nord Pool desde 2025 y parcialmente en Energinet. Histórico insuficiente para 2023-2024.
- **(B) Horaria:** Estándar histórico de la industria y de la literatura académica.
- **(C) Diaria:** Pierde información intradiaria clave (picos de mañana y tarde, horas de generación solar, precios negativos nocturnos).

**Razonamiento:** La resolución horaria es el estándar con el que opera la mayor parte de la industria y es la granularidad nativa de ENTSO-E y Energinet para el periodo objetivo. El sistema está diseñado para escalar a 15 minutos cuando el histórico lo permita.

---

## DT-03 · Zonas de precio incluidas

**Decisión:** Foco en DK1 y DK2 (zonas objetivo). Zonas complementarias: NO2, SE3, SE4, FI.

**Alternativas consideradas:**
- **(A) Solo DK1/DK2:** Perderíamos los precios de las zonas vecinas, que son las principales variables explicativas del spread danés.
- **(B) Todas las zonas nórdicas (NO1-5, SE1-4, FI, DK1-2):** Añade complejidad sin beneficio directo para la predicción de los precios daneses.

**Razonamiento:** NO2 es el principal proveedor de flexibilidad hidro hacia Dinamarca (cable Skagerrak). SE3/SE4 son las zonas suecas directamente conectadas. FI se incluye como zona vecina con buena disponibilidad de datos públicos. Las zonas NO1, NO3-5, SE1-2 se excluyen porque añaden volumen sin impacto directo en los precios daneses.

---

## DT-04 · Tratamiento de outliers y precios negativos

**Decisión:** NO eliminar precios negativos ni spikes extremos. Se marcan con flags binarias (`is_negative_price`, `is_spike_200`, `is_spike_500`) pero se conservan en el dataset.

**Alternativas consideradas:**
- **(A) Eliminar outliers:** Sesga el modelo hacia condiciones normales y le quita capacidad de predecir exactamente los eventos de mayor valor económico para un trader.
- **(B) Winsorizar (cap a percentil 99):** Distorsiona la distribución real del mercado.

**Razonamiento:** Los precios negativos ocurren cuando hay exceso de generación renovable — son un fenómeno estructural del mercado nórdico en crecimiento, no errores. Los spikes extremos son igualmente informativos. La solución correcta es que el modelo aprenda a predecirlos, no ignorarlos. Las flags binarias permiten al modelo tratar estos eventos como clase aparte si se desea.

---

## DT-05 · Tratamiento de valores faltantes

**Decisión:** Interpolación lineal para huecos de 1-3 horas. Forward fill para huecos de hasta 6 horas (3 + 3 adicionales). Huecos mayores de 6 horas se dejan como NaN.

**Alternativas consideradas:**
- **(A) Eliminar filas con NaN:** Rompe la continuidad temporal de la serie, introduce sesgos si los huecos no son aleatorios.
- **(B) Interpolación polinómica:** Overkill para huecos de 1-3h en series eléctricas. La interpolación lineal es suficiente y más estable.
- **(C) Imputación con modelo:** Innecesario para huecos tan pequeños y añade complejidad.

**Razonamiento:** Los huecos de 1-3h en ENTSO-E son típicamente fallos temporales de comunicación, no ausencias de datos reales. La interpolación lineal es una aproximación razonable. Huecos >6h son sospechosos (posible mantenimiento del sistema, problema de API) y no deben inventarse.

---

## DT-06 · Formato de almacenamiento intermedio

**Decisión:** Parquet como formato principal. CSV solo para muestra de inspección (primeras 2000 filas).

**Alternativas consideradas:**
- **(A) CSV:** Universal, legible con cualquier herramienta. Pero: no preserva tipos de datos (timestamps como strings, pérdida de timezone), 5-10x más pesado que Parquet, lectura lenta para datasets grandes.
- **(B) HDF5:** Buena performance pero requiere dependencias adicionales y tiene problemas de compatibilidad entre versiones.
- **(C) Feather:** Rápido pero no soporta todos los tipos de Pandas y menos estable a largo plazo.

**Razonamiento:** Parquet es el estándar de facto en data engineering moderno. Preserva tipos (timestamps con timezone, floats con precisión exacta), comprime 5-10x mejor que CSV y se lee en segundos con PyArrow. El dataset maestro completo (~50-100 MB en Parquet) cabría en memoria de cualquier portátil.

---

## DT-07 · Fuente primaria de precios para DK1/DK2

**Decisión:** Energinet como fuente primaria de precios para DK1 y DK2. ENTSO-E como fuente primaria para NO2, SE3, SE4, FI y como columna de validación cruzada para DK.

**Alternativas consideradas:**
- **(A) Solo ENTSO-E:** Una única fuente simplifica el pipeline pero pierde la granularidad y fiabilidad extra de Energinet para datos daneses.
- **(B) Solo Energinet:** No cubre las zonas no danesas.

**Razonamiento:** Energinet es el TSO danés — sus datos son la fuente oficial y de máxima fiabilidad para DK1/DK2. ENTSO-E agrega datos de todos los países de la UE con un ligero delay. Tener ambas fuentes permite validación cruzada: si difieren significativamente, indica un problema de datos en una de las fuentes.

---

## DT-08 · Conversión de zona horaria

**Decisión:** Todos los timestamps se almacenan en UTC explícito (tz-aware) en todo el pipeline. La conversión a hora local solo se hace en visualizaciones del EDA.

**Alternativas consideradas:**
- **(A) Hora local de cada zona (Europe/Copenhagen, etc.):** Intuitivo para interpretación, pero introduce ambigüedad en el cambio de hora DST (horas duplicadas en otoño, horas inexistentes en primavera).
- **(B) UTC naive (sin timezone):** Evita el DST pero es fuente de errores silenciosos al combinar fuentes de distintos países.

**Razonamiento:** UTC elimina la ambigüedad del horario de verano, que es el error más común en pipelines de series temporales eléctricas. El cambio de hora danés/nórdico ocurre en momentos distintos al español, lo que causaría alineaciones incorrectas si se mezclan zonas horarias locales. UTC tz-aware es el estándar de la industria para sistemas SCADA y trading.

---

## DT-09 · Generación negativa en datos ENTSO-E

**Decisión:** Valores negativos en columnas de generación se llevan a cero (`clip(lower=0)`).

**Razonamiento:** Los valores negativos en generación son artefactos de medición — en particular, el bombeo de las centrales hidroeléctricas se reporta por separado (como consumo) y no debe aparecer como generación negativa. No tiene significado físico en el contexto de este análisis.

---

## DT-10 · Paginación de la API de Energinet

**Decisión:** Descarga por bloques mensuales para datasets con alto volumen de registros.

**Problema:** La API de Energinet tiene un hard cap de 100.000 registros por petición. Para 2 años × 2 zonas × 24 horas × columnas múltiples, algunos datasets superan este límite.

**Razonamiento:** Bloques de 1 mes son seguros para todos los datasets actuales y futuros previsibles. La función `paginate_dataset()` detecta si la respuesta llega exactamente al límite y avisa para que se pueda reducir el tamaño del bloque.

---

## DT-11 · Variables meteorológicas y ubicaciones

**Decisión:** 7 ubicaciones estratégicas; `wind_speed_100m` como variable de viento principal; `timezone=UTC` solicitado directamente en la API Open-Meteo.

**Razonamiento de ubicaciones:** Las 7 ubicaciones representan los dos tipos de drivers meteorológicos: centros de generación eólica (Esbjerg, Aalborg, Kristiansand) y centros de demanda (Copenhagen, Oslo, Stockholm, Helsinki). Cubrir solo un tipo perdería información.

**Razonamiento de wind_speed_100m:** Las turbinas modernas offshore tienen bujes a 100-150m de altura. El viento a 10m subestima significativamente la velocidad real a la que trabajan los aerogeneradores daneses.

**Razonamiento de UTC en API:** Solicitar `timezone=UTC` directamente a Open-Meteo evita tener que hacer la conversión DST posterior, que es la fuente más común de errores en este tipo de pipelines (horas duplicadas/inexistentes en cambios de hora).

---

## DT-12 · Umbral de congestión en interconexiones

**Decisión:** Se considera que una interconexión está congestionada cuando el spread de precio entre zonas vecinas supera 10 EUR/MWh en valor absoluto.

**Alternativas consideradas:**
- **(A) 0 EUR/MWh:** Cualquier diferencia de precio indica congestión teórica, pero en la práctica pequeñas diferencias son ruido de mercado.
- **(B) 5 EUR/MWh:** Más sensible, pero captura muchos eventos de poca relevancia económica.
- **(C) 20 EUR/MWh:** Demasiado conservador, perdería congestiones moderadas relevantes.

**Razonamiento:** 10 EUR/MWh es el umbral habitual en la literatura académica de mercados nórdicos y en informes de análisis de Nord Pool. Por encima de este valor, la diferencia de precio tiene impacto económico claro para un trader o consumidor industrial.

---

## DT-13 · Auditoría de calidad y honestidad (revisión previa a la publicación)

Revisión completa del repositorio con foco en correctitud, ausencia de data
leakage y honestidad de los resultados mostrados. Cambios clave:

- **Leakage corregido:** los flags `is_congested_*` (derivados del spread de
  precio del MISMO timestamp) se excluyen del set de features en `train.py`.
  No se conocen a las 12:00 D-1; su versión causal `feat_congested_*` (lag 24h)
  ya existía. Verificado: máx. correlación feature↔precio mismo-timestamp = 0.64.
- **Meteo sin sesgo:** migración de Open-Meteo *Archive* (reanálisis ERA5 =
  observación) a *Historical Forecast* (la previsión disponible la víspera).
- **CO₂ EUA real:** se sustituye el proxy KRBN (ETF global en USD) por `CO2.L`
  (SparkChange Physical Carbon ETC, EUR/tCO₂, respaldo físico de EUAs).
  `thermal_marginal_cost` queda con unidades coherentes en EUR.
- **Conformal honesto:** la cobertura se mide *out-of-sample* (split del test en
  mitad de calibración / mitad de evaluación), no in-sample.
- **Métrica falsa eliminada:** el dashboard mostraba MAE 7.34 (valor MOCK). El
  MAE real es ~23 €/MWh (test, trimestre volátil) / ~17.6 (walk-forward). El
  dashboard ahora lee métricas reales de metadatos y muestra un banner que
  distingue datos reales de simulados.
- **Bugs corregidos:** claves `+AR_DA` en métricas de `evaluate.py`; guard de
  walk-forward vacío; `UnicodeEncodeError` en consola Windows (reconfiguración
  UTF-8 de stdout en `logger.py`); deprecaciones pandas 2.x (`.last`,
  `Styler.applymap`); KPI de MAE en `app.py`.
- **Unidad de viento:** documentada como km/h (default Open-Meteo), no m/s.

Resultado honesto (DK1, test Oct–Dec 2024): MAE 23.0 (ens+AR), DirAcc 77.5%,
+18% vs LEAR, +45% vs persistencia. Walk-forward MAE medio 17.6.

---

## DT-14 · Mejora del modelo basada en evidencia (estado del arte EPF)

**Contexto:** se buscó la mejora más robusta del MAE/métricas con datos gratuitos,
revisando el estado del arte en *electricity price forecasting* (Lago et al. 2021,
`epftoolbox`; Uniejewski, Weron & Ziel 2018 sobre transformaciones estabilizadoras
de varianza; Marcjasz & Weron sobre *averaging*).

**Método (medir antes de comprometer):** se montaron experimentos AISLADOS
(`scripts/experiment_*.py`) que comparan configuraciones en nuestro propio
dataset/split y, crucialmente, en **walk-forward 2024** (12 meses), no en un único
trimestre — para distinguir señal de ruido.

**Hallazgos:**
- **Transformación `asinh` (VST) DESCARTADA.** Aunque la literatura reporta +6–17%
  para modelos LINEALES (LEAR) y redes (NARX), en nuestro LightGBM **empeora** el
  MAE (−4 a −7%): los árboles ya son robustos a la escala del target, y comprimir
  los spikes hace que se infra-predigan (y los spikes dominan el MAE). Per-hora
  (24 modelos) y ventanas cortas también empeoran por falta de datos (18 meses).
- **Ensemble por DIVERSIDAD DE OBJETIVO adoptado.** Promediar 3 LightGBM con
  objetivos distintos (L2 + cuantil-0.5/mediana + Huber) mejora de forma robusta:
  **DirAcc +2.0 pp (mejor en 12/12 meses, ambas zonas)** y **MAE ~+1.5%**
  (8–10/12 meses). Sustituye al ensemble previo LightGBM+XGBoost+Ridge, que era
  redundante (corr LGBM↔XGB ≈ 0.99). Implementado en `src/model/diverse_ensemble.py`.

**Conclusión honesta:** no existe una mejora GRANDE de MAE alcanzable con datos
gratuitos en este periodo (2023-2024) excepcionalmente volátil; el error lo
dominan spikes intrínsecamente difíciles. La mejora robusta disponible es modesta
(direccionalidad) y se aplicó; el resto del MAE está cerca de su suelo. Documentar
esto — en vez de aplicar una transformación de moda que no ayuda — es la postura
profesional.

---

## DT-15 · Iteración de robustez (experimentos medidos + caso económico)

Continuando el principio de "medir antes de aplicar":

- **Feature pruning (top-K por importancia): NO aplicado.** En walk-forward, el
  mejor K (70 en DK1, 100 en DK2) mejora el MAE de forma marginal y ruidosa
  (13.40 vs 13.63 / 14.17 vs 14.41; curva plana). No justifica añadir un
  hiperparámetro "K". Se mantienen las 166 features. (`scripts/experiment_phase3.py`)
- **DNN (MLP de sklearn) como miembro diverso: RECHAZADO.** El MLP solo es mucho
  peor (MAE 34-35 vs ~19), y aunque es diverso (corr 0.70 con LightGBM), el
  promedio empeora; el test DM lo confirma (DM=+12/+15, p<0.001 → LightGBM solo
  significativamente mejor). Un DNN competitivo requeriría arquitectura/tuning
  serios, fuera de alcance.
- **Backtest de batería: dispatch por LP + corrección de la baseline naive.**
  Se detectó que la naive cargaba 4h en una batería de 2h (sobre-ciclado, comparación
  injusta). Corregido a respetar la capacidad. Con dispatch óptimo por LP
  (`scipy.optimize.linprog`), el **forecast bate a la naive en ~+21.000 €/año**
  (~+50%) y capta ~85% del oráculo. (Antes parecía que la naive ganaba: era el bug.)

- **Extensión a 2025-2026: código robusto listo, regeneración pendiente de
  ejecución supervisada.** Se descubrió que desde mediados de 2025 ENTSO-E publica
  precios y forecasts en resolución de 15 min (cambio de MTU europeo), con el rango
  por tanto MIXTO (horario + 15-min). Se añadió agregación automática a horario en
  `integration.safe_join` (no-op para series horarias). NO se ejecutó la
  regeneración completa de forma autónoma porque mezcla un cambio de esquema con
  rate-limits de Energinet y bastaría un fallo sutil para corromper el dataset
  verificado; se deja para una ejecución supervisada con validación cruzada.

Modelo de producción: **LightGBM único** (elegido por DM test). El ensemble
diverso, AR, conformal/ACI y el backtest de batería quedan construidos y evaluados,
pero solo se usan donde la evidencia lo respalda.

---

## DT-16 · Extensión del periodo a 2023-2026 (test reciente)

Se amplió la ventana de datos de 2023-2024 a **2023-01 → 2026-05** (29.928 h).

- **Reto (cambio de esquema):** desde 2025 ENTSO-E publica precios y forecasts en
  **15 minutos** (nuevo MTU europeo), con el rango por tanto MIXTO (horario hasta
  ~mediados-2025, 15-min después). La guarda de resampleo detecta sub-horario por
  **horas duplicadas** (`index.floor("h").has_duplicates`) — no por la mediana del
  paso, que fallaba en series mixtas — y agrega a horario (media) en extracción y
  en `integration.safe_join`. Validado: fichero mixto 47.431 filas → 29.929 h sin
  huecos ni duplicados.
- **Splits:** train 2023→Sep-2025, val Q4-2025, **test Ene–May 2026** (out-of-sample
  reciente; elimina el problema del test envejecido).
- **Gates de validación (todos OK):** grid horario completo (0 huecos/duplicados),
  precios DK1/DK2 sin nulos, máx. correlación feature↔precio mismo-ts = 0.81
  (forecast alemán, sin leakage), lags causales.
- **Resultado (test 2026):** MAE DK1 **14.8** / DK2 **17.1**; DirAcc **82%**;
  R² **0.81**/0.75; **+34% vs LEAR** (DK1 34%, DK2 35%, LEAR zone-aware — ver DT-19);
  batería capta **92-93%** del oráculo. Mejora real y, sobre todo, evaluación **actual**
  con 3.4 años de entrenamiento.
- **Limitaciones de la extracción:** la NTC day-ahead y el balance físico de
  Energinet quedaron parciales (algunos errores/rate-limits); son features menores
  que solapan con generación ENTSO-E (completa) → NaN tolerado. Documentado.

Trabajo hecho en una rama aparte con gates + auto-revert disponible;
fusionado a la rama principal tras pasar validación.

---

## DT-17 · Track separado de Finlandia (mercado FI)

Se añadió **Finlandia (zona FI)** como producto separado junto al de Dinamarca, con
un **toggle de país DK ↔ FI** en el dashboard, centrado en el mercado finlandés.
DK queda **intacto** (cambios estrictamente aditivos).

- **Por qué FI es distinto de DK:** el driver dominante no es Alemania sino **Suecia
  (SE1/SE3/SE4) y Estonia (EstLink/EE)**, más la **disponibilidad nuclear de Olkiluoto
  (OL3)**, que entró en operación comercial el **2023-04-16** y bajó el precio medio FI
  ~75% — un **quiebre estructural** (flag `feat_post_ol3`). FI es el mercado más volátil
  de Europa y el #1 en horas de precio negativo.
- **Datos nuevos (gratis):** extractor `08_finland_coupling.py` añade aditivamente al
  master (token ENTSO-E existente + Open-Meteo sin clave): **price_SE1** (norte barato
  30,5 €/MWh) y **price_EE** (Estonia cara 87,4), **flujos programados EstLink** FI-EE y
  FI-SE3/SE1, y meteo de **Vaasa / Olkiluoto / Pori** (hubs eólicos del oeste). +26
  columnas, 99-100% cobertura. El precio DA viene de ENTSO-E (Fingrid no lo publica).
- **Features (anti-leakage):** `add_sweden_features` emite precios de acoplamiento
  laggeados (SE1/SE3/SE4/EE a 24/48/168h — misma subasta EUPHEMIA, nunca contemporáneos),
  spreads FI-SEx/FI-EE, flujos netos EstLink, e interacción nuclear-out × bajo viento.
  215 features totales (+50 FI); DK byte-estable (verificado).
- **Selección de modelo por evidencia (Diebold-Mariano):** al revés que DK (donde gana
  LightGBM), en FI el **ensemble diverso** (L2+mediana+Huber) bate a LightGBM
  **significativamente** (MAE 21,4 vs 24,6; DM stat 7,0; p<0,0001) → `predict.py`
  usa `ENSEMBLE_ZONES={FI}`. El dashboard muestra el MAE del modelo **realmente servido**
  (21,4), no un sub-modelo interno optimista.
- **Métricas honestas (test ene–may 2026, 3.624 h):** MAE **21,4** | DirAcc **73,0%** |
  R² **0,71** | walk-forward 21,1. Baselines: Persistence 34,4 · LEAR (en su momento 33,8 →
  **corregido a 28,7** con LEAR zone-aware, ver DT-19). Bate a LEAR en todos los regímenes;
  cifras finales en DT-19. Conformal ACI 48,5%→80,1%. **MAE ~21 es honesto** (FI es el
  mercado más volátil de Europa), no inflado.
- **Siguiente paso (ver DT-18):** integrar Fingrid (TSO finlandés) para físicos FI
  autoritativos.

---

## DT-18 · Fingrid (TSO finlandés) — físicos day-ahead autoritativos para FI

Se integró **Fingrid Open Data** (el operador del sistema finlandés), dando a FI su
TSO nacional igual que DK tiene **Energinet**. Decisión guiada por un
criterio del proyecto: *el valor no es solo si baja el MAE, sino si la fuente es más
robusta y profesional*. Fingrid cumple ambas.

- **Datos (gratis):** `src/extract/09_fingrid_fi.py` añade aditivamente al master las
  **previsiones day-ahead propias de Fingrid** ("updated once a day" → publicadas antes
  de la subasta, leakage-safe): carga (165), eólica (246), solar (247) y el forecast del
  esquema de protección de **Olkiluoto-3** (372). Cobertura 99,9-100% desde 2023 (OL3 40%,
  dataset reciente). Suscripción gratuita "Open Data starter" → `FINGRID_API_KEY` en `.env`.
- **Robustez del extractor:** la paginación del API de Fingrid es inconsistente (capa a
  10k ptos, `nextPage` no fiable). Solución determinista: **trocear en ventanas de 30
  días** (≤8.640 ptos a 5-min < tope) + reintento exponencial ante 429. Agregado 5/15-min
  → horario.
- **Features (`add_fingrid_features`):** residual load propio del TSO, penetración eólica,
  lag 24h, y **desacuerdo de previsión Fingrid-vs-ENTSO-E** (señal de incertidumbre del
  sistema). Todas day-ahead → sin leakage. **224 features** (FI; DK 166).
- **Resultado — doble lectura honesta:**
  - **MAE servido FI: 21,4 → 21,05** (mejor, no solo neutro); vs LEAR **−7,6** (LEAR
    zone-aware, ver DT-19); todos los regímenes mejoran (negativos 4,7→3,3; frío 29,1→28,6).
  - **Autoridad de fuente confirmada empíricamente:** `feat_FI_fingrid_residual_load` es
    la **feature #1 de TODO el modelo FI**, por encima del residual load de ENTSO-E (#2).
    El desacuerdo Fingrid-vs-ENTSO-E entra en el top-25. El modelo *prefiere* la fuente
    del TSO → la elección profesional es además la más predictiva.
  - **Criterio de decisión:** mantener si MAE neutro-o-mejor + aporta robustez/autoridad;
    quitar solo si empeora o mete leakage. Aquí gana en ambas → se conserva.
- **DK intacto:** las features Fingrid son FI-específicas; los modelos DK quedan congelados.

---

## DT-19 · Auditoría exhaustiva + LEAR zone-aware (corrección de honestidad)

Auditoría del proyecto (investigación del mercado finlandés + revisión de código:
leakage, consistencia, rigor ML). **Conclusión principal: no hay leakage ni "trampas"** —
el método se verificó sólido (walk-forward gold-standard, conformal out-of-sample, LEAR
estándar de Lago et al., DM test, auditoría anti-leakage). Pero se encontró **un sesgo de
baseline** que se corrigió:

- **Bug:** `LEARBaseline` usaba **viento y temperatura de DK1 hardcodeados para todas las
  zonas** (`baselines.py` y `data_loader.load_lear_forecast`). El LEAR de FI usaba viento
  de DK1 (zona equivocada), debilitando artificialmente el baseline y **inflando la ventaja
  del modelo en FI**.
- **Fix:** `LEARBaseline(zone=...)` — cada zona usa su propio `feat_{zona}_wind_total_fc` y
  `wx_{zona}_temp_avg` (+ warning si falta viento). Aplicado en baselines y en el dashboard.
- **Impacto honesto (FI):** LEAR **33,8 → 28,7** (DirAcc 69→70,5; R² 0,50→0,64). La ventaja
  real del modelo FI pasa de **+37% a +27% vs LEAR** (MAE 21,05 vs 28,7). Sigue batiendo a
  LEAR en **todos** los regímenes: OL3-out 13,0 vs 28,3 · negativos 3,3 vs 5,5 · frío 28,6
  vs 32,6 €/MWh. DK apenas cambia: DK1 +34% (LEAR 22,5), DK2 +35% (LEAR 25,5→26,3).
- **Consistencia:** corregidas inconsistencias de documentación detectadas (test 2024→2026,
  conteo de features 166 DK / 224 FI, DirAcc 72,95%, % vs LEAR) para que ningún número se
  contradiga y cualquier lector pueda comprobar cada cifra.
- **Principio:** preferible un baseline MÁS fuerte y una ventaja MENOR pero **real**, que un
  número bonito sobre un rival de paja. La credibilidad vale más que el decimal.

---

## DT-20 · Battery Lab: revenue stacking multi-mercado con baterías comerciales

Se elevó el backtest de batería de "arbitraje DA con batería genérica" a un
**laboratorio de economía de flexibilidad multi-mercado** con productos
comerciales reales. Sustentado en una investigación verificada con fuentes
primarias (jun-2026).

**Qué se añadió:**
- `src/model/battery_specs.py` — librería de BESS comerciales (Tesla Megapack
  2 XL, Wärtsilä Quantum2, Fluence Gridstack Pro, Sungrow PowerTitan 2.0, CATL
  TENER, genérico-FI) con specs de **datasheet** (potencia/energía/RTE) y campos
  no publicados marcados en `estimated_fields` con rangos de literatura (Preger
  et al. 2020 para ciclos LFP; BNEF 2025 para CAPEX: Europa ~177 $/kWh, con
  datapoints reales FI 163-258 €/kWh; NREL para O&M). El claim "cero
  degradación" de CATL se trata como sobredimensionamiento (lectura de la
  industria), no como física.
- `src/model/revenue_stacking.py` — LP diario que co-optimiza FCR-N + FCR-D
  up/down + arbitraje DA con las **reglas LER reales** del área síncrona
  nórdica (requisitos técnicos FCR 14.5.2025): FCR-N exige potencia ±1,34×C y
  energía 1h×C por dirección; FCR-D, 0,2×C opuesta y ⅓h×C. Coste marginal de
  degradación en el objetivo (convención Xu et al.; 7-17 €/MWh). Dos modos:
  **oracle** (techo, reservas con precio realizado) y **causal** (puja con
  forecast naive lag-24h, acepta solo si la puja queda bajo el marginal real,
  cobra el marginal — la secuencia informacional de los gates reales: DA 12:00
  D-1 → FCR horario 18:30 D-1 → precios FCR publicados ~22:45).
- Extractores de precios de reserva REALIZADOS de TSO: `10_fingrid_balancing.py`
  (FI, ya existente, ahora registrado como paso 12) y `11_energinet_reserves.py`
  (DK: FcrNdDK2, AfrrReservesNordic, mFRRCapacityMarket, balancing/imbalance —
  Energi Data Service, sin clave, CC-BY 4.0), paso 13.
- `pages/5_Battery_Lab.py` — página nueva del dashboard.

**Validación contra la realidad (la prueba de honestidad):** el techo oracle
anual en FI reproduce los benchmarks públicos sin calibración alguna:
- 2023: ~457k €/MW brutos vs **488k €/MW/año reales** del activo de Exilion
  optimizado por Capalo AI (H2-2023, reportado) — y el desplome posterior:
  2024 ~404k → 2025 ~216k → **2026 YTD ~86k €/MW**, convergiendo al rango del
  European BESS Index (70-130k €/MW/año 2025-26).
- El propio dato de mercado lo confirma: Fingrid compró **0 MW de FCR-N en la
  subasta anual de 2026** (>1.000 MW de baterías instaladas en FI, feb-2026,
  saturaron la reserva). En nuestros datos el precio horario FCR-N cae de 47
  €/MW,h (2023) a 8,6 (2026 YTD).

**Decisiones de honestidad explícitas:**
- aFRR/mFRR/intradía NO se modelan (reglas de activación complejas; los precios
  realizados SÍ están en el master para análisis). El stack mostrado es por
  tanto un **suelo conservador** para 2025-26 — se dice en la página.
- El throughput de regulación FCR-N (0,2 MWh/MW·h por dirección) es una
  **estimación documentada**, no dato del TSO; añade pérdidas y desgaste, y se
  ignora conservadoramente el ingreso por energía de activación FCR-N (que en
  realidad se liquida a max(spot, imbalance) a favor del proveedor).
- Price-taker: una batería de 1-5 MW no mueve estos mercados; la canibalización
  de la flota completa es endógena en los precios realizados contra los que se
  liquida (es exactamente la historia 2023→2026 que muestra la página).
- En el periodo de test (ene-may 2026), con el forecast real del modelo FI:
  stack causal ~66k €/MW brutos (87% del techo oracle), vs ~31k solo-DA.
  Con CAPEX 2026 el payback a precios de reserva actuales supera los 10 años —
  coherente con el giro del mercado real a tolling/floors (Statkraft-OX2,
  Luxcara) y a aFRR/mFRR. **No se maquilla: un entrante 2026 no recupera el
  CAPEX solo con FCR+DA.**

> Note (2026-10-23): the Battery Lab figures in this entry were computed before the fix described in DT-23 and are superseded there.

---

## DT-21 · ¿Se pueden predecir los precios FCR? — experimento del gap de literatura

La revisión de literatura (jun-2026) detectó que **no existe ningún paper
revisado que pronostique precios FCR-N/FCR-D nórdicos** (solo estudios
alemanes de pay-as-bid). Se montó `scripts/experiment_fcr_forecast.py` con el
split del proyecto (train 2023→sep-2025, test ene–may-2026), features legítimas
en el gate FCR de las 18:30 D-1 (precio DA del día D — publicado a las 13:00
D-1 —, previsiones del TSO, lags ≥24h del propio precio FCR) y baseline =
naive lag-24h (el que usa el modo causal del Battery Lab).

**Resultados (MAE test, €/MW,h):**
- **Formulación en NIVEL: fracaso catastrófico** (FI FCR-N 10,70 vs naive 3,79;
  −182% a −447%). Causa: colapso de régimen — el modelo aprende niveles de
  2023-24 (media 46-47) y el test 2026 está en 8,6. Lección de manual sobre
  no-estacionariedad: en un mercado en colapso, el nivel histórico desinforma.
- **Formulación en DESVIACIÓN (target = precio − lag24) + ventana corta:**
  recupera la paridad pero solo bate al naive en **2/5 targets**: DK2 FCR-D up
  **+8,1%**, DK2 FCR-N +0,8%; en FI queda en empate técnico (−0,4 a −0,5%) y
  FI FCR-D up pierde (−38%).

**Conclusiones honestas:**
1. En el régimen post-saturación, los precios FCR finlandeses son casi un
   paseo aleatorio a niveles bajos — **el naive lag-24h es el forecast de puja
   correcto**, lo que valida empíricamente el diseño del modo causal del
   Battery Lab (no era una simplificación, es el estado del arte alcanzable).
2. DK2 (mercado menos saturado, precios más altos) sí muestra predictibilidad
   modesta — un upgrade futuro del bidding DK2 con la variante C es razonable.
3. Es un resultado con potencial publicable (nadie lo ha medido en el área
   nórdica): "FCR price forecastability under battery-driven regime collapse".

Resultados completos en `data/metadata/experiment_fcr_results.json`.

---

## DT-22 · Observatorio de transparencia

El análisis del mercado (jun-2026) identificó varios huecos que no cubre ningún
recurso abierto; el proyecto ya cubría tres (Battery Lab con datos reales de
Fingrid, estudio de predictibilidad FCR y benchmark zonal con protocolo
publicado). Esta decisión añade el observatorio de transparencia. También se
exploraron features flow-based, que quedaron fuera de los modelos servidos.

**Observatorio de transparencia — `pages/6_Transparency.py`.**
Ningún vendor publica la precisión de sus forecasts nórdicos (Volue/Montel:
cero MAE públicos) y los proyectos abiertos populares no publican protocolo
out-of-sample. La página publica el nuestro **calculado en vivo al renderizar**
desde los artefactos reales (sin números precalculados que puedan desincronizarse):
- MAE/rMAE del modelo servido vs naive-24h (computado del precio real) y vs
  LEAR zone-aware, con media±σ de la ventana al lado (sin contexto un MAE no
  significa nada).
- **Test de Diebold-Mariano en vivo** (varianza HAC + corrección HLN — el
  estándar de Lago et al. 2021) modelo-vs-naive y modelo-vs-LEAR.
- Walk-forward mensual con rMAE mensual (persistencia mensual computada del
  master) y la σ de precio de cada mes como contexto.
- Desglose por regímenes (spikes >150, negativos, normal) con los tres
  competidores en cada régimen.
- Cobertura empírica P10-P90 del test y las "reglas del marcador" (todas las
  limitaciones impresas).
Verificación de coherencia: la página computa en vivo DK1 MAE 14,55 / rMAE
0,55 / LEAR 22,39 / ventana 97±46 — consistente con los valores documentados
del pipeline (14,5/0,54/22,5/96,6±46,7; las décimas difieren por el cómputo
independiente, que es exactamente el propósito).

---

## DT-23 · Battery linear programs: power limit and round-trip efficiency (2026-10-23)

A review of the code before publication found four defects, now fixed. The first two are covered by `tests/test_battery_lp.py`, the third by `tests/test_lear_zone.py` and the fourth by `tests/test_energinet_dedup.py`:

- **Power limit.** The daily linear programs in `src/model/battery_backtest.py` and `src/model/revenue_stacking.py` allowed charge and discharge in the same hour above the power rating. Both now enforce charge plus discharge at most the power rating in every hour.
- **Round-trip efficiency applied twice.** The Battery Lab program (`revenue_stacking.py`) applied the efficiency in the state-of-charge balance and again on the day-ahead revenue, which understated day-ahead arbitrage. It is now applied once, as the square root of the round-trip efficiency on charge and on discharge.
- **LEAR in `evaluate.py`.** The evaluation script still built its LEAR row with DK1 weather (the DT-19 fix had reached the baselines and the dashboard only). It now passes the zone; its LEAR row matches `results/baselines_test2026.json`.
- **Energi Data Service duplicates.** The datasets `Elspotprices`, `ElectricityBalanceNonv` and `ProductionConsumptionSettlement` hold one row per zone (`PriceArea`) and hour. `process_and_save` in `src/extract/04_energinet_dk.py`, and `remove_duplicates` in `src/transform/cleaning.py`, dropped duplicates by timestamp alone, so only one zone survived each hour. In the June 2026 run that zone was DK2: `Elspotprices` kept 24,094 DK2 rows, `price_DK1` came from ENTSO-E for the whole period and `price_DK2` from Energi Data Service up to 30 September 2025 (ENTSO-E after that). Over those hours the two sources agree on DK2 to within 0.0001 €/MWh. The Danish balance columns kept 744 hours (DK1) and 1 hour (DK2), and the settlement columns DK2 only; none of them is a model feature (the models read only `feat_*` columns and the calendar flags `is_weekend` and `is_holiday*`; none of the `feat_*` columns is built from the Energinet balance or settlement data), so the published forecasts do not change. Both functions now deduplicate by timestamp and `PriceArea`.

Effect: the day-ahead backtest barely moves (share of perfect foresight 92.6% DK1, 92.1% DK2, 74.9% FI). The Battery Lab figures change: perfect-foresight ceiling in Finland 461.1 / 408.8 / 223.1 / 96.4 thousand EUR per MW (2023 to 2026), and in the 2026 test window the causal stack earns 71.2 thousand EUR per MW gross, 88% of the 81.2 oracle, against 39.1 for day-ahead arbitrage alone. All current figures are in `results/battery_lab_fi.json` and `results/evaluation_test2026.json`; they replace the ones in DT-20.

---

*Documento vivo — se actualiza conforme avanza el proyecto.*
