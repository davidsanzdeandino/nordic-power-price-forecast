"""
run_pipeline.py
===============
Orquestador del pipeline completo — Fase 1 (ETL) + Fase 2 (Modelo).

Ejecuta en orden todas las etapas del proyecto. Todas las fuentes de datos
son PÚBLICAS y GRATUITAS (ENTSO-E Transparency, Open-Meteo, Energinet,
Fingrid, Yahoo Finance). ENTSO-E y Fingrid requieren una clave gratuita
(ver .env.example).

Etapas:
  Extracción:
    1. ENTSO-E — Precios day-ahead
    2. ENTSO-E — Datos físicos (generación, carga, flujos, NTC)
    3. Open-Meteo — Meteorología
    4. Energinet — Balance y settlement de Dinamarca
    5. ENTSO-E — Indisponibilidades REMIT (mantenimientos)
    6. Yahoo Finance — Commodities (gas TTF, CO2 EUA en EUR/tCO2)
    7. ENTSO-E — Embalses hidro (semanal) + intercambios programados day-ahead
  Transformación:
    8. Limpieza raw -> clean
    9. Integración -> dataset maestro
  Augmentación del master (aditiva):
   10. Acoplamiento FI — SE1/EE + EstLink + meteo Vaasa
   11. Fingrid — previsiones day-ahead propias (load/wind/solar + OL3)
   12. Fingrid — precios de reservas/balancing FI (FCR-N/D, aFRR, mFRR, imbalance)
   13. Energinet — precios de reservas DK (FCR-N/D DK2, aFRR, mFRR, imbalance)
  Modelo:
   14. Feature engineering -> dataset_features
   15-17. Baselines (persistencia, media semanal, LEAR zone-aware) DK1/DK2/FI
   18-20. Entrenamiento DK1/DK2/FI con los hiperparámetros congelados de
          src/model/best_params.json: elección LightGBM/ensemble, AR day-ahead
          y conformal en validación (oct-dic 2025), cuantiles, walk-forward
          -> results/training_<zona>.json
   21-23. Evaluación en el test DK1/DK2/FI -> results/evaluation_test2026_<zona>.json
   24-26. Verificación de predicción day-ahead (smoke test) DK1/DK2/FI
  Métricas del modelo servido:
   27-29. DK1/DK2/FI (scripts/gen_served_metrics.py)
          -> data/metadata/served_metrics_<zona>.json
   30. Regímenes de FI (scripts/evaluate_fi_regimes.py)
          -> results/fi_regimes_test2026.json

NOTA: los pasos de entrenamiento (train_zone) producen internamente los
modelos quantile, el calibrador conformal, el corrector AR y el ensemble.
No son pasos separados del pipeline. Los pasos 27-30 lanzan un script de
scripts/ en un subproceso (función None en STEPS).

Uso:
  python run_pipeline.py                 # Pipeline completo (parámetros congelados)
  python run_pipeline.py --optuna        # Vuelve a buscar hiperparámetros con Optuna (lento)
  python run_pipeline.py --step 3        # Solo el paso 3
  python run_pipeline.py --from 14       # Desde feature engineering en adelante
  python run_pipeline.py --from 14 --until 20
  python run_pipeline.py --zone DK1      # Limita los pasos por zona a DK1
"""

import argparse
import subprocess
import time
import sys
import os
from datetime import datetime

# Consola UTF-8 (Windows usa cp1252 al redirigir; los símbolos del log, como
# € o →, romperían print()). Mismo criterio que src/utils/logger.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Añadir la raíz del proyecto al path (los módulos hacen `from src...`)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_script(module_path: str, argv: list) -> None:
    """Lanza un script de scripts/ en un subproceso (leen sys.argv, no kwargs)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    sys.stdout.flush()   # que la cabecera del paso salga antes que el subproceso
    subprocess.run([sys.executable, os.path.join(ROOT, module_path), *argv],
                   cwd=ROOT, env=env, check=True)


def served_metrics_blocker(zone: str, results: list):
    """Motivo para no generar served_metrics de una zona, o None si se puede.

    Exige el dataset de features y el modelo entrenado de la zona, y que en esta
    ejecución no haya fallado ningún paso de modelo (14 en adelante) de esa zona
    o común a todas.
    """
    step_zone = {s[0]: s[5] for s in STEPS}
    failed = [sn for sn, _, ok in results
              if not ok and sn >= 14 and step_zone.get(sn) in (None, zone)]
    if failed:
        return f"fallaron pasos anteriores ({', '.join(map(str, failed))})"
    needed = [os.path.join(ROOT, "data", "master", "dataset_features.parquet"),
              os.path.join(ROOT, "data", "models", f"lightgbm_{zone}.pkl")]
    missing = [os.path.relpath(p, ROOT) for p in needed if not os.path.exists(p)]
    if missing:
        return f"faltan {', '.join(missing)} (entrena primero el modelo)"
    return None


def run_step(step_num: int, name: str, module_path: str,
             func_name, kwargs: dict) -> bool:
    """Ejecuta un paso del pipeline con timing y manejo de errores.

    Si func_name es None, el paso es un script que se lanza en un subproceso
    con los argumentos de kwargs["argv"].
    """
    print(f"\n{'='*60}")
    print(f"PASO {step_num}: {name}")
    if kwargs:
        print(f"  args: {kwargs}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        if func_name is None:
            run_script(module_path, kwargs.get("argv", []))
            elapsed = time.time() - t0
            print(f"\nOK Paso {step_num} completado en {elapsed:.1f}s")
            return True
        # Import dinámico por ruta (los extractores empiezan por dígito y
        # no se pueden importar con `import` normal).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"pipeline_step_{step_num}", module_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        func = getattr(mod, func_name)
        func(**kwargs)
        elapsed = time.time() - t0
        print(f"\nOK Paso {step_num} completado en {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\nERROR Paso {step_num} FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


# (num, nombre, ruta_módulo, función, kwargs, zona|None)
STEPS = [
    # ── Fase 1: Extracción (fuentes públicas) ─────────────────────────────────
    (1,  "Extracción ENTSO-E — Precios Day-Ahead",            "src/extract/01_entsoe_prices.py",        "extract_prices",      {},                None),
    (2,  "Extracción ENTSO-E — Físicos (gen/carga/flujos/NTC)","src/extract/02_entsoe_physical.py",      "extract_physical",    {},                None),
    (3,  "Extracción Open-Meteo — Meteorología",              "src/extract/03_openmeteo_weather.py",    "extract_weather",     {},                None),
    (4,  "Extracción Energinet — Datos Dinamarca",            "src/extract/04_energinet_dk.py",         "extract_energinet",   {},                None),
    (5,  "Extracción ENTSO-E — Indisponibilidades REMIT",     "src/extract/05_remit_unavailabilities.py","extract_remit",      {},                None),
    (6,  "Extracción Commodities — Gas TTF + CO2 (Yahoo Fin.)","src/extract/06_commodities.py",          "extract_commodities", {},                None),
    (7,  "Extracción ENTSO-E — Embalses hidro + intercambios programados","src/extract/07_entsoe_hydro_exchanges.py","extract_hydro_exchanges",{},          None),
    # ── Fase 1: Transformación ────────────────────────────────────────────────
    (8,  "Limpieza de datos raw -> clean",                    "src/transform/cleaning.py",              "run_cleaning",        {},                None),
    (9,  "Integración -> dataset maestro",                    "src/transform/integration.py",           "load_and_merge",      {},                None),
    # Acoplamiento FI: AUGMENTA el master (price_SE1/EE + flujos EstLink + meteo Vaasa),
    # por eso va DESPUÉS de integración y ANTES de features. DK queda intacto.
    (10, "Acoplamiento FI — SE1/EE + EstLink + meteo Vaasa (ENTSO-E/Open-Meteo)", "src/extract/08_finland_coupling.py", "main", {},          None),
    # Fingrid (TSO finlandés): previsiones day-ahead propias (carga/eólica/solar + OL3).
    # También AUGMENTA el master. Requiere FINGRID_API_KEY; si no, el paso falla limpio.
    (11, "Físicos Fingrid FI — previsiones load/wind/solar + OL3 (Fingrid Open Data)", "src/extract/09_fingrid_fi.py", "main", {},           None),
    # Reservas/balancing FI (FCR-N/D, aFRR, mFRR, imbalance): precios para el módulo
    # de revenue stacking y features de estrés laggeadas. AUGMENTA el master.
    (12, "Reservas/balancing Fingrid FI — FCR-N/D, aFRR, mFRR, imbalance (Fingrid Open Data)", "src/extract/10_fingrid_balancing.py", "main", {},  None),
    # Reservas DK (Energi Data Service, sin clave): FCR-N/D de DK2 (mercado nórdico),
    # aFRR/mFRR DK1+DK2, balancing/imbalance. Para el Battery Lab DK. AUGMENTA el master.
    (13, "Reservas Energinet DK — FCR-N/D DK2, aFRR, mFRR, imbalance (Energi Data Service)", "src/extract/11_energinet_reserves.py", "main", {},  None),
    # ── Fase 2: Modelo predictivo ─────────────────────────────────────────────
    (14, "Feature Engineering -> dataset_features",           "src/model/features.py",                  "generate_features",   {},                None),
    (15, "Baselines (persistencia, media semanal, LEAR) DK1", "src/model/baselines.py",                 "run_all_baselines",   {"zone": "DK1"},   "DK1"),
    (16, "Baselines (persistencia, media semanal, LEAR) DK2", "src/model/baselines.py",                 "run_all_baselines",   {"zone": "DK2"},   "DK2"),
    (17, "Baselines (persistencia, media semanal, LEAR) FI",  "src/model/baselines.py",                 "run_all_baselines",   {"zone": "FI"},    "FI"),
    (18, "Entrenamiento DK1 (elección, AR y conformal en validación)", "src/model/train.py",            "train_zone",          {"zone": "DK1"},   "DK1"),
    (19, "Entrenamiento DK2 (elección, AR y conformal en validación)", "src/model/train.py",            "train_zone",          {"zone": "DK2"},   "DK2"),
    (20, "Entrenamiento FI (elección, AR y conformal en validación)",  "src/model/train.py",            "train_zone",          {"zone": "FI"},    "FI"),
    (21, "Evaluación completa DK1",                           "src/model/evaluate.py",                  "evaluate_zone",       {"zone": "DK1"},   "DK1"),
    (22, "Evaluación completa DK2",                           "src/model/evaluate.py",                  "evaluate_zone",       {"zone": "DK2"},   "DK2"),
    (23, "Evaluación completa FI",                            "src/model/evaluate.py",                  "evaluate_zone",       {"zone": "FI"},    "FI"),
    (24, "Verificación de predicción day-ahead DK1",          "src/model/predict.py",                   "predict_day_ahead",   {"zone": "DK1"},   "DK1"),
    (25, "Verificación de predicción day-ahead DK2",          "src/model/predict.py",                   "predict_day_ahead",   {"zone": "DK2"},   "DK2"),
    (26, "Verificación de predicción day-ahead FI",           "src/model/predict.py",                   "predict_day_ahead",   {"zone": "FI"},    "FI"),
    # ── Métricas del modelo servido ───────────────────────────────────────────
    # Escriben data/metadata/served_metrics_<zona>.json (lo que muestra el
    # dashboard). Se lanzan como script en un subproceso (función None).
    (27, "Métricas del modelo servido DK1",                   "scripts/gen_served_metrics.py",          None,                  {"argv": ["DK1"]}, "DK1"),
    (28, "Métricas del modelo servido DK2",                   "scripts/gen_served_metrics.py",          None,                  {"argv": ["DK2"]}, "DK2"),
    (29, "Métricas del modelo servido FI",                    "scripts/gen_served_metrics.py",          None,                  {"argv": ["FI"]},  "FI"),
    (30, "Regímenes del modelo servido FI",                   "scripts/evaluate_fi_regimes.py",         None,                  {},                "FI"),
]


def main():
    parser = argparse.ArgumentParser(description="Pipeline ETL+Modelo — Nordic day-ahead price forecast")
    parser.add_argument("--step",  type=int, help="Ejecutar solo el paso N")
    parser.add_argument("--from",  type=int, dest="from_step", help="Ejecutar desde el paso N")
    parser.add_argument("--until", type=int, help="Ejecutar hasta el paso N (inclusive)")
    parser.add_argument("--zone",  choices=["DK1", "DK2", "FI"],
                        help="Limita los pasos específicos de zona a la indicada")
    parser.add_argument("--optuna", action="store_true",
                        help="Vuelve a buscar hiperparámetros con Optuna (lento); "
                             "por defecto se usan los de src/model/best_params.json")
    parser.add_argument("--fast",  action="store_true",
                        help="Compatibilidad: sin Optuna, que ya es el comportamiento por defecto")
    args = parser.parse_args()

    print(f"\nPIPELINE NORDIC DAY-AHEAD PRICE FORECAST — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if args.optuna and not args.fast:
        print("   Modo --optuna: búsqueda de hiperparámetros con Optuna")

    steps_to_run = list(STEPS)
    if args.step:
        steps_to_run = [s for s in steps_to_run if s[0] == args.step]
    elif args.from_step:
        steps_to_run = [s for s in steps_to_run if s[0] >= args.from_step]
    if args.until:
        steps_to_run = [s for s in steps_to_run if s[0] <= args.until]
    if args.zone:
        # Mantiene los pasos sin zona (None) y los de la zona pedida
        steps_to_run = [s for s in steps_to_run if s[5] in (None, args.zone)]

    results = []
    t_total = time.time()
    for step_num, name, module_path, func_name, kwargs, zone in steps_to_run:
        # Optuna solo con --optuna (y nunca con --fast)
        run_kwargs = dict(kwargs)
        if func_name == "train_zone":
            run_kwargs["run_optuna"] = bool(args.optuna and not args.fast)
        if func_name is None:
            # Pasos de métricas servidas: sin modelo, data_loader cae al modo
            # demo y se escribirían métricas simuladas en served_metrics_*.json.
            reason = served_metrics_blocker(zone, results)
            if reason:
                print(f"\nWARN Paso {step_num} omitido: {reason}")
                results.append((step_num, name, False))
                continue
        ok = run_step(step_num, name, module_path, func_name, run_kwargs)
        results.append((step_num, name, ok))
        # Si falla un paso de extracción/transformación, abortar (los siguientes dependen)
        if not ok and step_num <= 10:
            print(f"\nERROR Paso {step_num} es prerequisito de los siguientes. Abortando.")
            break

    print(f"\n{'='*60}")
    print("RESUMEN PIPELINE")
    print(f"{'='*60}")
    for sn, name, ok in results:
        status = "OK   " if ok else "ERROR"
        print(f"  {status} Paso {sn}: {name}")

    total_elapsed = time.time() - t_total
    failed = sum(1 for _, _, ok in results if not ok)
    print(f"\nTiempo total: {total_elapsed/60:.1f} min | Pasos fallidos: {failed}/{len(results)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
