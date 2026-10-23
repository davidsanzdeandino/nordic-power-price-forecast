"""Comprobación manual de render (headless) de pages/5_Battery_Lab.py con streamlit.testing.

Ejecuta la página completa (Finlandia por defecto) y sale con código 1 si hay
excepciones. No es un test de pytest: necesita los datos y modelos que genera
run_pipeline.py.

Uso:  python checks/test_battery_lab_render.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "pages" / "5_Battery_Lab.py"), default_timeout=600)
at.run()

if at.exception:
    for e in at.exception:
        print("EXCEPTION:", e.value)
        print(e.stack_trace)
    sys.exit(1)

print(f"OK — página renderizada sin excepciones")
print(f"  títulos: {[t.value for t in at.title]}")
print(f"  metrics: {len(at.metric)} | warnings: {len(at.warning)} | dataframes: {len(at.dataframe)}")
