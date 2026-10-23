"""Comprobación manual de render (headless) de pages/6_Transparency.py (default = Denmark/DK1).

Ejecuta la página con streamlit.testing y sale con código 1 si hay excepciones.
No es un test de pytest: necesita los datos y modelos que genera run_pipeline.py.

Uso:  python checks/test_transparency_render.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "pages" / "6_Transparency.py"), default_timeout=600)
at.run()

if at.exception:
    for e in at.exception:
        print("EXCEPTION:", e.value)
        print(e.stack_trace)
    sys.exit(1)

print("OK — página renderizada sin excepciones")
print(f"  metrics: {len(at.metric)} | dataframes: {len(at.dataframe)} | markdown blocks: {len(at.markdown)}")
for m in at.metric:
    print(f"  metric: {m.label} = {m.value}")
