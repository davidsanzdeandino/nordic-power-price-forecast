"""
utils/logger.py
===============
Utilidad de logging compartida para todos los scripts del pipeline.
Genera logs en data/metadata/ con timestamp, parámetros y errores.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# En Windows, al redirigir/pipe la salida, Python usa la codificación local
# (cp1252) en lugar de UTF-8, lo que rompe print() con caracteres como →, €,
# q̂, ±. Reconfiguramos stdout/stderr a UTF-8 para todos los scripts del
# pipeline (este módulo lo importan todos). errors="replace" como red de
# seguridad para que NUNCA un carácter exótico aborte una ejecución.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

METADATA_DIR = Path("data/metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger configurado que escribe en consola Y en archivo.
    
    Args:
        name: Nombre del módulo (ej: 'entsoe_prices')
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evitar duplicar handlers si se llama varias veces
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Handler consola
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler archivo
    log_file = METADATA_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def robust_pickle_dump(obj, path, retries: int = 6, delay: float = 1.5):
    """
    Guarda un objeto con pickle reintentando ante OSError transitorio.

    MOTIVO: el proyecto puede vivir bajo OneDrive/Dropbox, que bloquean ficheros
    durante la sincronización y provocan `OSError [Errno 22] Invalid argument`
    en escrituras grandes (.pkl de modelos). Se escribe a un fichero temporal y
    se reemplaza atómicamente, reintentando con backoff.
    """
    import os, time, pickle, tempfile
    path = str(path)
    last_err = None
    for attempt in range(retries):
        try:
            d = os.path.dirname(path) or "."
            fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=d)
            with os.fdopen(fd, "wb") as f:
                pickle.dump(obj, f)
            os.replace(tmp, path)
            return path
        except OSError as e:
            last_err = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(delay * (attempt + 1))
    # Último intento directo (por si el problema era os.replace)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_run_metadata(script_name: str, params: dict, stats: dict):
    """
    Guarda un JSON con metadatos de la ejecución del script.
    
    Args:
        script_name: Nombre del script (ej: 'entsoe_prices')
        params:      Parámetros usados (fechas, zonas, etc.)
        stats:       Resultados (filas descargadas, errores, etc.)
    """
    metadata = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "stats": stats
    }
    
    out_path = METADATA_DIR / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return out_path