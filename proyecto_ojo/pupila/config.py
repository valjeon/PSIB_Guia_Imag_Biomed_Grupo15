"""Carga de configuración (config.yaml) con valores por defecto."""
from pathlib import Path
import copy
import yaml

DEFAULTS = {
    "video_path": "data/raw/Registro A_2 copia.mov",
    "output_dir": "resultados",
    "frame_referencia": None,
    "centro_pupila_inicial": None,
    "deteccion": {"ojo": "auto", "frames_muestra": 9},
    "roi": {"ancho": 320, "alto": 130, "centro_pupila_en_roi": [160, 50]},
    "video": {"radio_busqueda_inicial": 150, "radio_busqueda": 80,
              "umbral_circularidad": 0.89},
    "fisiologia": {"t_estimulo_s": None, "ventana_base_s": 2.0, "k_sigma": 3,
                   "n_sostenido": 5, "ventana_mediana": 15,
                   "fraccion_constriccion": 0.9, "fraccion_redilatacion": 0.9},
    "generar_reporte": True,
    "anotaciones_manuales": None,
}


def _fusionar(base, nuevo):
    """Mezcla recursiva de diccionarios (nuevo pisa a base)."""
    for k, v in (nuevo or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _fusionar(base[k], v)
        else:
            base[k] = v
    return base


def cargar_config(ruta_config="config.yaml", video_path=None, output_dir=None):
    """Lee el YAML, completa con valores por defecto y resuelve rutas.

    Las rutas relativas se interpretan respecto de la carpeta del config.
    `video_path` / `output_dir` permiten pisar lo que dice el archivo
    (por ejemplo desde la línea de comandos).
    """
    cfg = copy.deepcopy(DEFAULTS)
    ruta_config = Path(ruta_config).expanduser().resolve()
    if ruta_config.exists():
        with open(ruta_config, "r", encoding="utf-8") as f:
            _fusionar(cfg, yaml.safe_load(f) or {})
    base = ruta_config.parent

    if video_path is not None:
        cfg["video_path"] = str(video_path)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)

    def resolver(p):
        p = Path(p).expanduser()
        return str(p if p.is_absolute() else (base / p).resolve())

    cfg["video_path"] = resolver(cfg["video_path"])
    cfg["output_dir"] = resolver(cfg["output_dir"])
    if cfg.get("anotaciones_manuales"):
        cfg["anotaciones_manuales"] = resolver(cfg["anotaciones_manuales"])
    return cfg
