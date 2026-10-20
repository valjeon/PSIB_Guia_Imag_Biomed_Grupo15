"""Ejecuta el pipeline completo sobre un video.

Uso:
    python run_pipeline.py                              # usa config.yaml
    python run_pipeline.py --video otro_registro.mov    # pisa video_path
    python run_pipeline.py --config mi_config.yaml --out resultados_B
"""
import argparse
import matplotlib
matplotlib.use("Agg")                      # sin ventanas: corre en cualquier máquina

from pupila.config import cargar_config
from pupila.pipeline import ejecutar_pipeline

def main():
    ap = argparse.ArgumentParser(description="Pipeline de reflejo fotomotor pupilar")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None, help="ruta del video (pisa video_path del config)")
    ap.add_argument("--out", default=None, help="carpeta de salida (pisa output_dir del config)")
    args = ap.parse_args()
    cfg = cargar_config(args.config, video_path=args.video, output_dir=args.out)
    ejecutar_pipeline(cfg)


if __name__ == "__main__":
    main()
