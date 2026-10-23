"""Selección manual de la ROI del ojo y del centro de la pupila.

Uso (desde la carpeta del proyecto):
    python3 seleccionar_roi.py                    # usa video_path de config.yaml, frame central
    python3 seleccionar_roi.py --frame 300        # elegir otro frame
    python3 seleccionar_roi.py --video "data/Registro B_1.MOV"   # no toca config.yaml

Pasos:
  1) Se abre el frame: arrastrá un rectángulo alrededor del ojo y presioná ENTER.
  2) Se abre la ROI ampliada: hacé clic en el centro de la pupila y presioná ENTER.
Los valores se guardan en config.yaml (frame_referencia, centro_pupila_inicial, roi.*).
Después corré:  python3 run_pipeline.py
"""
import argparse
import re
import cv2

from pupila.config import cargar_config
from pupila.video import info_video, leer_frame, seleccionar_centro_cv2

ANCHO_VENTANA = 1400          # ancho máximo con el que se muestra el frame completo


def _reemplazar(texto, clave, valor, sangria=""):
    """Reemplaza el valor de `clave:` conservando el comentario de la línea."""
    patron = re.compile(rf"^({sangria}{clave}:\s*)([^#\n]*?)(\s*(#.*)?)$", re.M)
    if not patron.search(texto):
        raise RuntimeError(f"No encontré '{clave}:' en config.yaml")
    return patron.sub(lambda m: f"{m.group(1)}{valor}{m.group(3)}", texto, count=1)


def guardar_seleccion_en_config(ruta_config, frame_idx, rect, centro_frame, centro_roi):
    """Escribe la selección en config.yaml sin perder los comentarios."""
    with open(ruta_config, "r", encoding="utf-8") as f:
        t = f.read()
    t = _reemplazar(t, "frame_referencia", int(frame_idx))
    t = _reemplazar(t, "centro_pupila_inicial", f"[{int(centro_frame[0])}, {int(centro_frame[1])}]")
    t = _reemplazar(t, "ancho", int(rect[2]), sangria="  ")
    t = _reemplazar(t, "alto", int(rect[3]), sangria="  ")
    t = _reemplazar(t, "centro_pupila_en_roi", f"[{int(centro_roi[0])}, {int(centro_roi[1])}]", sangria="  ")
    with open(ruta_config, "w", encoding="utf-8") as f:
        f.write(t)


def main():
    ap = argparse.ArgumentParser(description="Selección manual de ROI y centro pupilar")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frame", type=int, default=None, help="frame de referencia (default: el central)")
    args = ap.parse_args()

    cfg = cargar_config(args.config, video_path=args.video)
    info = info_video(cfg["video_path"])
    idx = args.frame if args.frame is not None else (cfg.get("frame_referencia") or info["n_frames"] // 2)
    idx = max(0, min(int(idx), info["n_frames"] - 1))
    frame = leer_frame(cfg["video_path"], idx)

    # 1) ROI: se muestra reducido para que entre en pantalla
    esc = min(1.0, ANCHO_VENTANA / frame.shape[1])
    chico = cv2.resize(frame, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    r = cv2.selectROI(f"Frame {idx}: dibuja un rectangulo alrededor del ojo y ENTER", chico, showCrosshair=False)
    cv2.destroyAllWindows(); cv2.waitKey(1)
    x, y, w, h = [int(round(v / esc)) for v in r]
    if w < 10 or h < 10:
        raise SystemExit("No se seleccionó una ROI válida. Volvé a ejecutar.")
    rect = (x, y, w, h)

    # 2) centro de la pupila dentro de la ROI (ampliada para poder clickear con precisión)
    roi = frame[y:y + h, x:x + w]
    esc2 = max(1, int(1000 / max(w, 1)))
    cx_roi, cy_roi = seleccionar_centro_cv2(roi, escala=esc2)
    centro_frame = (x + cx_roi, y + cy_roi)

    if args.video is None:
        guardar_seleccion_en_config(args.config, idx, rect, centro_frame, (cx_roi, cy_roi))
        print(f"\nGuardado en {args.config}:")
    else:
        print("\nValores (copialos a config.yaml; con --video no se modifica el archivo):")
    print(f"  frame_referencia: {idx}")
    print(f"  centro_pupila_inicial: [{centro_frame[0]}, {centro_frame[1]}]")
    print(f"  roi: ancho={w}, alto={h}, centro_pupila_en_roi=[{cx_roi}, {cy_roi}]")
    print("\nAhora corré:  python3 run_pipeline.py")


if __name__ == "__main__":
    main()
