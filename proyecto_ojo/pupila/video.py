"""Lectura de video, detección del centro pupilar y recorte de la ROI."""
from pathlib import Path
import cv2
import numpy as np

KERNEL_REFLEJO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))


# ----------------------------------------------------------------- video
def info_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if info["n_frames"] <= 0:
        raise RuntimeError("El video no contiene fotogramas legibles.")
    info["duracion_s"] = info["n_frames"] / info["fps"] if info["fps"] > 0 else float("nan")
    return info


def leer_frame(video_path, indice):
    """Devuelve el frame `indice` en BGR."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"No se pudo leer el frame {indice}.")
    return frame


# ------------------------------------------------------ seguimiento pupila
def seguir_pupila(frame_bgr, centro_previo, radio_busqueda=80,
                  kernel_suavizado=31, margen_oscuro=15):
    """Centro de la región más oscura dentro de una ventana alrededor de `centro_previo`."""
    h, w = frame_bgr.shape[:2]
    cx, cy = int(centro_previo[0]), int(centro_previo[1])
    x0, x1 = max(cx - radio_busqueda, 0), min(cx + radio_busqueda, w)
    y0, y1 = max(cy - radio_busqueda, 0), min(cy + radio_busqueda, h)

    ventana = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ventana = cv2.morphologyEx(ventana, cv2.MORPH_OPEN, KERNEL_REFLEJO)   # borra reflejos brillantes
    ventana = cv2.GaussianBlur(ventana, (kernel_suavizado, kernel_suavizado), 0)
    valor_min, _, punto_min, _ = cv2.minMaxLoc(ventana)

    _, binaria = cv2.threshold(ventana, int(valor_min) + margen_oscuro, 255, cv2.THRESH_BINARY_INV)
    _, etiquetas = cv2.connectedComponents(binaria)
    region = (etiquetas == etiquetas[punto_min[1], punto_min[0]]).astype(np.uint8)
    M = cv2.moments(region, binaryImage=True)
    if M["m00"] == 0:
        return cx, cy
    return x0 + int(M["m10"] / M["m00"]), y0 + int(M["m01"] / M["m00"])


def detectar_centro_inicial(frame_bgr):
    """Detección automática: busca la zona oscura dominante en todo el frame."""
    h, w = frame_bgr.shape[:2]
    return seguir_pupila(frame_bgr, (w // 2, h // 2), radio_busqueda=max(h, w))


# ------------------------------------------------------------------- ROI
def calcular_origen_roi(shape, centro, roi_w, roi_h, centro_en_roi):
    h, w = shape[:2]
    x0 = int(np.clip(centro[0] - centro_en_roi[0], 0, max(w - roi_w, 0)))
    y0 = int(np.clip(centro[1] - centro_en_roi[1], 0, max(h - roi_h, 0)))
    return x0, y0


def recortar_roi(frame_bgr, centro, roi_w=320, roi_h=130, centro_en_roi=(160, 50)):
    """Recorta la ROI con la pupila en `centro_en_roi`. Devuelve (roi, centro dentro de la ROI)."""
    roi_w, roi_h = min(roi_w, frame_bgr.shape[1]), min(roi_h, frame_bgr.shape[0])
    x0, y0 = calcular_origen_roi(frame_bgr.shape, centro, roi_w, roi_h, centro_en_roi)
    roi = frame_bgr[y0:y0 + roi_h, x0:x0 + roi_w]
    return roi, (int(centro[0] - x0), int(centro[1] - y0))


def preparar_referencia(cfg):
    """Frame de referencia, centro pupilar y ROI (RGB y gris) a partir de la config.

    Devuelve un diccionario con todo lo que usan los Módulos 1 a 4.
    """
    info = info_video(cfg["video_path"])
    idx = cfg.get("frame_referencia")
    idx = info["n_frames"] // 2 if idx is None else int(np.clip(idx, 0, info["n_frames"] - 1))
    frame_bgr = leer_frame(cfg["video_path"], idx)

    centro = cfg.get("centro_pupila_inicial")
    automatico = centro is None
    deteccion = None
    if automatico:
        from .deteccion import detectar_en_video
        d = cfg.get("deteccion", {})
        deteccion = detectar_en_video(cfg["video_path"], d.get("frames_muestra", 9), ojo=d.get("ojo", "auto"))
        centro = deteccion["centro"] if deteccion else tuple(detectar_centro_inicial(frame_bgr))
    else:
        centro = tuple(int(v) for v in centro)

    r = cfg["roi"]
    roi_bgr, centro_roi = recortar_roi(frame_bgr, centro, r["ancho"], r["alto"], r["centro_pupila_en_roi"])
    x0, y0 = calcular_origen_roi(frame_bgr.shape, centro, roi_bgr.shape[1], roi_bgr.shape[0],
                                 r["centro_pupila_en_roi"])
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    return {
        "info": info,
        "frame_index": idx,
        "frame_bgr": frame_bgr,
        "frame_rgb": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        "centro_frame": centro,
        "centro_automatico": automatico,
        "deteccion": deteccion,
        "roi_rect": (x0, y0, roi_bgr.shape[1], roi_bgr.shape[0]),
        "roi_rgb": roi_rgb,
        "roi_gray": cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY),
        "centro_roi": centro_roi,
    }


def guardar_vista_previa(ctx, ruta):
    """Guarda el frame de referencia con el centro y la ROI dibujados (control visual)."""
    vis = ctx["frame_bgr"].copy()
    x0, y0, w, h = ctx["roi_rect"]
    cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 3)
    cv2.circle(vis, tuple(map(int, ctx["centro_frame"])), 8, (0, 255, 0), -1)
    det = ctx.get("deteccion")
    if det:
        cv2.circle(vis, tuple(map(int, ctx["centro_frame"])), int(det["radio"] * 1.8), (0, 255, 0), 3)
        for gx, gy, votos in det["grupos"]:                      # otros candidatos (p. ej. el otro ojo)
            if (gx, gy) != tuple(ctx["centro_frame"]):
                cv2.circle(vis, (gx, gy), 20, (0, 200, 255), 2)
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), vis)
    return ruta


# ------------------------------------------- selección manual (opcional)
def seleccionar_roi_cv2(frame_bgr):
    """Abre una ventana de OpenCV para dibujar la ROI. Devuelve (x, y, w, h)."""
    rect = cv2.selectROI("Seleccioná la ROI del ojo y presioná Enter", frame_bgr, showCrosshair=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if rect[2] == 0 or rect[3] == 0:
        raise RuntimeError("No se seleccionó una ROI válida.")
    return tuple(int(v) for v in rect)


def seleccionar_centro_cv2(roi_bgr, escala=3):
    """Abre una ventana para hacer clic en el centro de la pupila (coordenadas de la ROI)."""
    grande = cv2.resize(roi_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    pto = []

    def click(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pto.append((x, y))

    cv2.namedWindow("Clic en el centro de la pupila (Enter para aceptar)")
    cv2.setMouseCallback("Clic en el centro de la pupila (Enter para aceptar)", click)
    while True:
        vis = grande.copy()
        if pto:
            cv2.circle(vis, pto[-1], 4, (0, 0, 255), -1)
        cv2.imshow("Clic en el centro de la pupila (Enter para aceptar)", vis)
        if (cv2.waitKey(20) & 0xFF) == 13 and pto:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return int(pto[-1][0] / escala), int(pto[-1][1] / escala)
