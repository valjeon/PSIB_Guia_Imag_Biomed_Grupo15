"""Detección automática y robusta del centro de la pupila.

Idea: no basta con "lo más oscuro" (cejas, pelo, sombras y pestañas lo son más que la pupila).
Se generan candidatos oscuros con varios umbrales y se puntúa cada uno por lo que SÍ distingue
a una pupila:
  - forma compacta y redonda (circularidad, solidez, relación de aspecto);
  - tamaño plausible respecto del frame;
  - rodeada de un anillo claro (iris) -> alto contraste de borde;
  - sin otras regiones oscuras pegadas (ceja/pelo/sombra);
  - no toca el borde del frame;
  - reflejo del flash (punto brillante) cerca -> bonus.
Se repite sobre varios frames del video y se vota (mediana de los frames que coinciden),
lo que descarta parpadeos y frames mal iluminados.
"""
import cv2
import numpy as np

ANCHO_TRABAJO = 640          # se detecta en una versión reducida (rápido y menos sensible al ruido)
UMBRALES = tuple(sorted({int(round(4 * 1.22 ** k)) for k in range(0, 22)}))   # 6 ... ~190, paso geométrico (sirve en escenas oscuras)
DIAM_MIN_FRAC, DIAM_MAX_FRAC = 0.015, 0.07      # diámetro de la pupila / ancho del frame


def _preparar(frame_bgr):
    h, w = frame_bgr.shape[:2]
    esc = min(1.0, ANCHO_TRABAJO / w)
    chico = cv2.resize(frame_bgr, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else frame_bgr
    gris = cv2.cvtColor(chico, cv2.COLOR_BGR2GRAY)
    k = max(3, int(round(chico.shape[1] * 0.012)) | 1)
    # la apertura de grises borra reflejos brillantes pequeños (no la pupila)
    sin_reflejo = cv2.morphologyEx(gris, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    suave = cv2.GaussianBlur(sin_reflejo, (0, 0), max(1.0, chico.shape[1] * 0.004))
    hsv = cv2.cvtColor(chico, cv2.COLOR_BGR2HSV)
    return gris, suave, esc, hsv


def candidatos_pupila(frame_bgr, max_candidatos=5):
    """Lista de candidatos [{'centro': (x, y), 'radio': r, 'score': s, ...}] en coordenadas del frame."""
    gris, suave, esc, hsv = _preparar(frame_bgr)
    h, w = suave.shape
    amin = np.pi * (DIAM_MIN_FRAC * w / 2) ** 2
    amax = np.pi * (DIAM_MAX_FRAC * w / 2) ** 2
    margen = int(0.01 * w)
    encontrados = []

    for p in UMBRALES:
        thr = p
        binaria = (suave <= thr).astype(np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if not (amin <= area <= amax):
                continue
            if x <= margen or y <= margen or x + bw >= w - margen or y + bh >= h - margen:
                continue                                  # toca el borde: ceja, pelo, marco
            if max(bw, bh) / max(min(bw, bh), 1) > 1.6:
                continue                                  # alargado: ceja, pestañas
            mascara = (etiquetas == i).astype(np.uint8)
            cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cont, key=cv2.contourArea)
            per = cv2.arcLength(c, True)
            circ = 4 * np.pi * cv2.contourArea(c) / per ** 2 if per > 0 else 0
            solidez = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
            if circ < 0.78 or solidez < 0.85:
                continue
            r = np.sqrt(area / np.pi)
            M = cv2.moments(mascara, binaryImage=True)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

            # anillo exterior (iris) vs. interior
            kk = max(3, int(round(r * 0.7)) * 2 + 1)
            kel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
            anillo = cv2.subtract(cv2.dilate(mascara, kel), cv2.dilate(mascara, np.ones((3, 3), np.uint8)))
            if anillo.sum() < 10:
                continue
            m_int = cv2.mean(suave, mask=mascara * 255)[0]
            m_ext = cv2.mean(suave, mask=anillo * 255)[0]
            contraste = m_ext - m_int
            # fracción del anillo que también es oscura (pegada a otra región oscura)
            oscuro_vecino = float((suave[anillo > 0] <= thr + max(6, 0.15 * m_ext)).mean())
            contraste_rel = contraste / (m_ext + 1.0)          # relativo: no depende de la iluminación
            if contraste < 8 or contraste_rel < 0.35 or oscuro_vecino > 0.35:
                continue

            # reflejo del flash DENTRO de la pupila candidata (en 'gris' sin abrir)
            interior = mascara > 0     # la apertura de grises borró el reflejo de 'suave': queda dentro de la máscara
            vals = gris[interior]
            brillantes = float((vals >= max(m_ext * 1.8 + 15, 60)).mean())
            # un reflejo es un punto chico y brillante dentro de una región por lo demás oscura
            reflejo = bool(0 < brillantes <= 0.2 and np.median(vals) < m_ext * 0.75)

            # esclerótica a AMBOS lados (izquierda y derecha) de la pupila: clara y poco saturada
            R = int(round(r * 9))
            x0, x1 = max(int(cx) - R, 0), min(int(cx) + R + 1, w)
            y0, y1 = max(int(cy) - R, 0), min(int(cy) + R + 1, h)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist = np.hypot(xx - cx, yy - cy)
            anulo = (dist > 1.8 * r) & (dist < 9 * r)
            clara = (hsv[y0:y1, x0:x1, 1] < 60) & (gris[y0:y1, x0:x1] > m_ext * 1.5)
            izq, der = anulo & (xx < cx), anulo & (xx > cx)
            hay_esclera = bool(izq.any() and der.any() and clara[izq].mean() > 0.05 and clara[der].mean() > 0.05)

            score = ((circ ** 3) * min(solidez, 1.0) * min(contraste_rel / 0.5, 1.0) * (1 - oscuro_vecino)
                     * (2.5 if reflejo else 1.0) * (1.6 if hay_esclera else 1.0))
            encontrados.append({"centro": (cx / esc, cy / esc), "radio": r / esc, "score": float(score),
                                "circularidad": float(circ), "solidez": float(solidez),
                                "contraste": float(contraste), "reflejo": reflejo, "esclerotica": hay_esclera, "umbral": p})

    # quedarse con el mejor de cada zona (los umbrales distintos generan duplicados)
    encontrados.sort(key=lambda d: -d["score"])
    finales = []
    for c in encontrados:
        if all(np.hypot(c["centro"][0] - f["centro"][0], c["centro"][1] - f["centro"][1]) > f["radio"] for f in finales):
            finales.append(c)
        if len(finales) >= max_candidatos:
            break
    return finales


def detectar_pupila(frame_bgr):
    """Mejor candidato en un frame o None."""
    c = candidatos_pupila(frame_bgr)
    return c[0] if c else None


def detectar_en_video(video_path, n_frames_muestra=9, score_minimo=0.25, ojo="auto"):
    """Detección por votación en varios frames del video.

    ojo: "auto" (el grupo mejor puntuado), "izquierdo" o "derecho" (según se ve en la imagen;
    sirve cuando se ven los dos ojos). Devuelve un dict con centro (x, y), radio, confianza
    (fracción de frames que coinciden) y las detecciones; None si no se detecta nada creíble.
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.unique(np.linspace(0, max(n - 1, 0), n_frames_muestra).astype(int))
    dets = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        for c in candidatos_pupila(frame, max_candidatos=4):
            if c["score"] >= score_minimo:
                dets.append({**c, "frame": int(i)})
    cap.release()
    if not dets:
        return None

    # agrupar detecciones que coinciden en posición (a menos de 1.5 radios) a lo largo de los frames
    grupos = []
    for d in sorted(dets, key=lambda e: -e["score"]):
        for g in grupos:
            c0 = g[0]
            if np.hypot(d["centro"][0] - c0["centro"][0], d["centro"][1] - c0["centro"][1]) <= max(d["radio"], c0["radio"]) * 1.5:
                g.append(d)
                break
        else:
            grupos.append([d])
    # valor = apoyo temporal (frames distintos) x puntaje medio: un objeto estable pero poco "pupilar" no gana
    puntaje = lambda g: float(np.mean([e["score"] for e in g]))
    valores = [len({e["frame"] for e in g}) * puntaje(g) for g in grupos]
    mejor_valor = max(valores)
    if ojo in ("izquierdo", "derecho"):
        # los dos ojos son grupos de puntaje comparable: se elige por posición horizontal
        finalistas = [g for g, v in zip(grupos, valores) if v >= 0.6 * mejor_valor]
        x_med = lambda g: np.median([e["centro"][0] for e in g])
        elegido = min(finalistas, key=x_med) if ojo == "izquierdo" else max(finalistas, key=x_med)
    else:
        elegido = grupos[int(np.argmax(valores))]
    xs = np.median([e["centro"][0] for e in elegido])
    ys = np.median([e["centro"][1] for e in elegido])
    r = np.median([e["radio"] for e in elegido])
    frames_ok = len({e["frame"] for e in elegido})
    return {"centro": (int(round(xs)), int(round(ys))), "radio": float(r),
            "confianza": frames_ok / len(indices), "frames_usados": frames_ok,
            "frames_muestreados": len(indices), "detecciones": dets,
            "grupos": [(int(np.median([e["centro"][0] for e in g])), int(np.median([e["centro"][1] for e in g])),
                        len({e["frame"] for e in g})) for g in grupos]}

def refinar_centro_pupila(gris, centro, radio, factores=(0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8),
                          ancho_anillo=0.3, radio_busqueda=1.5):
    """Centro (y radio) del disco oscuro con más contraste contra su anillo exterior, probando varios
    radios alrededor de 'radio'. Sin umbrales: sirve con pupilas no uniformes y radios aproximados."""
    k = max(3, int(round(radio * 0.6)) | 1)
    a = cv2.morphologyEx(gris, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))   # quita reflejos
    a = cv2.GaussianBlur(a, (0, 0), max(1.0, 0.1 * radio)).astype(np.float32)
    yv, xv = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    lejos = np.hypot(xv - centro[0], yv - centro[1]) > radio_busqueda * radio
    mejor, pos = -np.inf, (float(centro[0]), float(centro[1]))
    for f in factores:
        r = f * radio
        h = int(round((1 + ancho_anillo) * r)) + 1
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]; d = np.hypot(xx, yy)
        disco, anillo = d <= r, (d > r + 1) & (d <= (1 + ancho_anillo) * r)
        nucleo = (anillo / anillo.sum() - disco / disco.sum()).astype(np.float32)
        resp = cv2.filter2D(a, -1, nucleo, borderType=cv2.BORDER_REPLICATE)     # media del anillo - media del disco
        resp[lejos] = -np.inf
        y, x = np.unravel_index(np.argmax(resp), resp.shape)
        if resp[y, x] > mejor:
            mejor, pos = resp[y, x], (float(x), float(y))
    return pos