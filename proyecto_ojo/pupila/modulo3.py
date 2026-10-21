"""Módulo 3: descriptores, textura, procesamiento del video y parámetros fisiológicos."""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from .modulo2 import segmentar_pupila
from .video import seguir_pupila, recortar_roi

# ======================================================================
#  3.1  Descriptores morfométricos y de textura (frame de referencia)
# ======================================================================
NIVELES = 32
DISTANCIAS = [1]
ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def cuantizar_roi(roi_rgb, roi_gray, niveles=NIVELES):
    paso = 256 // niveles
    return (roi_rgb // paso).astype(np.uint8), (roi_gray // paso).astype(np.uint8)


def circularidad(contorno):
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    return 4 * np.pi * area / perimetro ** 2 if perimetro > 0 else np.nan


def descriptores_morfometricos(contorno):
    """Área, perímetro, centroide, circularidad, excentricidad. Devuelve (dict, elipse ajustada)."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    m = cv2.moments(contorno)
    xc, yc = m["m10"] / m["m00"], m["m01"] / m["m00"]
    elipse = cv2.fitEllipse(contorno)
    (_, _), (d1, d2), _ = elipse
    mayor, menor = max(d1, d2), min(d1, d2)
    return {
        "Área (px²)": area,
        "Perímetro (px)": perimetro,
        "Centroide X_c (px)": xc,
        "Centroide Y_c (px)": yc,
        "Circularidad": 4 * np.pi * area / perimetro ** 2,
        "Excentricidad": np.sqrt(1 - (menor / mayor) ** 2),
        "Eje mayor (px)": mayor,
        "Eje menor (px)": menor,
    }, elipse


def _kernel_elipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def segmentar_iris(roi_gray, mask_pupila, centro, anillo=(6, 12), fondo_semiejes=(78, 70), y_parpado=10):
    """Watershed del iris con tres marcadores: iris (anillo), pupila y fondo. Devuelve máscara 0/255."""
    cx, cy = int(round(centro[0])), int(round(centro[1]))
    marc = np.zeros(roi_gray.shape, dtype=np.int32)
    zona = np.zeros(roi_gray.shape, dtype=np.uint8)
    cv2.ellipse(zona, (cx, cy), fondo_semiejes, 0, 0, 360, 1, -1)
    marc[zona == 0] = 2
    marc[:y_parpado, :] = 2
    semilla = cv2.subtract(cv2.dilate(mask_pupila, _kernel_elipse(anillo[1])),
                           cv2.dilate(mask_pupila, _kernel_elipse(anillo[0])))
    marc[semilla > 0] = 1
    marc[cv2.erode(mask_pupila, _kernel_elipse(2)) > 0] = 3
    suave = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    ws = cv2.watershed(cv2.cvtColor(suave, cv2.COLOR_GRAY2BGR), marc.copy())
    return (ws == 1).astype(np.uint8) * 255, marc


def mascara_sin_reflejos(roi_gray, mask, umbral_brillo=205, dilatacion=5):
    """Quita de `mask` los píxeles de reflejo (muy claros). Devuelve (máscara limpia, máscara de brillo)."""
    _, brillo = cv2.threshold(roi_gray, umbral_brillo, 255, cv2.THRESH_BINARY)
    brillo = cv2.dilate(brillo, np.ones((dilatacion, dilatacion), np.uint8))
    return cv2.bitwise_and(mask, cv2.bitwise_not(brillo)), brillo


def estadisticas_primer_orden(img_cuant, mascara, niveles=NIVELES):
    """Media, desvío, entropía y asimetría por canal R, G, B dentro de la máscara."""
    medias, desvios = cv2.meanStdDev(img_cuant, mask=mascara)
    filas = {}
    for i, canal in enumerate("RGB"):
        hist = cv2.calcHist([img_cuant], [i], mascara, [niveles], [0, niveles]).ravel()
        p = hist / hist.sum()
        p = p[p > 0]
        valores = img_cuant[:, :, i][mascara > 0]
        filas[canal] = {"Media": medias[i, 0], "Desvío": desvios[i, 0],
                        "Entropía (bits)": -np.sum(p * np.log2(p)),
                        "Asimetría": pd.Series(valores).skew()}
    return pd.DataFrame(filas).T


def glcm_mascara(canal, mascara, niveles=NIVELES):
    """GLCM (suma de 4 ángulos, d=1) restringida a la máscara."""
    img = canal.copy()
    img[mascara == 0] = niveles                         # nivel extra = "fuera de la región"
    glcm = graycomatrix(img, distances=DISTANCIAS, angles=ANGULOS,
                        levels=niveles + 1, symmetric=True, normed=False)
    glcm = glcm[:niveles, :niveles].astype(float)
    glcm /= glcm.sum(axis=(0, 1), keepdims=True)
    return glcm


def atributos_glcm(glcm):
    props = {"Contraste": "contrast", "Homogeneidad": "homogeneity",
             "Energía": "energy", "Correlación": "correlation"}
    return {n: graycoprops(glcm, p).mean() for n, p in props.items()}


def textura_glcm_por_canal(roi_cuant, mask_pupila, mask_iris, niveles=NIVELES):
    """Devuelve (tabla pupila, tabla iris, dict de GLCMs)."""
    glcms, fp, fi = {}, {}, {}
    for idx, canal in enumerate("RGB"):
        img = roi_cuant[:, :, idx]
        glcms[("Pupila", canal)] = glcm_mascara(img, mask_pupila, niveles)
        glcms[("Iris", canal)] = glcm_mascara(img, mask_iris, niveles)
        fp[canal] = atributos_glcm(glcms[("Pupila", canal)])
        fi[canal] = atributos_glcm(glcms[("Iris", canal)])
    return pd.DataFrame(fp).T, pd.DataFrame(fi).T, glcms


# ======================================================================
#  3.2  Segmentación frame a frame
# ======================================================================
UMBRAL_REFLEJO = 200
RADIO_ZONA_REFLEJO = 45
KERNEL_DILATAR_REFLEJO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
RADIOS_MARCADOR = (10, 16, 22)
MARGEN_CONTRASTE = 8
KERNEL_COLITAS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
KERNEL_CIERRE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))


def quitar_reflejos(gris, cx, cy):
    """Rellena (inpainting) los reflejos brillantes cercanos a la pupila."""
    _, brillo = cv2.threshold(gris, UMBRAL_REFLEJO, 255, cv2.THRESH_BINARY)
    brillo = cv2.dilate(brillo, KERNEL_DILATAR_REFLEJO)
    cerca = np.zeros(gris.shape, np.uint8)
    cv2.circle(cerca, (int(cx), int(cy)), RADIO_ZONA_REFLEJO, 255, -1)
    brillo = cv2.bitwise_and(brillo, cerca)
    return cv2.inpaint(gris, brillo, 5, cv2.INPAINT_TELEA)


def contorno_con_marcador(limpia, cx, cy, radio):
    """Watershed + apertura + cierre + envolvente convexa para un radio de marcador dado."""
    try:
        mascara, _ = segmentar_pupila(limpia, cx, cy, radio=radio)
    except (ValueError, cv2.error):
        return None
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL_COLITAS)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL_CIERRE)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None
    return cv2.convexHull(max(contornos, key=cv2.contourArea))


def contraste_borde(gris, contorno, ancho=4):
    """Gris medio del anillo exterior menos el del interior (alto = pupila oscura rodeada de iris)."""
    mascara = np.zeros(gris.shape, np.uint8)
    cv2.drawContours(mascara, [contorno], -1, 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ancho + 1, 2 * ancho + 1))
    ext = cv2.subtract(cv2.dilate(mascara, k), mascara)
    int_ = cv2.subtract(mascara, cv2.erode(mascara, k))
    return cv2.mean(gris, mask=ext)[0] - cv2.mean(gris, mask=int_)[0]


def medir_pupila(roi_gris, cx, cy):
    """Contorno de la pupila en una ROI (o None si no segmenta nada)."""
    limpia = quitar_reflejos(roi_gris, cx, cy)
    mejor = contorno_con_marcador(limpia, cx, cy, RADIOS_MARCADOR[0])
    if mejor is None:
        return None
    mejor_c = contraste_borde(limpia, mejor)
    for radio in RADIOS_MARCADOR[1:]:
        cand = contorno_con_marcador(limpia, cx, cy, radio)
        if cand is None:
            continue
        c = contraste_borde(limpia, cand)
        if c > mejor_c + MARGEN_CONTRASTE:
            mejor, mejor_c = cand, c
    return mejor


def procesar_video(video_path, centro_inicial, cfg, progreso=True):
    """Segmenta la pupila en TODOS los frames (una sola pasada por el video).

    Devuelve un DataFrame con una fila por frame:
    frame, tiempo_s, area_cruda_px2, diametro_crudo_px, circularidad, valido,
    area_px2, diametro_px, interpolado, centro_x_px, centro_y_px, brillo_medio.
    Los frames inválidos (circularidad < umbral o sin segmentación) se interpolan.
    """
    r, v = cfg["roi"], cfg["video"]
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    centro = tuple(int(c) for c in centro_inicial)
    filas = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            filas.append((i, i / fps, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        brillo = cv2.mean(cv2.cvtColor(cv2.resize(frame, None, fx=0.1, fy=0.1,
                                                  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))[0]
        radio = v["radio_busqueda_inicial"] if i == 0 else v["radio_busqueda"]
        centro = seguir_pupila(frame, centro, radio)
        roi, (cx, cy) = recortar_roi(frame, centro, r["ancho"], r["alto"], r["centro_pupila_en_roi"])
        cont = medir_pupila(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cx, cy)
        if cont is None:
            filas.append((i, i / fps, np.nan, np.nan, np.nan, centro[0], centro[1], brillo))
        else:
            area = cv2.contourArea(cont)
            filas.append((i, i / fps, area, 2 * np.sqrt(area / np.pi), circularidad(cont),
                          centro[0], centro[1], brillo))
        if progreso and (i % 100 == 0 or i == n - 1):
            print(f"  frame {i + 1}/{n}", end="\r")
    cap.release()
    if progreso:
        print()

    df = pd.DataFrame(filas, columns=["frame", "tiempo_s", "area_cruda_px2", "diametro_crudo_px",
                                      "circularidad", "centro_x_px", "centro_y_px", "brillo_medio"])
    df["valido"] = df["circularidad"] >= v["umbral_circularidad"]      # NaN => inválido
    for crudo, final in [("area_cruda_px2", "area_px2"), ("diametro_crudo_px", "diametro_px")]:
        df[final] = df[crudo].where(df["valido"]).interpolate(method="linear", limit_direction="both")
    df["interpolado"] = ~df["valido"]
    return df


def resumen_artefactos(df, fps):
    inv = df["interpolado"]
    rachas = inv.ne(inv.shift()).cumsum()[inv].value_counts()
    return {"frames_interpolados": int(inv.sum()),
            "frames_interpolados_pct": 100 * float(inv.mean()),
            "racha_mas_larga_frames": int(rachas.max()) if len(rachas) else 0,
            "racha_mas_larga_s": float(rachas.max() / fps) if len(rachas) else 0.0}


# ----------------------------------------------- validación vs. manual
def validar_contra_manual(df, ruta_csv):
    """MAE/RMSE/sesgo del área y diámetro automáticos vs. anotaciones manuales.

    El CSV debe tener columnas: frame, area_manual_px2, diametro_manual_px.
    """
    manual = pd.read_csv(ruta_csv).rename(columns={"Área manual (px²)": "area_manual_px2",
                                                   "Diámetro manual (px)": "diametro_manual_px"})
    comp = manual.merge(df, on="frame")

    def m(auto, man):
        e = auto - man
        return {"n": len(e), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                "Sesgo (auto-manual)": e.mean(), "Error relativo medio (%)": 100 * (e.abs() / man).mean()}

    tabla = pd.DataFrame({"Área (px²)": m(comp["area_px2"], comp["area_manual_px2"]),
                          "Diámetro (px)": m(comp["diametro_px"], comp["diametro_manual_px"])}).T
    return tabla, comp


def anotar_frames_manualmente(video_path, indices, centros, cfg, ruta_csv, escala=4):
    """Anotación interactiva (ventana OpenCV): clic en el borde de la pupila (mín. 5 puntos).

    Enter acepta, u deshace, s saltea. Guarda/continúa un CSV con frame, area_manual_px2, diametro_manual_px.
    `centros` es el array (n_frames, 2) de centros pupilares (df[['centro_x_px','centro_y_px']]).
    """
    r = cfg["roi"]
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    manual = (pd.read_csv(ruta_csv).rename(columns={"Área manual (px²)": "area_manual_px2",
                                                    "Diámetro manual (px)": "diametro_manual_px"})
              if os.path.exists(ruta_csv)
              else pd.DataFrame(columns=["frame", "area_manual_px2", "diametro_manual_px"]))
    for idx in indices:
        if idx in manual["frame"].values:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        roi, _ = recortar_roi(frame, (int(centros[idx][0]), int(centros[idx][1])),
                              r["ancho"], r["alto"], r["centro_pupila_en_roi"])
        grande = cv2.resize(roi, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
        puntos = []

        def click(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                puntos.append((x, y))

        ventana = f"frame {idx} - clic en el borde (min 5) | Enter | u: deshacer | s: saltear"
        cv2.namedWindow(ventana)
        cv2.setMouseCallback(ventana, click)
        saltear = False
        while True:
            vis = grande.copy()
            for p in puntos:
                cv2.circle(vis, p, 3, (0, 0, 255), -1)
            if len(puntos) >= 5:
                cv2.ellipse(vis, cv2.fitEllipse(np.array(puntos, np.float32)), (0, 255, 0), 1)
            cv2.imshow(ventana, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and len(puntos) >= 5:
                break
            if k == ord("u") and puntos:
                puntos.pop()
            if k == ord("s"):
                saltear = True
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if saltear:
            continue
        (_, _), (d1, d2), _ = cv2.fitEllipse(np.array(puntos, np.float32) / escala)
        area = np.pi * d1 * d2 / 4
        manual.loc[len(manual)] = [idx, area, 2 * np.sqrt(area / np.pi)]
        manual.to_csv(ruta_csv, index=False)
    cap.release()
    return manual


# ======================================================================
#  3.3  Parámetros fisiológicos
# ======================================================================
def suavizar(serie, ventana=15):
    """Mediana móvil centrada (descarta picos aislados)."""
    return pd.Series(serie).rolling(ventana, center=True, min_periods=1).median()


def detectar_estimulo(brillo, fps, t_estimulo_s=None):
    """Índice del frame del estímulo: salto de brillo (o el tiempo dado en la config)."""
    brillo = pd.Series(brillo).reset_index(drop=True)
    n = len(brillo)
    if t_estimulo_s is not None:
        return int(np.clip(round(t_estimulo_s * fps), 0, n - 1))
    b = brillo.rolling(5, center=True, min_periods=1).median()
    salto = b.diff(5).abs()
    if salto.isna().all():
        return None
    i_pico = int(salto.idxmax())
    base = brillo.iloc[max(0, i_pico - 60): max(0, i_pico - 10)]
    if base.dropna().size < 3:
        base = brillo.iloc[:max(i_pico, 1)]
    mu, sd = base.median(), base.std()
    sd = 0.0 if np.isnan(sd) else sd
    umbral = mu + max(3 * sd, 0.05 * (brillo.iloc[i_pico] - mu))
    ini = max(0, i_pico - 15)
    return ini + int(np.argmax(brillo.iloc[ini:i_pico + 1].values > umbral))


def detectar_inicio_constriccion(diametro, i_est, fps, ventana_base_s=2.0, k_sigma=3, n_sostenido=5):
    """Primer frame (desde el estímulo) con N frames consecutivos por debajo de base - k·σ.
    Devuelve (índice o None, info de línea base)."""
    d = np.asarray(diametro, dtype=float)
    n_base = int(round(ventana_base_s * fps))
    base = d[max(0, i_est - n_base): i_est]
    if np.isfinite(base).sum() < 3:
        base = d[:max(i_est, 3)]
    mu, sd = np.nanmedian(base), np.nanstd(base)
    umbral = mu - k_sigma * sd
    debajo = d < umbral
    info = {"base": mu, "sd": sd, "umbral": umbral}
    for i in range(i_est, len(d) - n_sostenido + 1):
        if debajo[i:i + n_sostenido].all():
            return i, info
    return None, info


def velocidad_constriccion(serie, t, i_est, i_constr, fps, ventana_base_s=2.0, vent=15, frac=0.9):
    """Velocidad media de constricción: caída de la línea base hasta `frac` de la caída total,
    dividida por el tiempo transcurrido desde el inicio de la constricción."""
    nan = {"base": np.nan, "minimo": np.nan, "i_fin": None, "duracion_s": np.nan, "velocidad": np.nan}
    if i_constr is None:
        return nan
    s = suavizar(serie, vent)
    n_base = int(round(ventana_base_s * fps))
    base = s.iloc[max(0, i_est - n_base): i_est].median()
    minimo = s.iloc[i_constr:].min()
    if not (base > minimo):
        return nan
    nivel = base - frac * (base - minimo)
    hit = s.iloc[i_constr:].values <= nivel
    i_fin = i_constr + int(np.argmax(hit))
    dur = t[i_fin] - t[i_constr]
    vel = (base - s.iloc[i_fin]) / dur if dur > 0 else np.nan
    return {"base": base, "minimo": minimo, "i_fin": i_fin, "duracion_s": dur, "velocidad": vel}


def velocidad_redilatacion(serie, t, i_desde, fps, ventana_final_s=2.0, vent=15,
                           frac=0.9, k_sigma=3, referencia_base=None):
    """Velocidad media de redilatación (10 % -> `frac` de la recuperación), desde el mínimo en adelante.

    Solo se informa si la serie recupera de forma apreciable (más que k·σ del ruido y, si se da,
    más del 20 % de la caída respecto de `referencia_base`). Si no, devuelve NaN.
    """
    nan = {"minimo": np.nan, "final": np.nan, "i_ini": None, "i_fin": None,
           "duracion_s": np.nan, "velocidad": np.nan}
    s = suavizar(serie, vent)
    resto = s.iloc[i_desde:]
    if resto.dropna().empty:
        return nan
    i_min = int(resto.idxmin())
    minimo = s.iloc[i_min]
    n_fin = int(round(ventana_final_s * fps))
    final = s.iloc[-n_fin:].median()
    recup = final - minimo
    # ruido de alta frecuencia: MAD del residuo entre la serie y su versión suavizada
    resid = np.asarray(serie, float) - s.values
    ruido = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    exigido = k_sigma * (0 if np.isnan(ruido) else ruido)
    if referencia_base is not None and referencia_base > minimo:
        exigido = max(exigido, 0.2 * (referencia_base - minimo))
    if not (recup > exigido and recup > 0):
        return {**nan, "minimo": minimo, "final": final}
    vals = s.iloc[i_min:].values
    i_fin = i_min + int(np.argmax(vals >= minimo + frac * recup))
    previos = np.where(s.iloc[i_min:i_fin + 1].values <= minimo + 0.1 * recup)[0]
    i_ini = i_min + int(previos[-1]) if len(previos) else i_min
    dur = t[i_fin] - t[i_ini]
    vel = (s.iloc[i_fin] - s.iloc[i_ini]) / dur if dur > 0 else np.nan
    return {"minimo": minimo, "final": final, "i_ini": i_ini, "i_fin": i_fin,
            "duracion_s": dur, "velocidad": vel}


def calcular_parametros(df, fps, cfg):
    """Parámetros fisiológicos del reflejo fotomotor.

    Devuelve (dict de parámetros, dict de eventos para graficar).
    Los parámetros que no se pueden calcular quedan como NaN (el CSV conserva siempre las mismas filas).
    """
    f = cfg["fisiologia"]
    t = df["tiempo_s"].values
    d, a = df["diametro_px"], df["area_px2"]
    nan = np.nan
    P = {k: nan for k in [
        "t_estimulo_s", "frame_estimulo", "t_inicio_constriccion_s", "latencia_ms",
        "diametro_basal_px", "diametro_minimo_px", "amplitud_diametro_px",
        "area_basal_px2", "area_minima_px2", "amplitud_area_px2", "amplitud_area_pct",
        "t_minimo_s", "velocidad_constriccion_diam_px_s", "velocidad_constriccion_area_px2_s",
        "duracion_constriccion_s", "t_inicio_redilatacion_s",
        "velocidad_redilatacion_diam_px_s", "velocidad_redilatacion_area_px2_s"]}
    ev = {"i_estimulo": None, "i_constriccion": None, "i_min": None,
          "i_redilatacion": None, "area_basal": nan, "area_min": nan}

    i_est = detectar_estimulo(df["brillo_medio"], fps, f["t_estimulo_s"])
    if i_est is None or d.isna().all():
        return P, ev
    ev["i_estimulo"] = i_est
    P["frame_estimulo"], P["t_estimulo_s"] = i_est, t[i_est]

    # --- latencia ---
    i_c, info = detectar_inicio_constriccion(d, i_est, fps, f["ventana_base_s"], f["k_sigma"], f["n_sostenido"])
    ev["i_constriccion"] = i_c
    if i_c is not None:
        P["t_inicio_constriccion_s"] = t[i_c]
        P["latencia_ms"] = (t[i_c] - t[i_est]) * 1000

    # --- amplitud ---
    n_base = int(round(f["ventana_base_s"] * fps))
    a_s, d_s = suavizar(a, f["ventana_mediana"]), suavizar(d, f["ventana_mediana"])
    desde = i_c if i_c is not None else i_est
    area_basal = a_s.iloc[max(0, i_est - n_base): i_est].median()
    diam_basal = d_s.iloc[max(0, i_est - n_base): i_est].median()
    i_min = int(a_s.iloc[desde:].idxmin())
    ev.update({"i_min": i_min, "area_basal": area_basal, "area_min": a_s.iloc[i_min]})
    P.update({"area_basal_px2": area_basal, "area_minima_px2": a_s.iloc[i_min],
              "amplitud_area_px2": area_basal - a_s.iloc[i_min],
              "amplitud_area_pct": 100 * (area_basal - a_s.iloc[i_min]) / area_basal,
              "diametro_basal_px": diam_basal, "diametro_minimo_px": d_s.iloc[desde:].min(),
              "amplitud_diametro_px": diam_basal - d_s.iloc[desde:].min(), "t_minimo_s": t[i_min]})
    if i_c is None:
        for k in ("amplitud_diametro_px", "amplitud_area_px2", "amplitud_area_pct"):
            P[k] = nan

    # --- velocidades ---
    vc_d = velocidad_constriccion(d, t, i_est, i_c, fps, f["ventana_base_s"], f["ventana_mediana"], f["fraccion_constriccion"])
    vc_a = velocidad_constriccion(a, t, i_est, i_c, fps, f["ventana_base_s"], f["ventana_mediana"], f["fraccion_constriccion"])
    P["velocidad_constriccion_diam_px_s"] = vc_d["velocidad"]
    P["velocidad_constriccion_area_px2_s"] = vc_a["velocidad"]
    P["duracion_constriccion_s"] = vc_d["duracion_s"]

    vr_d = velocidad_redilatacion(d, t, desde, fps, f["ventana_base_s"], f["ventana_mediana"],
                                  f["fraccion_redilatacion"], f["k_sigma"], referencia_base=diam_basal)
    vr_a = velocidad_redilatacion(a, t, desde, fps, f["ventana_base_s"], f["ventana_mediana"],
                                  f["fraccion_redilatacion"], f["k_sigma"], referencia_base=area_basal)
    P["velocidad_redilatacion_diam_px_s"] = vr_d["velocidad"]
    P["velocidad_redilatacion_area_px2_s"] = vr_a["velocidad"]
    if vr_d["i_ini"] is not None:
        P["t_inicio_redilatacion_s"] = t[vr_d["i_ini"]]
        ev["i_redilatacion"] = vr_d["i_ini"]
    ev["detalle_constriccion"], ev["detalle_redilatacion"] = vc_d, vr_d
    ev["linea_base_latencia"] = info
    return P, ev


UNIDADES = {"t_estimulo_s": "s", "frame_estimulo": "frame", "t_inicio_constriccion_s": "s",
            "latencia_ms": "ms", "diametro_basal_px": "px", "diametro_minimo_px": "px",
            "amplitud_diametro_px": "px", "area_basal_px2": "px2", "area_minima_px2": "px2",
            "amplitud_area_px2": "px2", "amplitud_area_pct": "%", "t_minimo_s": "s",
            "velocidad_constriccion_diam_px_s": "px/s", "velocidad_constriccion_area_px2_s": "px2/s",
            "duracion_constriccion_s": "s", "t_inicio_redilatacion_s": "s",
            "velocidad_redilatacion_diam_px_s": "px/s", "velocidad_redilatacion_area_px2_s": "px2/s"}


def parametros_a_tabla(P, extra=None):
    """DataFrame de formato fijo: parametro, valor, unidad."""
    filas = [(k, v, UNIDADES.get(k, "")) for k, v in P.items()]
    for k, (v, u) in (extra or {}).items():
        filas.append((k, v, u))
    return pd.DataFrame(filas, columns=["parametro", "valor", "unidad"])


def variabilidad_entre_registros(lista_parametros, nombres=None):
    """Media, desvío y CV (%) de cada parámetro entre varios registros/trials."""
    t = pd.DataFrame(lista_parametros, index=nombres)
    out = pd.DataFrame({"media": t.mean(), "desvio": t.std(), "n": t.count()})
    out["CV (%)"] = 100 * out["desvio"] / out["media"].abs()
    return t.T, out
