"""Módulo 2: segmentación de la pupila y morfología matemática."""
import cv2
import numpy as np
import pandas as pd
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.morphology import skeletonize
from skimage.segmentation import active_contour


def segmentar_otsu(gris, ksize=5, sigma=2):
    """Umbral global de Otsu. Devuelve (umbral, máscara 0/255 con la región oscura)."""
    blur = cv2.GaussianBlur(gris, (ksize, ksize), sigmaX=sigma)
    umbral, mascara = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return umbral, mascara


# --------------------------------------------------------------- K-Means
def segmentar_kmeans(roi_rgb, ks=range(2, 6)):
    """K-Means en el espacio HSV. Devuelve {k: máscara del cluster más oscuro (0/255)}."""
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    pixeles = hsv.reshape((-1, 3)).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    resultados = {}
    for k in ks:
        _, etiquetas, centros = cv2.kmeans(pixeles, k, None, criterio, 10, cv2.KMEANS_PP_CENTERS)
        etiquetas = etiquetas.flatten()
        oscuro = np.argmin(centros[:, 2])
        resultados[k] = ((etiquetas == oscuro).astype(np.uint8) * 255).reshape(hsv.shape[:2])
    return resultados

# --------------------------------------------------------------- buscar pupila
def estimar_radio_pupila(gris, centro, r_max=50, r_min=5, sigma=2):
    """Radio de la pupila a partir del perfil radial de intensidad (mediana sobre los 360°).
    El borde pupilar es donde el perfil sube más rápido (pupila oscura -> iris más claro)."""
    suave = cv2.GaussianBlur(gris, (0, 0), sigma)
    polar = cv2.warpPolar(suave, (r_max, 360), (float(centro[0]), float(centro[1])),
                          r_max, cv2.WARP_POLAR_LINEAR)           # filas = ángulo, columnas = radio
    perfil = np.median(polar, axis=0).astype(np.float64)
    perfil = cv2.GaussianBlur(perfil.reshape(1, -1), (0, 0), 1.5).ravel()
    return r_min + int(np.argmax(np.gradient(perfil)[r_min:]))

# --------------------------------------------------------------- Snakes
def segmentar_snake(gris, centro, radio=40, n_puntos=150, sigma=2,
                    alpha=0.05, beta=0.01, gamma=0.5, w_line=0.1, w_edge=1,
                    max_num_iter=1500, convergence=0.001):
    """Contorno activo desde una circunferencia inicial. Devuelve (init, snake, máscara 0/1)."""
    img = gaussian(gris.astype(np.float64) / 255.0, sigma=sigma)
    theta = np.linspace(0, 2 * np.pi, n_puntos)
    x = centro[0] + radio * np.cos(theta)
    y = centro[1] + radio * np.sin(theta)
    init = np.column_stack((y, x))
    snake = active_contour(img, init, alpha=alpha, beta=beta, gamma=gamma, w_line=w_line,
                           w_edge=w_edge, max_num_iter=max_num_iter, convergence=convergence)
    mascara = np.zeros(gris.shape, dtype=np.uint8)
    rr, cc = polygon(snake[:, 0], snake[:, 1], shape=gris.shape)
    mascara[rr, cc] = 1
    return init, snake, mascara

# ------------------------------------------------------------ Watershed
def preparar_para_watershed(gris, contraste=1.5, ksize=5):
    """Aumenta el contraste y suaviza (paso previo del Watershed)."""
    c = np.clip(gris.astype(np.float32) * contraste, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(c, (ksize, ksize), 0)


def gradiente_sobel(gris_suave, ksize=7):
    gx = cv2.Sobel(gris_suave, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gris_suave, cv2.CV_64F, 0, 1, ksize=ksize)
    g = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def marcadores_pupila(shape, cx, cy, radio=20, semiejes=(60, 55)):
    """1 = pupila segura (círculo), 2 = fondo seguro (fuera de la elipse)."""
    marc = np.zeros(shape, dtype=np.int32)
    cv2.circle(marc, (int(cx), int(cy)), radio, 1, -1)
    zona = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(zona, (int(cx), int(cy)), semiejes, 0, 0, 360, 1, -1)
    marc[zona == 0] = 2
    return marc


def segmentar_pupila(gris, cx, cy, radio=20, semiejes=(60, 55), cerrar=True):
    """Watershed con marcadores. Devuelve (máscara 0/255, contorno externo o None).

    cerrar=True agrega un cierre morfológico elíptico de 5x5 (como en el Módulo 3).
    """
    suave = preparar_para_watershed(gris)
    marc = marcadores_pupila(gris.shape, cx, cy, radio, semiejes)
    ws = cv2.watershed(cv2.cvtColor(suave, cv2.COLOR_GRAY2BGR), marc.copy())
    mascara = (ws == 1).astype(np.uint8) * 255
    if cerrar:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)
    contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contorno = max(contornos, key=cv2.contourArea) if contornos else None
    return mascara, contorno


# ------------------------------------------------------------ Morfología
def _k(tam):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))


def operaciones_morfologicas(mascara, tam=5):
    k = _k(tam)
    return {"Original": mascara,
            "Erosión": cv2.erode(mascara, k),
            "Dilatación": cv2.dilate(mascara, k),
            "Apertura": cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k),
            "Cierre": cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)}


def combinaciones_morfologicas(mascara, tam=5):
    """Operaciones simples + combinaciones secuenciales y lógicas (máscaras booleanas/0-255)."""
    k = _k(tam)
    base = operaciones_morfologicas(mascara, tam)
    ap, ci = base["Apertura"], base["Cierre"]
    abrir = lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    cerrar = lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    comb = dict(base)
    comb.update({
        "Apertura + cierre": cerrar(abrir(mascara)) > 0,
        "Cierre + apertura": abrir(cerrar(mascara)) > 0,
        "Erosión + dilatación": cv2.dilate(cv2.erode(mascara, k), k) > 0,
        "Dilatación + erosión": cv2.erode(cv2.dilate(mascara, k), k) > 0,
        "Intersección apertura/cierre": np.logical_and(ap > 0, ci > 0),
        "Unión apertura/cierre": np.logical_or(ap > 0, ci > 0),
    })
    return comb


def esqueletizar_contorno(mascara):
    """Devuelve (borde fino de 1-2 px, esqueleto del borde)."""
    k = _k(3)
    borde = cv2.subtract(mascara, cv2.erode(mascara, k))
    return borde, skeletonize(borde > 0)


# -------------------------------------------------------------- Métricas
def metricas_mascara(pred, referencia):
    """Dice, IoU (=Jaccard), cambio de área y píxeles distintos de `pred` respecto de `referencia`."""
    p, r = np.asarray(pred) > 0, np.asarray(referencia) > 0
    inter = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    suma = p.sum() + r.sum()
    return {
        "Área (px)": int(p.sum()),
        "Cambio de área (%)": 100 * (p.sum() - r.sum()) / r.sum() if r.sum() > 0 else 0.0,
        "Dice": 2 * inter / suma if suma > 0 else 1.0,
        "IoU (Jaccard)": inter / union if union > 0 else 1.0,
        "Píxeles modificados (%)": 100 * np.logical_xor(p, r).sum() / r.size,
    }


def tabla_metricas(mascaras, referencia):
    """DataFrame con las métricas de cada máscara contra `referencia`
    (anotación manual si se tiene; si no, la propia segmentación base)."""
    filas = {n: metricas_mascara(m, referencia) for n, m in mascaras.items()}
    return pd.DataFrame(filas).T


def mascara_desde_puntos(puntos, shape):
    """Máscara 0/255 de la elipse que mejor ajusta a >=5 puntos marcados a mano sobre el borde pupilar
    (coordenadas (x, y) de la ROI). Sirve como referencia manual para Dice/IoU."""
    elipse = cv2.fitEllipse(np.array(puntos, np.float32))
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, elipse, 255, -1)
    return m
