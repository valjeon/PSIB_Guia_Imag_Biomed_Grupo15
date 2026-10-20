"""Módulo 1: procesamiento de imagen, ruido/filtrado, bordes, Fourier, PSD y filtrado en frecuencia."""
import cv2
import numpy as np
from skimage import filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ------------------------------------------------- representación básica
def info_imagen(img):
    alto, ancho = img.shape[:2]
    return {"ancho": ancho, "alto": alto,
            "canales": img.shape[2] if img.ndim == 3 else 1,
            "bits_por_canal": img.dtype.itemsize * 8}


def submuestrear(img, factores=(0.50, 0.25, 0.05)):
    """Devuelve {factor: imagen reducida}."""
    return {f: cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for f in factores}


def cuantizar_bits(img, bits):
    """Cuantiza a `bits` por canal (manteniendo la escala 0-255)."""
    paso = 2 ** (8 - bits)
    return (img // paso) * paso


def separar_canales_rgb(img_rgb):
    return img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]


def convertir_hsv(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


# --------------------------------------------- brillo, contraste, histograma
def modificar_brillo_contraste(gris, brillo=50, contraste=1.5):
    con_brillo = np.clip(gris.astype(np.int16) + brillo, 0, 255).astype(np.uint8)
    con_contraste = np.clip(gris.astype(np.float32) * contraste, 0, 255).astype(np.uint8)
    return con_brillo, con_contraste


def transformacion_lineal(gris, f1=20, f2=200):
    out = (gris.astype(np.float32) - f1) / (f2 - f1) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def ecualizar(gris, clip_limit=2.0, tile=(8, 8)):
    """Ecualización global y CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    return {"global": cv2.equalizeHist(gris), "clahe": clahe.apply(gris)}


# ------------------------------------------------------ ruido y filtrado
def agregar_ruido(gris, sigma_gauss=20, prob_sp=0.02, sigma_speckle=0.2, seed=None):
    """Devuelve {'gaussiano', 'sal_pimienta', 'speckle'} (uint8)."""
    rng = np.random.default_rng(seed)
    img = gris.astype(np.float32)

    gauss = np.clip(img + rng.normal(0, sigma_gauss, img.shape), 0, 255).astype(np.uint8)

    sp = gris.copy()
    r = rng.random(sp.shape)
    sp[r < prob_sp / 2] = 0
    sp[r > 1 - prob_sp / 2] = 255

    speckle = np.clip(img + img * rng.normal(0, sigma_speckle, img.shape), 0, 255).astype(np.uint8)
    return {"gaussiano": gauss, "sal_pimienta": sp, "speckle": speckle}


def filtrar_espacial(img, k=5):
    """Filtros de media, mediana y gaussiano (ventana k x k)."""
    return {"media": cv2.blur(img, (k, k)),
            "mediana": cv2.medianBlur(img, k),
            "gaussiano": cv2.GaussianBlur(img, (k, k), 0)}


# ---------------------------------------------------------------- bordes
def detectar_bordes(img, canny_bajo=50, canny_alto=150):
    """Roberts, Sobel, Prewitt, Laplaciano y Canny."""
    return {"Roberts": filters.roberts(img),
            "Sobel": filters.sobel(img),
            "Prewitt": filters.prewitt(img),
            "Laplaciano": np.abs(cv2.Laplacian(img, cv2.CV_64F)),
            "Canny": cv2.Canny(img, canny_bajo, canny_alto)}


def continuidad_contorno_pupilar(borde, mascara_pupila, tolerancia=2, percentil=90):
    """Fracción (0-1) del contorno de la máscara de la pupila cubierta por píxeles de borde.

    Mide cuantitativamente la 'continuidad' del contorno pupilar detectado.
    """
    b = np.asarray(borde, dtype=np.float64)
    binario = (b > 0) if len(np.unique(b)) <= 2 else (b > np.percentile(b, percentil))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerancia + 1, 2 * tolerancia + 1))
    binario = cv2.dilate(binario.astype(np.uint8), k) > 0
    m = (np.asarray(mascara_pupila) > 0).astype(np.uint8)
    contorno = (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    if contorno.sum() == 0:
        return float("nan")
    return float(np.logical_and(contorno, binario).sum() / contorno.sum())


# ------------------------------------------------------------- Fourier 2D
def fft2d(gris):
    F = np.fft.fftshift(np.fft.fft2(gris))
    mag = np.abs(F)
    return {"F": F, "magnitud": mag, "magnitud_log": np.log1p(mag), "fase": np.angle(F)}


def reconstruir_desde_espectro(magnitud, fase):
    return np.real(np.fft.ifft2(np.fft.ifftshift(magnitud * np.exp(1j * fase))))


def reconstrucciones_fourier(img_a, img_b):
    """Reconstrucciones: solo magnitud, solo fase y cruzadas entre A y B (mismo tamaño)."""
    A, B = fft2d(img_a), fft2d(img_b)
    return {
        "solo_magnitud": reconstruir_desde_espectro(A["magnitud"], 0),
        "solo_fase": reconstruir_desde_espectro(1.0, A["fase"]),
        "magnitud_A_fase_B": reconstruir_desde_espectro(A["magnitud"], B["fase"]),
        "magnitud_B_fase_A": reconstruir_desde_espectro(B["magnitud"], A["fase"]),
    }


# ---------------------------------------------------------------------- PSD
def psd(gris):
    """Densidad espectral de potencia |F|^2 (centrada) y su versión logarítmica."""
    P = fft2d(gris)["magnitud"] ** 2
    return P, np.log1p(P)


def psd_radial(P):
    """Perfil radial promedio de la PSD (frecuencia radial en píxeles del espectro)."""
    h, w = P.shape
    y, x = np.indices(P.shape)
    r = np.hypot(x - w // 2, y - h // 2).astype(int)
    suma = np.bincount(r.ravel(), P.ravel())
    cuenta = np.bincount(r.ravel())
    return suma / np.maximum(cuenta, 1)


def fraccion_energia_baja_frecuencia(P, radio):
    h, w = P.shape
    y, x = np.indices(P.shape)
    D = np.hypot(x - w // 2, y - h // 2)
    return float(P[D <= radio].sum() / P.sum())


# ------------------------------------------------- filtrado en frecuencia
def filtro_frecuencia(gris, d0, tipo="pasa_bajos", forma="ideal"):
    """Filtro pasa-bajos / pasa-altos en el dominio frecuencial.

    forma: 'ideal' (corte abrupto) o 'gaussiano' (transición suave, sin ringing).
    Devuelve (imagen filtrada, máscara H).
    """
    h, w = gris.shape
    y, x = np.ogrid[:h, :w]
    D = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    if forma == "ideal":
        H = (D <= d0).astype(float)
    elif forma == "gaussiano":
        H = np.exp(-(D ** 2) / (2 * d0 ** 2))
    else:
        raise ValueError("forma debe ser 'ideal' o 'gaussiano'")
    if tipo == "pasa_altos":
        H = 1 - H
    F = np.fft.fftshift(np.fft.fft2(gris))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H))), H


# --------------------------------------------------------- calidad PSNR/SSIM
def a_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def metricas_calidad(referencia, imagen):
    """PSNR (dB) y SSIM de `imagen` respecto de `referencia` (escala de grises, 8 bits)."""
    ref, img = a_uint8(referencia), a_uint8(imagen)
    return {"PSNR (dB)": peak_signal_noise_ratio(ref, img, data_range=255),
            "SSIM": structural_similarity(ref, img, data_range=255)}
