"""Módulo 4: compresión sin pérdida (Huffman, RLE) y con pérdida (JPEG)."""
import heapq
from collections import Counter
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .modulo2 import segmentar_pupila


# ------------------------------------------------------------- Huffman
def huffman_codificar(imagen):
    """Devuelve (cadena de bits, tabla de códigos, frecuencias)."""
    datos = np.asarray(imagen, dtype=np.uint8).ravel()
    frecuencias = Counter(datos.tolist())
    heap, contador = [], 0
    for simbolo, frec in frecuencias.items():
        heapq.heappush(heap, (frec, contador, simbolo))
        contador += 1

    if len(heap) == 1:
        codigos = {heap[0][2]: "0"}
    else:
        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, contador, (n1, n2)))
            contador += 1
        codigos = {}

        def construir(nodo, prefijo=""):
            if isinstance(nodo, int):
                codigos[nodo] = prefijo or "0"
            else:
                construir(nodo[0], prefijo + "0")
                construir(nodo[1], prefijo + "1")

        construir(heap[0][2])

    # Se arma por tablas de búsqueda para que sea rápido
    tabla = [codigos.get(v, "") for v in range(256)]
    bits = "".join(tabla[v] for v in datos.tolist())
    return bits, codigos, frecuencias


def huffman_decodificar(bits, codigos, forma):
    inverso = {c: s for s, c in codigos.items()}
    datos, actual = [], ""
    for b in bits:
        actual += b
        if actual in inverso:
            datos.append(inverso[actual])
            actual = ""
    if actual:
        raise ValueError("La secuencia binaria no se pudo decodificar por completo.")
    return np.array(datos, dtype=np.uint8).reshape(forma)


def evaluar_huffman(imagen):
    """Codifica, decodifica y mide. Los bits NO incluyen el costo de guardar la tabla de códigos."""
    bits, codigos, frecuencias = huffman_codificar(imagen)
    recon = huffman_decodificar(bits, codigos, imagen.shape)
    bits_orig = imagen.size * 8
    return {"reconstruida": recon, "bits": len(bits), "bits_originales": bits_orig,
            "bytes": (len(bits) + 7) // 8, "n_simbolos": len(frecuencias),
            "ratio": bits_orig / len(bits), "reduccion_pct": 100 * (1 - len(bits) / bits_orig),
            "identica": bool(np.array_equal(imagen, recon))}


# ----------------------------------------------------------------- RLE
def rle_codificar(imagen, direccion="horizontal"):
    """Lista de (valor, longitud). 'horizontal' recorre fila por fila; 'vertical', columna por columna."""
    img = imagen if direccion == "horizontal" else imagen.T
    codif = []
    for linea in img:
        actual, largo = int(linea[0]), 1
        for v in linea[1:]:
            v = int(v)
            if v == actual:
                largo += 1
            else:
                codif.append((actual, largo))
                actual, largo = v, 1
        codif.append((actual, largo))
    return codif


def rle_decodificar(codif, forma, direccion="horizontal"):
    alto, ancho = forma
    datos = np.zeros(alto * ancho, dtype=np.uint8)
    i = 0
    for valor, largo in codif:
        datos[i:i + largo] = valor
        i += largo
    return datos.reshape(forma) if direccion == "horizontal" else datos.reshape(ancho, alto).T


def evaluar_rle(imagen, direccion="horizontal", bytes_valor=1, bytes_longitud=4):
    """Tamaño estimado = corridas × (bytes_valor + bytes_longitud). Con 1+4 se reproduce la estimación original."""
    codif = rle_codificar(imagen, direccion)
    recon = rle_decodificar(codif, imagen.shape, direccion)
    tam = len(codif) * (bytes_valor + bytes_longitud)
    return {"reconstruida": recon, "corridas": len(codif), "bytes": tam,
            "ratio": imagen.nbytes / tam, "reduccion_pct": 100 * (1 - tam / imagen.nbytes),
            "identica": bool(np.array_equal(imagen, recon))}


# ---------------------------------------------------------------- JPEG
def jpeg_comprimir(imagen, calidad):
    """Devuelve (imagen reconstruida, bytes del archivo JPEG). Acepta gris o BGR/RGB(3 canales)."""
    ok, buf = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
    if not ok:
        raise RuntimeError(f"No se pudo comprimir con Q={calidad}")
    flag = cv2.IMREAD_GRAYSCALE if imagen.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(buf, flag), len(buf)


def jpeg_rgb(roi_rgb, calidades=(90, 50, 10, 5)):
    """JPEG sobre la ROI a color. Devuelve ({Q: imagen RGB}, tabla de tamaños vs. PNG)."""
    bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    tam_png = len(cv2.imencode(".png", bgr)[1])
    imgs, filas = {}, []
    for q in calidades:
        rec_bgr, n = jpeg_comprimir(bgr, q)
        rec = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2RGB)
        imgs[q] = rec
        filas.append({"Calidad JPEG": q, "Tamaño JPEG (bytes)": n, "Tamaño PNG de referencia (bytes)": tam_png,
                      "Reducción respecto a PNG (%)": 100 * (1 - n / tam_png),
                      "Error absoluto medio": float(np.mean(np.abs(roi_rgb.astype(np.float32) - rec.astype(np.float32))))})
    return imgs, pd.DataFrame(filas)


def medida_bloques(imagen, bloque=8):
    """Índice de blocking: salto medio de intensidad en los bordes de los bloques de 8x8
    dividido por el salto medio en el resto de las posiciones. Conviene compararlo con el
    de la imagen original (ver `comparar_compresion`, que lo informa relativo al original)."""
    img = imagen.astype(np.float32)
    dx = np.abs(np.diff(img, axis=1)).mean(axis=0)          # salto entre columnas j y j+1
    dy = np.abs(np.diff(img, axis=0)).mean(axis=1)
    def razon(d):
        idx = np.arange(len(d))
        borde = (idx % bloque) == (bloque - 1)
        return d[borde].mean() / max(d[~borde].mean(), 1e-9)
    return float((razon(dx) + razon(dy)) / 2)


# --------------------------------------------- comparación de los esquemas
def comparar_compresion(gris, calidades=(90, 50, 10, 5), bytes_valor=1, bytes_longitud=4):
    """Huffman, RLE (H y V) y JPEG: tamaño, ratio, PSNR y SSIM contra el original (gris, 8 bits).

    Devuelve (tabla, dict de imágenes reconstruidas).
    """
    gris = gris.astype(np.uint8)
    original_bytes = gris.nbytes
    filas, recons = [], {}
    bloques_original = medida_bloques(gris)

    def agregar(nombre, recon, tam):
        recons[nombre] = recon
        idem = np.array_equal(gris, recon)
        filas.append({"Método": nombre, "Tamaño (bytes)": tam,
                      "Ratio de compresión": original_bytes / tam,
                      "Reducción (%)": 100 * (1 - tam / original_bytes),
                      "PSNR (dB)": np.inf if idem else peak_signal_noise_ratio(gris, recon, data_range=255),
                      "SSIM": structural_similarity(gris, recon, data_range=255),
                      "Índice de bloques (rel. al original)": medida_bloques(recon) / bloques_original,
                      "Reconstrucción idéntica": idem})

    h = evaluar_huffman(gris)
    agregar("Huffman", h["reconstruida"], h["bytes"])
    for d in ("horizontal", "vertical"):
        r = evaluar_rle(gris, d, bytes_valor, bytes_longitud)
        agregar(f"RLE {d}", r["reconstruida"], r["bytes"])
    for q in calidades:
        rec, n = jpeg_comprimir(gris, q)
        agregar(f"JPEG Q={q}", rec, n)
    return pd.DataFrame(filas), recons


def efecto_jpeg_en_diametro(gris, cx, cy, calidades=(50, 10)):
    """Mide la pupila (Watershed del Módulo 3) en la ROI original y en versiones JPEG.

    Devuelve (tabla con área/diámetro/cambio, dict de máscaras)."""
    imgs = {"Original": gris.astype(np.uint8)}
    for q in calidades:
        imgs[f"Q={q}"], _ = jpeg_comprimir(gris.astype(np.uint8), q)
    filas, mascaras = [], {}
    for nombre, im in imgs.items():
        mascara, cont = segmentar_pupila(im, cx, cy, radio=20)
        mascaras[nombre] = mascara
        area = cv2.contourArea(cont) if cont is not None else np.nan
        filas.append({"Imagen": nombre, "Área (px²)": area,
                      "Diámetro equivalente (px)": 2 * np.sqrt(area / np.pi) if area == area else np.nan})
    tabla = pd.DataFrame(filas).set_index("Imagen")
    d0 = tabla.loc["Original", "Diámetro equivalente (px)"]
    tabla["Cambio diámetro (px)"] = tabla["Diámetro equivalente (px)"] - d0
    tabla["Cambio diámetro (%)"] = 100 * tabla["Cambio diámetro (px)"] / d0
    return tabla, mascaras, imgs
