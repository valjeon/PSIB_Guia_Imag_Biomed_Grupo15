from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import heapq
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = str(BASE_DIR / "data" / "raw" / "Registro A_2.mov")
VIDEO_PATH_2 = str(BASE_DIR / "data" / "raw" / "Registro B_1.mov")

# VIDEO_PATH = "../data/raw/Registro A_2.MOV"
# VIDEO_PATH_2 = "../data/raw/Registro B_1.MOV"
# BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "resultados_ojo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROI_PATH = str(OUT_DIR / "roi_A_gray.png")
ROI_RGB_PATH = str(OUT_DIR / "roi_A_rgb.png")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")
print("Video abierto correctamente.")

# Estado compartido entre módulos
ROI_RECT = None                 # (x, y, ancho, alto) en el frame completo
PUPIL_CENTER = None             # (x, y) dentro de la ROI
roi_rgb = None
roi_gray = None
frame_rgb = None
frame_bgr = None


def display(obj):
    """Muestra un objeto en Jupyter o lo imprime en la terminal."""
    try:
        from IPython.display import display as ipy_display
        ipy_display(obj)
    except Exception:
        print(obj)


def preparar_roi():
    """
    Permite elegir el video, un fotograma, la ROI del ojo
    y el centro aproximado de la pupila.
    """
    global VIDEO_PATH, ROI_RECT, PUPIL_CENTER
    global roi_rgb, roi_gray, frame_rgb, frame_bgr

    # 1. Seleccionar el video
    ruta = input(
        f"Ruta del video [Enter = {VIDEO_PATH}]: "
    ).strip() or VIDEO_PATH

    ruta = str(Path(ruta).expanduser().resolve())

    cap = cv2.VideoCapture(ruta)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta}")

    # Guardar la ruta elegida para que la utilicen los demás módulos
    VIDEO_PATH = ruta

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if n_frames <= 0:
        cap.release()
        raise RuntimeError("El video no contiene fotogramas legibles.")

    sugerido = n_frames // 2

    print(
        f"Resolución: {width} x {height} | "
        f"FPS: {fps:.2f} | Frames: {n_frames}"
    )

    # 2. Elegir el fotograma
    dato = input(
        f"Número de frame [Enter = {sugerido}]: "
    ).strip()

    try:
        frame_index = sugerido if not dato else int(dato)
    except ValueError:
        cap.release()
        raise ValueError("El número de frame debe ser un entero.")

    frame_index = int(np.clip(frame_index, 0, n_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None or frame_bgr.size == 0:
        raise RuntimeError("No se pudo leer el fotograma seleccionado.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 3. Seleccionar la ROI con el mouse mediante Matplotlib
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(frame_rgb)
    ax.set_title(
        "Mantené el clic izquierdo y arrastrá sobre el ojo. "
        "Cerrá la ventana para confirmar."
    )
    ax.axis("off")

    estado = {
        "inicio": None,
        "rectangulo": None,
        "roi": None
    }

    def al_presionar(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        estado["inicio"] = (event.xdata, event.ydata)

        if estado["rectangulo"] is not None:
            estado["rectangulo"].remove()

        rect = Rectangle(
            estado["inicio"],
            0,
            0,
            fill=False,
            edgecolor="red",
            linewidth=2
        )

        ax.add_patch(rect)
        estado["rectangulo"] = rect
        fig.canvas.draw_idle()

    def al_mover(event):
        if estado["inicio"] is None:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = estado["inicio"]
        x1, y1 = event.xdata, event.ydata

        rect = estado["rectangulo"]
        rect.set_xy((min(x0, x1), min(y0, y1)))
        rect.set_width(abs(x1 - x0))
        rect.set_height(abs(y1 - y0))

        fig.canvas.draw_idle()

    def al_soltar(event):
        if estado["inicio"] is None:
            return

        if event.xdata is None or event.ydata is None:
            estado["inicio"] = None
            return

        x0, y0 = estado["inicio"]
        x1, y1 = event.xdata, event.ydata

        x = int(round(min(x0, x1)))
        y = int(round(min(y0, y1)))
        w = int(round(abs(x1 - x0)))
        h = int(round(abs(y1 - y0)))

        if w > 5 and h > 5:
            # Limitar la selección a los bordes del fotograma
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)

            if w > 5 and h > 5:
                estado["roi"] = (x, y, w, h)
                ax.set_title(
                    "ROI seleccionada. Cerrá la ventana para confirmar."
                )
                fig.canvas.draw_idle()

        estado["inicio"] = None

    fig.canvas.mpl_connect("button_press_event", al_presionar)
    fig.canvas.mpl_connect("motion_notify_event", al_mover)
    fig.canvas.mpl_connect("button_release_event", al_soltar)

    plt.show(block=True)

    if estado["roi"] is None:
        plt.close(fig)
        raise RuntimeError(
            "No se seleccionó una ROI válida. "
            "Volvé a ejecutar la selección."
        )

    ROI_RECT = estado["roi"]
    plt.close(fig)

    x, y, w, h = ROI_RECT
    roi_rgb = frame_rgb[y:y + h, x:x + w].copy()

    if roi_rgb.size == 0:
        raise RuntimeError("La ROI seleccionada está vacía.")

    # 4. Seleccionar el centro aproximado de la pupila
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(roi_rgb)
    ax.set_title("Hacé clic en el centro aproximado de la pupila")
    ax.axis("on")

    puntos = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not puntos:
        raise RuntimeError("No se marcó el centro de la pupila.")

    px, py = puntos[0]

    PUPIL_CENTER = (
        int(round(np.clip(px, 0, roi_rgb.shape[1] - 1))),
        int(round(np.clip(py, 0, roi_rgb.shape[0] - 1)))
    )

    # 5. Generar y guardar las imágenes de referencia
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(ROI_PATH), roi_gray):
        raise RuntimeError("No se pudo guardar la ROI en escala de grises.")

    if not cv2.imwrite(
        str(ROI_RGB_PATH),
        cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError("No se pudo guardar la ROI en color.")

    print(f"Video seleccionado: {VIDEO_PATH}")
    print(f"Fotograma seleccionado: {frame_index}")
    print(f"ROI seleccionada: {ROI_RECT}")
    print(f"Centro pupilar dentro de la ROI: {PUPIL_CENTER}")
    print(f"Imágenes guardadas en: {OUT_DIR.resolve()}")


def asegurar_roi():
    """Comprueba que exista una ROI antes de ejecutar un módulo."""
    if (
        globals().get("roi_rgb") is None
        or globals().get("roi_gray") is None
    ):
        preparar_roi()

def ejecutar_modulo(numero):
    asegurar_roi()
    print(f"\n{'='*20} MÓDULO {numero} {'='*20}\n")
    funciones = {1: modulo_1, 2: modulo_2, 3: modulo_3, 4: modulo_4}
    funciones[numero]()

def modulo_1():
    roi_height, roi_width = roi_rgb.shape[:2]
    # Profundidad de bits por canal
    bits_por_canal = roi_rgb.dtype.itemsize * 8

    print(f"Ancho de la ROI: {roi_width} píxeles")
    print(f"Alto de la ROI: {roi_height} píxeles")
    print(f"Canales: {roi_rgb.shape[2]}")
    print(f"Profundidad: {bits_por_canal} bits por canal")

    # Submuestro de la ROI al 50%, 25% y 5%
    roi_50 = cv2.resize(roi_rgb, None, fx=0.50, fy=0.50, interpolation=cv2.INTER_AREA)
    roi_25 = cv2.resize(roi_rgb, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    roi_05 = cv2.resize(roi_rgb, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)

    print("ROI original:", roi_rgb.shape)
    print("ROI 50%:", roi_50.shape)
    print("ROI 25%:", roi_25.shape)
    print("ROI 5%:", roi_05.shape)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(roi_rgb)
    axes[0].set_title("Original")

    axes[1].imshow(roi_50)
    axes[1].set_title("50%")

    axes[2].imshow(roi_25)
    axes[2].set_title("25%")

    axes[3].imshow(roi_05)
    axes[3].set_title("5%")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Cuantización a 4 bits
    roi_4bits = (roi_rgb // 16) * 16

    # Cuantización a 2 bits
    roi_2bits = (roi_rgb // 64) * 64

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(roi_rgb)
    axes[0].set_title("Original — 8 bits")

    axes[1].imshow(roi_4bits)
    axes[1].set_title("Cuantizada — 4 bits")

    axes[2].imshow(roi_2bits)
    axes[2].set_title("Cuantizada — 2 bits")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Espacios de color --> separar los canales y convertir de RGB a HSV
    R = roi_rgb[:, :, 0]
    G = roi_rgb[:, :, 1]
    B = roi_rgb[:, :, 2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(R, cmap="gray")
    axes[0].set_title("Canal R")

    axes[1].imshow(G, cmap="gray")
    axes[1].set_title("Canal G")

    axes[2].imshow(B, cmap="gray")
    axes[2].set_title("Canal B")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Convetimos la ROI a escala de grises
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("../data/reference/roi_A_gray.png", roi_gray)
    plt.figure(figsize=(10, 6))
    plt.imshow(roi_gray, cmap="gray")
    plt.axis("off")
    plt.title("ROI en escala de grises")
    plt.show()

    # Convertimos a HSV
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    H = roi_hsv[:, :, 0]
    S = roi_hsv[:, :, 1]
    V = roi_hsv[:, :, 2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(H, cmap="gray")
    axes[0].set_title("H — Matiz")

    axes[1].imshow(S, cmap="gray")
    axes[1].set_title("S — Saturación")

    axes[2].imshow(V, cmap="gray")
    axes[2].set_title("V — Valor")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"H: {H.min()} → {H.max()}")
    print(f"S: {S.min()} → {S.max()}")
    print(f"V: {V.min()} → {V.max()}")

    # HISTOGRAMA

    plt.figure(figsize=(10, 5))

    plt.hist(roi_gray.ravel(), bins=256, range=(0, 256))

    plt.title("Histograma de la ROI en escala de grises")
    plt.xlabel("Intensidad de gris")
    plt.ylabel("Cantidad de píxeles")
    plt.xlim(0, 255)

    plt.show()

    # Modificamos el brillo y el contraste

    roi_brillo = np.clip(roi_gray.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    roi_contraste = np.clip(roi_gray.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(roi_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")

    axes[1].imshow(roi_brillo, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Brillo +50")

    axes[2].imshow(roi_contraste, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Contraste ×1.5")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(roi_gray.ravel(), bins=256, range=(0, 256))
    axes[0].set_title("Original")

    axes[1].hist(roi_brillo.ravel(), bins=256, range=(0, 256))
    axes[1].set_title("Brillo +50")

    axes[2].hist(roi_contraste.ravel(), bins=256, range=(0, 256))
    axes[2].set_title("Contraste ×1.5")

    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 500)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Cantidad de píxeles")

    plt.tight_layout()
    plt.show()

    # Al aumentarle el brillo sumandole una constante al histograma, se corre todo el espectro en conjunto. 
    # En cambio, cuando mejoramos el contraste de la imagen multiplicando por un factor de 1.5, se observa un efecto similar a una ecualización. 
    # Los extremos del espectro de nuestra imagen se corren a los extremos del histograma, usándose mejor el rango de 0 a 255. 
    # Escalado por rango de interés

    plt.figure(figsize=(12, 7))

    plt.imshow(roi_gray, cmap="gray", vmin=0, vmax=255)
    plt.colorbar(label="Intensidad")
    plt.title("ROI en escala de grises")
    plt.axis("off")

    plt.show()

    f1 = 20   
    f2 = 200 

    roi_lineal = (roi_gray.astype(np.float32) - f1) / (f2 - f1) * 255

    roi_lineal = np.clip(roi_lineal, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(roi_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")

    axes[1].imshow(roi_lineal, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Transformación lineal [{f1:.0f}, {f2:.0f}]")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(roi_gray.ravel(), bins=256, range=(0, 256))
    axes[0].set_title("Histograma original")

    axes[1].hist(roi_lineal.ravel(), bins=256, range=(0, 256))
    axes[1].set_title("Histograma transformado")

    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 500)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Cantidad de píxeles")

    plt.tight_layout()
    plt.show()

    # Ecualización del histograma

    roi_equalizada = cv2.equalizeHist(roi_gray)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    roi_clahe = clahe.apply(roi_gray)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].imshow(roi_lineal, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Lineal")

    axes[1].imshow(roi_equalizada, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Ecualización global")

    axes[2].imshow(roi_equalizada, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Ecualización Clahe")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # La ecualización global y CLAHE se ven con menor nitidez con respecto a la transformación lineal. Además, hacen que la parte de las pestañas se vean igual de oscuras que la pupila haciendo que se vean más juntas y se distinga menos la separación. 
    # Se podría decir que se le agrega ruido gaussiano. 
    # 
    # La transformación lineal es la que nos permite distinguir mejor entre la pupila y el iris. CLAHE es la que da peores resultados.

    # %%
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].hist(roi_gray.ravel(), bins=256, range=(0, 256))
    axes[0].set_title("Original")

    axes[1].hist(roi_lineal.ravel(), bins=256, range=(0, 256))
    axes[1].set_title("Transformación lineal")

    axes[2].hist(roi_clahe.ravel(), bins=256, range=(0, 256))
    axes[2].set_title("Ecualización CLAHE")

    axes[3].hist(roi_equalizada.ravel(), bins=256, range=(0, 256))
    axes[3].set_title("Ecualización global")

    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 500)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Cantidad de píxeles")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Como en la transformación lineal, se eligen manualmente los extremos del histograma, se observa que en la transformación hay una redistribución general del original. 
    # CLAHE, por otro lado, al hacer una ecualización local en las regiones donde hay menor variación, el histograma no queda completamente distribuido de 0 a 255. 
    # En la ecualización global, se observan más amplificados los extremos que en la transformación lineal. Esto explica por qué en la imagen, se ven más oscurecidos ciertas zonas como la de las pestañas. 

    # %% [markdown]
    # Análisis de ruido y filtrado espacial

    # %%
    #Gaussiano
    img = roi_gray.astype(np.float32)

    mu = 0
    sigma = 20

    ruido_gaussiano = np.random.normal(mu, sigma, img.shape)

    img_gaussiano = img + ruido_gaussiano
    img_gaussiano = np.clip(img_gaussiano, 0, 255).astype(np.uint8)

    #Sal y pimienta
    img_sp = roi_gray.copy()

    probabilidad = 0.02

    random = np.random.random(img_sp.shape)

    img_sp[random < probabilidad / 2] = 0
    img_sp[random > 1 - probabilidad / 2] = 255

    #Speckle
    img = roi_gray.astype(np.float32)

    sigma_speckle = 0.2

    ruido_speckle = np.random.normal(
        0,
        sigma_speckle,
        img.shape
    )

    img_speckle = img + img * ruido_speckle
    img_speckle = np.clip(img_speckle, 0, 255).astype(np.uint8)

    #Comparamos los 3
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(roi_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")

    axes[1].imshow(img_gaussiano, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Ruido gaussiano")

    axes[2].imshow(img_sp, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Sal y pimienta")

    axes[3].imshow(img_speckle, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Speckle")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Aplicamos filtros de media, mediana y gaussiano a las 3 imágenes:

    # %%
    img_gauss_media = cv2.blur(img_gaussiano, (5, 5))
    img_sp_media = cv2.blur(img_sp, (5, 5))
    img_speckle_media = cv2.blur(img_speckle, (5, 5))

    img_gauss_mediana = cv2.medianBlur(img_gaussiano, 5)
    img_sp_mediana = cv2.medianBlur(img_sp, 5)
    img_speckle_mediana = cv2.medianBlur(img_speckle, 5)

    img_gauss_gaussiano = cv2.GaussianBlur(img_gaussiano, (5, 5), 0)

    img_sp_gaussiano = cv2.GaussianBlur(img_sp, (5, 5), 0)

    img_speckle_gaussiano = cv2.GaussianBlur(img_speckle, (5, 5), 0)


    # %%
    fig, axes = plt.subplots(2, 4, figsize=(18, 5))

    axes[0][0].imshow(img_gaussiano, cmap="gray", vmin=0, vmax=255)
    axes[0][0].set_title("Con ruido gaussiano")

    axes[0][1].imshow(img_gauss_media, cmap="gray", vmin=0, vmax=255)
    axes[0][1].set_title("Filtro de media")

    axes[0][2].imshow(img_gauss_mediana, cmap="gray", vmin=0, vmax=255)
    axes[0][2].set_title("Filtro de mediana")

    axes[0][3].imshow(img_gauss_gaussiano, cmap="gray", vmin=0, vmax=255)
    axes[0][3].set_title("Filtro gaussiano")

    axes[1][0].hist(img_gaussiano.ravel(), bins=256, range=(0, 256))


    axes[1][1].hist(img_gauss_media.ravel(), bins=256, range=(0, 256))


    axes[1][2].hist(img_gauss_mediana.ravel(), bins=256, range=(0, 256))


    axes[1][3].hist(img_gauss_gaussiano.ravel(), bins=256, range=(0, 256))




    plt.tight_layout()
    plt.show()

    # Se puede observar que los filtros de media y de mediana, suavizan más la imagen que el gaussiano perdiéndose nitidez. 
    # En el caso de esta imagen, al ver los histogramas se muestra que los 3 filtros eliminan de forma similar el ruido gaussiano. 

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img_sp, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Sal y pimienta")

    axes[1].imshow(img_sp_media, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Filtro de media")

    axes[2].imshow(img_sp_mediana, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Filtro de mediana")

    axes[3].imshow(img_sp_gaussiano, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Filtro gaussiano")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # El filtro que mejor elimina el ruido sal y pimienta es el filtro de mediana como se vio durante el curso. Con los filtros de media y el gaussiano, quedan en la imagen los puntos blancos y oscuros más suavizados pero no se llegan a eliminar por completo. 

    # %%
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img_speckle, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Speckle")

    axes[1].imshow(img_speckle_media, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Filtro de media")

    axes[2].imshow(img_speckle_mediana, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Filtro de mediana")

    axes[3].imshow(img_speckle_gaussiano, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Filtro gaussiano")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Para el ruido speckle, los filtros de media y mediana dan resultados similares. Con el de media, se obtiene el mayor suavizado pero se pierde bastante nitidez. Mientras que con el filtro gaussiano, se mantiene la nitdez pero se sigue observando considerablemente moteada la imagen. 

    # %% [markdown]
    # Detección de bordes

    # %%
    from skimage import filters

    def detectar_bordes(img):
        roberts = filters.roberts(img)
        sobel = filters.sobel(img)
        prewitt = filters.prewitt(img)

        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = np.abs(laplacian)

        canny = cv2.Canny(img, 50, 150)

        return roberts, sobel, prewitt, laplacian, canny

    bordes_original = detectar_bordes(roi_gray)

    bordes_media = detectar_bordes(img_gauss_media)
    bordes_mediana = detectar_bordes(img_gauss_mediana)
    bordes_gaussiano = detectar_bordes(img_gauss_gaussiano)

    def mostrar_bordes(bordes, titulo):
        nombres = ["Roberts", "Sobel", "Prewitt", "Laplaciano", "Canny"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        axes = axes.ravel()

        for ax, borde, nombre in zip(axes, bordes, nombres):
            ax.imshow(borde, cmap="gray")
            ax.set_title(nombre, fontsize=14)
            ax.axis("off")

        # Ocultar el sexto espacio
        axes[5].axis("off")

        fig.suptitle(titulo, fontsize=18)
        plt.tight_layout()
        plt.show()

    mostrar_bordes(bordes_original, "Imagen original")
    mostrar_bordes(bordes_media, "Después del filtro de media")
    mostrar_bordes(bordes_mediana, "Después del filtro de mediana")
    mostrar_bordes(bordes_gaussiano, "Después del filtro gaussiano")

    # %% [markdown]
    # El detector de bordes que mantiene el borde de la pupila más continua es el Canny. Aún así, no se logra un borde totalmente continuo. 
    # 
    # Con el Canny aplicado sobre la imagen original se obtiene el mejor contorno de ambos la pupila y el iris. 

    # %% [markdown]
    # Transformada de Fourier

    # %%
    # Transformada de Fourier 2D
    F = np.fft.fft2(roi_gray)

    # Desplazamos las bajas frecuencias al centro
    F_shift = np.fft.fftshift(F)

    magnitude = np.abs(F_shift)
    magnitude_log = np.log1p(magnitude)

    plt.figure(figsize=(10, 6))

    plt.imshow(magnitude_log, cmap="gray")
    plt.title("Magnitud del espectro de Fourier 2D")
    plt.axis("off")
    plt.colorbar(label="log(1 + |F|)")

    plt.show()

    # %%
    phase = np.angle(F_shift)
    plt.figure(figsize=(10, 6))

    plt.imshow(phase, cmap="gray")
    plt.title("Fase del espectro de Fourier 2D")
    plt.axis("off")
    plt.colorbar(label="Fase [rad]")

    plt.show()

    # %% [markdown]
    # Reconstruimos la imagen a través de la transformada inversa:

    # %%
    #Tranformada de Fourier de la imagen original

    F_original = np.fft.fft2(roi_gray)
    F_original_shift = np.fft.fftshift(F_original)

    magnitud_original = np.abs(F_original_shift)
    fase_original = np.angle(F_original_shift)


    # 2. Transformada de Fourier de la imagen con ruido gaussiano + filtro gaussiano

    F_filtrada = np.fft.fft2(img_gauss_gaussiano)
    F_filtrada_shift = np.fft.fftshift(F_filtrada)

    magnitud_filtrada = np.abs(F_filtrada_shift)
    fase_filtrada = np.angle(F_filtrada_shift)

    # Reconstrucción usando solamente la magnitud

    F_solo_magnitud = magnitud_original * np.exp(1j * 0)
    recon_magnitud = np.fft.ifft2(
        np.fft.ifftshift(F_solo_magnitud)
    )
    recon_magnitud = np.real(recon_magnitud)

    # Reconstrucción usando solamente la fase

    F_solo_fase = np.exp(1j * fase_original)
    recon_fase = np.fft.ifft2(
        np.fft.ifftshift(F_solo_fase)
    )
    recon_fase = np.real(recon_fase)


    # Reconstrucción cruzada --> magnitud original + fase filtrada

    F_cruzada_1 = magnitud_original * np.exp(1j * fase_filtrada)
    recon_cruzada_1 = np.fft.ifft2(
        np.fft.ifftshift(F_cruzada_1)
    )
    recon_cruzada_1 = np.real(recon_cruzada_1)

    # Otra reconstrucción cruzada: magnitud filtrada + fase original

    F_cruzada_2 = magnitud_filtrada * np.exp(1j * fase_original)
    recon_cruzada_2 = np.fft.ifft2(
        np.fft.ifftshift(F_cruzada_2)
    )
    recon_cruzada_2 = np.real(recon_cruzada_2)


    #Visualización

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(
        np.log1p(magnitud_original),
        cmap="gray"
    )
    axes[0].set_title("Magnitud — imagen original")

    axes[1].imshow(
        fase_original,
        cmap="gray"
    )
    axes[1].set_title("Fase — imagen original")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(
        recon_magnitud,
        cmap="gray"
    )
    axes[0, 0].set_title("Reconstrucción — solo magnitud")

    axes[0, 1].imshow(
        recon_fase,
        cmap="gray"
    )
    axes[0, 1].set_title("Reconstrucción — solo fase")

    axes[1, 0].imshow(
        recon_cruzada_1,
        cmap="gray"
    )
    axes[1, 0].set_title(
        "Magnitud original + fase filtrada"
    )

    axes[1, 1].imshow(
        recon_cruzada_2,
        cmap="gray"
    )
    axes[1, 1].set_title(
        "Magnitud filtrada + fase original"
    )

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Densidad espectral de potencia

    # %%
    # PSD de la imagen original
    PSD_original = np.abs(F_original_shift)**2

    # PSD de la imagen filtrada
    PSD_filtrada = np.abs(F_filtrada_shift)**2

    # Lo pasamos a escala logarítmica para visualizar

    PSD_original_log = np.log1p(PSD_original)
    PSD_filtrada_log = np.log1p(PSD_filtrada)

    # Visualización

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(
        PSD_original_log,
        cmap="gray"
    )
    axes[0].set_title("PSD — imagen original")
    axes[0].axis("off")

    axes[1].imshow(
        PSD_filtrada_log,
        cmap="gray"
    )
    axes[1].set_title("PSD — imagen filtrada")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Se observa que en la PSD de la imagen filtrada, la energía queda más concentrada en la región de las bajas frecuencias. Mientras que en la de la imagen original, se distribute de forma más uniforme. Esto muestra que el filtro gaussiano atenúa las componentes de alta frecuencia de la imagen. 
    # Sin embargo, como los bordes son regiones donde hay una alta variación de la intensidad y esto implica que en el dominio frecuencial tienen componentes de altas frecuencias, el filtro gaussiano puede hacer que perdamos información de los bordes. 

    # %% [markdown]
    # Filtrado en frecuencia

    # %%
    h, w = roi_gray.shape

    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    # Distancia al centro del espectro
    D = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Frecuencias de corte
    D0_bajos = 40
    D0_altos = 30


    # FILTRO PASA-BAJOS
    filtro_bajos = D <= D0_bajos
    F_bajos = F_original_shift * filtro_bajos
    img_bajos = np.fft.ifft2(
        np.fft.ifftshift(F_bajos)
    )
    img_bajos = np.real(img_bajos)

    # FILTRO PASA-ALTOS
    filtro_altos = D >= D0_altos
    F_altos = F_original_shift * filtro_altos
    img_altos = np.fft.ifft2(
        np.fft.ifftshift(F_altos)
    )
    img_altos = np.real(img_altos)


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(filtro_bajos, cmap="gray")
    axes[0].set_title(f"Pasa-bajos — D₀ = {D0_bajos}")
    axes[0].axis("off")

    axes[1].imshow(filtro_altos, cmap="gray")
    axes[1].set_title(f"Pasa-altos — D₀ = {D0_altos}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


    #resultados
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(roi_gray, cmap="gray")
    axes[0].set_title("Imagen original")

    axes[1].imshow(img_bajos, cmap="gray")
    axes[1].set_title(f"Pasa-bajos — D₀ = {D0_bajos}")

    axes[2].imshow(img_altos, cmap="gray")
    axes[2].set_title(f"Pasa-altos — D₀ = {D0_altos}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # COMPARACIÓN: FILTRADO ESPACIAL VS FILTRADO FRECUENCIAL

    img_gaussiano = cv2.GaussianBlur(img, (5, 5), 0)

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.imshow(img_gaussiano, cmap="gray")
    axes.set_title("Imagen filtrada con gaussiano")

    axes.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Al comparar la imagen original + pasabajos y la imagen filtrada con gaussiano, se observa que con el filtro gaussiano se obtiene mayor suavizado y se pierde nitidez. El filtro pasabajo, logra un suavizado de menor intensidad. Manteniéndose mejor los bordes. 

    # %% [markdown]
    # CáLculo de PSNR y SSIM

    # %%
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # PSNR
    psnr_media = peak_signal_noise_ratio(roi_gray, img_gauss_media)
    psnr_mediana = peak_signal_noise_ratio(roi_gray, img_gauss_mediana)
    psnr_gaussiano = peak_signal_noise_ratio(roi_gray, img_gauss_gaussiano)

    # SSIM
    ssim_media = structural_similarity(roi_gray, img_gauss_media)
    ssim_mediana = structural_similarity(roi_gray, img_gauss_mediana)
    ssim_gaussiano = structural_similarity(roi_gray, img_gauss_gaussiano)

    print("Filtro de media:")
    print(f"  PSNR = {psnr_media:.2f} dB")
    print(f"  SSIM = {ssim_media:.4f}")

    print("\nFiltro de mediana:")
    print(f"  PSNR = {psnr_mediana:.2f} dB")
    print(f"  SSIM = {ssim_mediana:.4f}")

    print("\nFiltro gaussiano:")
    print(f"  PSNR = {psnr_gaussiano:.2f} dB")
    print(f"  SSIM = {ssim_gaussiano:.4f}")

    # %% [markdown]
    # Cuanto mayor es el valor de la PSNR, menor es la diferencia de la imagen filtrada respecto de la imagen original. Y cuanto más se acerque el valor de SSIM a 1, mayor es la simillitud entructural entre las imágenes en cuanto a brillo, contraste y estructura.
    # 
    # Se obtuvo que la imagen filtrada que más se parece a la original es a la que se le aplicó el filtro de media.

def modulo_2():

    # Se carga el video y la ROI seleccionada en el módulo 1 para aplicarle los métodos de segmentación y observar su funcionamiento. 
    # Se aplica el método de Otsu para determinar automáticamente un umbral que permita separar la región oscura correspondiente a la pupila del resto de la imagen. 
    # Luego se realiza una operación morfológica para eliminar pequeños objetos y suavizar la región segmentada.

    # Otsu sin operaciones morfológicas
    blur = cv2.GaussianBlur(roi_gray, (5, 5), sigmaX=2)
    threshold, mask_otsu_original = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    print("Umbral de Otsu:", threshold)

    plt.figure(figsize=(6, 5))
    plt.imshow(mask_otsu_original, cmap="gray")
    plt.title("Máscara original de Otsu")
    plt.axis("off")
    plt.show()

    # Se utiliza el algoritmo K-Means para agrupar los píxeles de la imagen en distintos grupos según sus características de color e intensidad. Se evalúan valores de K entre 2 y 5 para analizar cómo cambia la segmentación de la pupila al aumentar la cantidad de grupos.

    # Se utiliza la ROI RGB seleccionada en el menú.
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    results_kmeans = {}

    for k in range(2, 6):
        _, labels, centers = cv2.kmeans(pixels,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)

        labels = labels.flatten()
        centers = centers.astype(np.uint8)

        cluster_image = centers[labels]
        cluster_image = cluster_image.reshape(hsv.shape)

        value_image = cluster_image[:, :, 2]

        darkest_cluster = np.argmin(centers[:, 2])

        mask = (labels == darkest_cluster).astype(np.uint8) * 255
        mask = mask.reshape(hsv.shape[:2])

        results_kmeans[k] = mask

    plt.figure(figsize=(15, 8))

    for i, k in enumerate(range(2, 6)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(results_kmeans[k], cmap="gray")
        plt.title(f"K-Means, K={k}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Usamos un contorno activo para localizar los bordes de la pupila. Definimos manualmente una curva cerrada próxima al borde pupilar y dejamos que evolucione en función de las características de la imagen hasta converger hacia la frontera de la estructura.



    # %%
    gray_eye = roi_gray.astype(np.float64) / 255.0
    smooth_eye = gaussian(gray_eye, sigma=2)

    center_x = PUPIL_CENTER[0]
    center_y = PUPIL_CENTER[1]

    # Un poco más grande que la pupila
    radius_x = 40
    radius_y = 40
    theta = np.linspace(0, 2*np.pi, 150)

    x = center_x + radius_x * np.cos(theta)
    y = center_y + radius_y * np.sin(theta)

    init = np.column_stack((y, x))

    snake = active_contour(smooth_eye,init,alpha=0.05,beta=0.01,gamma=0.5,w_line=0.1,w_edge=1,max_num_iter=1500,convergence=0.001)

    # Crear una máscara inicialmente negra
    mask_snake = np.zeros(gray_eye.shape, dtype=np.uint8)

    # Coordenadas del contorno: filas (y) y columnas (x)
    rr, cc = polygon(
        snake[:, 0],
        snake[:, 1],
        shape=gray_eye.shape
    )

    # Rellenar el interior del contorno
    mask_snake[rr, cc] = 1

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gray_eye, cmap="gray")
    plt.plot(init[:, 1],init[:, 0],"--",linewidth=2,label="Contorno inicial")
    plt.plot(snake[:, 1],snake[:, 0],linewidth=2,label="Snake final")
    plt.title("Segmentación de la pupila mediante Snakes")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_snake, cmap="gray", vmin=0, vmax=1)
    plt.title("Máscara binaria de la pupila mediante Snakes")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Antes de aplicar Watershed, primero calculamos el gradiente de la imagen y definimos marcadores de regiones. 
    # 
    # Decidimos aumentar el contraste de la imagen porque no se observaba bien el cambio entre el iris y la pupila ya a simple vista. 

    # %%
    roi_contraste = np.clip(
        roi_gray.astype(np.float32) * 1.5,
        0,
        255
    ).astype(np.uint8)

    roi_suavizada = cv2.GaussianBlur(
        roi_contraste,
        (5, 5),
        0
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_gray, cmap="gray")
    plt.title("ROI original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(roi_suavizada, cmap="gray")
    plt.title("Contraste aumentado + suavizado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # %%
    # Gradiente horizontal y vertical
    grad_x = cv2.Sobel(
        roi_suavizada,
        cv2.CV_64F,
        1, 0,
        ksize=7
    )

    grad_y = cv2.Sobel(
        roi_suavizada,
        cv2.CV_64F,
        0, 1,
        ksize=7
    )

    # Magnitud del gradiente
    gradiente = cv2.magnitude(
        grad_x.astype(np.float32),
        grad_y.astype(np.float32)
    )

    # Normalización para visualizar
    gradiente_norm = cv2.normalize(
        gradiente,
        None,
        0, 255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_gray, cmap="gray")
    plt.title("ROI original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gradiente_norm, cmap="gray")
    plt.title("Magnitud del gradiente")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # %%
    markers = np.zeros(roi_gray.shape, dtype=np.int32)

    # Región pequeña dentro de la pupila
    cv2.circle(
        markers,
        (int(center_x), int(center_y)),
        20,
        1,
        -1
    )

    # Todo lo que esté fuera de esta elipse
    # se considera background seguro.
    roi_mask = np.zeros(roi_gray.shape, dtype=np.uint8)

    cv2.ellipse(
        roi_mask,
        (int(center_x), int(center_y)),
        (60, 55),
        0,
        0,
        360,
        1,
        -1
    )

    # Background seguro = fuera de la elipse
    markers[roi_mask == 0] = 2

    plt.figure(figsize=(10, 7))

    plt.imshow(roi_gray, cmap="gray")

    plt.contour(
        markers == 1,
        colors="r",
        linewidths=2
    )

    plt.contour(
        markers == 2,
        colors="b",
        linewidths=2
    )

    plt.title("Marcadores para Watershed")
    plt.axis("off")
    plt.show()

    # %%
    roi_bgr = cv2.cvtColor(roi_suavizada,cv2.COLOR_GRAY2BGR)

    markers_ws = cv2.watershed(roi_bgr,markers.copy())

    mask_watershed = (markers_ws == 1).astype(np.uint8) * 255

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(roi_gray, cmap="gray")
    plt.title("ROI original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_watershed, cmap="gray")
    plt.title("Máscara Watershed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(roi_gray, cmap="gray")
    plt.contour(
        mask_watershed > 0,
        colors="r",
        linewidths=2
    )
    plt.title("Pupila segmentada")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Con el método de Watershed, se obtiene un contorno bastante cercano al de la pupila pero irregular. 

    # %% [markdown]
    # Ahora aplicamos erosión, dilatación, apertura y cierre sobre la máscara binaria obtenida por watershed, puesto que es la que mejor segmenta la pupila. Buscamos visualizar la eliminación de destellos luminosos (reflejos corneales del flash), oclusión por pestañas o sombras que pide la consigna si utilizamos dicha máscara para aplicar las operaciones.
    # La eliminación de dichos artefactos se observa mejor mediante la máscara de Otsu pero se busca continuar con una segmentación adecuada de la pupila. 
    # Las propiedades morfológicas se utilizan para remover imperfecciones de la imagen segmentada para luego obtener información de la estructura y forma de los objetos.

    # %% [markdown]
    # El kernel determina el tamaño del kernel del elemento estructurante que se utiliza en cada operación. Un kernel demasiado grande puede borrar partes de la pupila o alterar su forma, mientras que uno muy pequeño puede no producir cambios útiles. Es por esto que elegimos un kernel mediano (5x5)

    # %%
    # Usamos directamente la máscara que ya tenemos de Otsu, sin operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))

    # Aplicar operaciones morfológicas
    mask_erosion = cv2.erode(mask_watershed, kernel, iterations=1)
    mask_dilatacion = cv2.dilate(mask_watershed, kernel, iterations=1)
    mask_apertura = cv2.morphologyEx(mask_watershed, cv2.MORPH_OPEN, kernel)
    mask_cierre = cv2.morphologyEx(mask_watershed, cv2.MORPH_CLOSE, kernel)

    resultados = {
        "Watershed original": mask_watershed,
        "Erosión": mask_erosion,
        "Dilatación": mask_dilatacion,
        "Apertura": mask_apertura,
        "Cierre": mask_cierre}

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(roi_gray, cmap="gray")
    plt.contour(
        mask_watershed > 0,
        colors="r",
        linewidths=2
    )
    plt.title("Pupila segmentada")
    plt.axis("off")

    for i, (nombre, mascara) in enumerate(resultados.items()):
        plt.subplot(2, 3, i + 2)
        plt.imshow(mascara, cmap="gray")
        plt.title(nombre)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Luego, genero la esqueletización, donde me quedo con la estructura mínima que conserva la forma del objeto original. 
    # Pretende obtener de la imagen un patrón continuo que contenga la menor cantidad de datos posibles, pero que siga aún, conteniendo un rastro del objeto original.
    # Genero el esqueleto a partir del algoritmo de watershed. Este es mucho más preciso para la segmentación de la pupila.

    from skimage.morphology import skeletonize

    # Extraemos los contornos externos de la región pupilar
    contornos, _ = cv2.findContours(mask_watershed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Dibujamos el contorno sobre la ROI original
    imagen_contorno = cv2.cvtColor(roi_gray,cv2.COLOR_GRAY2RGB)

    # Creamos un borde fino a partir de la máscara de Watershed
    kernel_borde = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    borde_pupila = cv2.subtract(mask_watershed,cv2.erode(mask_watershed, kernel_borde))

    # Esqueletizamos el borde extraído
    esqueleto = skeletonize(borde_pupila > 0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(mask_watershed, cmap="gray")
    plt.title("Región pupilar")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(borde_pupila, cmap="gray")
    plt.title("Borde extraído")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(esqueleto, cmap="gray")
    plt.title("Esqueleto del contorno")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


    # %% [markdown]
    # Ahora genero combinaciones de las operaciones morfológicas

    # %%
    # Genero operaciones secuenciales
    apertura_cierre = cv2.morphologyEx(cv2.morphologyEx(mask_watershed * 255, cv2.MORPH_OPEN, kernel),
                                       cv2.MORPH_CLOSE, kernel) > 0
    cierre_apertura = cv2.morphologyEx(cv2.morphologyEx(mask_watershed * 255, cv2.MORPH_CLOSE, kernel),
                                       cv2.MORPH_OPEN, kernel) > 0
    erosion_dilatacion = cv2.dilate(cv2.erode(mask_watershed * 255, kernel), kernel) > 0
    dilatacion_erosion = cv2.erode(cv2.dilate(mask_watershed * 255, kernel), kernel) > 0

    # Combino máscaras ya calculadas
    interseccion_ap_cie = np.logical_and(mask_apertura > 0, mask_cierre > 0)
    union_ap_cie = np.logical_or(mask_apertura > 0, mask_cierre > 0)

    mascaras = {
        "Watershed original": mask_watershed,
        "Erosión": mask_erosion,
        "Dilatación": mask_dilatacion,
        "Apertura": mask_apertura,
        "Cierre": mask_cierre,
        "Apertura + cierre":apertura_cierre,
        "Cierre + apertura": cierre_apertura,
        "Erosión + dilatación": erosion_dilatacion,
        "Dilatación + erosión": dilatacion_erosion,
        "Intersección apertura/cierre": interseccion_ap_cie,
        "Unión apertura/cierre": union_ap_cie
    }

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    axes = axes.ravel()

    for ax, (nombre, mascara) in zip(axes, mascaras.items()):
        ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
        ax.set_title(nombre, fontsize=10)
        ax.axis("off")

    # Ocultar paneles sobrantes
    for ax in axes[len(mascaras):]:
        ax.axis("off")

    plt.suptitle(
        "Comparación de combinaciones morfológicas sobre Watershed",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Por último, comparamos la combinación de máscaras resultantes basandonos en la segmentación watershed, considerando el área segmentada, el cambio porcentual de área y las métricas Dice e IoU calculadas respecto de la máscara original. Estas últimas se utilizaron para cuantificar las modificaciones introducidas por cada operación, y no como una medida directa de precisión respecto de la pupila real.

    # %%
    import pandas as pd

    def calcular_metricas(mascara, original):
        pred = np.asarray(mascara) > 0
        base = np.asarray(original) > 0

        interseccion = np.logical_and(pred, base).sum()
        union = np.logical_or(pred, base).sum()

        area = pred.sum()
        area_original = base.sum()

        # Solapamiento con la máscara original (no con la pupila real)
        iou_original = (interseccion / union if union > 0 else 1.0)

        dice_original = (2 * interseccion / (area + area_original)
            if (area + area_original) > 0 else 1.0 )

        cambio_area_pct = (100 * (area - area_original) / area_original
            if area_original > 0 else 0.0)

        pixeles_modificados_pct = (100 * np.logical_xor(pred, base).sum() / base.size)

        return {
            "Área (píxeles)": area,
            "Cambio de área (%)": cambio_area_pct,
            "Dice vs. Watershed": dice_original,
            "IoU vs. Watershed": iou_original,
            "Píxeles modificados (%)": pixeles_modificados_pct
        }
    resultados = []

    for nombre, mascara in mascaras.items():
        metricas = calcular_metricas(mascara, mask_watershed)
        metricas["Método"] = nombre
        resultados.append(metricas)

    tabla = pd.DataFrame(resultados).set_index("Método")

    display(tabla.round(4))


    # La tabla muestra que la erosión fue la que más área de la pupila quitó mientras que la dilatación la que más agregó. 
    # Estos cambios evidencian que ambas operaciones modifican de manera más significativa el tamaño de la región, por lo que su aplicación debe evaluarse cuidadosamente para evitar la pérdida de partes de la pupila o la incorporación de regiones ajenas a ella.
    # La operación de apertura modificó una proporción muy pequeña de la segmentación mientras que el cierre, en cambio, no produjo cambios medibles en la máscara para el elemento estructurante utilizado.
    # Las combinaciones evaluadas tampoco generaron mejoras cuantitativas apreciables respecto de la máscara original, pero podríamos considerar que una de ellas es la más adecuada ya que mantiene significatimente la morfología de la pupila, reduciendo solo donde es necesario y cuya comparación con watershed (Dice o IoU) está más cercano al valor unitario. 
    # Siguiendo este criterio podríamos decir que el cierre, una combinación compuesta por dilatación seguida de erosión o la unión entre cierre y apertura son las combinaciones más óptimas.

def modulo_3():
    # %% [markdown]
    # Cargamos la ROI generada en el Módulo 1 (en gris y en color). El video se usará posteriormente.

    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from skimage.feature import graycomatrix, graycoprops

    # VIDEO_PATH, ROI_PATH, ROI_RGB_PATH y ROI se definen al inicio y en preparar_roi().
    print("ROI gris:", roi_gray.shape)
    print("ROI RGB:", roi_rgb.shape)

    print("ROI gris:", roi_gray.shape)
    print("ROI RGB:", roi_rgb.shape)

    # %% [markdown]
    # Se utilizará la ROI cuantizada a 32 niveles (5 bits por canal). Esto será útil para que la GLCM sea mucho más chica y que sus estadísticas sean más estables con pocos píxeles.

    # %%
    NIVELES = 32                    # Puedo cambiar niveles para ver que pasa después si lo paso a 64
    paso = 256 // NIVELES          

    # Cuantización
    roi_cuant = (roi_rgb // paso).astype(np.uint8)
    gray_cuant = (roi_gray // paso).astype(np.uint8)

    # %% [markdown]
    # Segmentación de la pupila
    # Usamos la segmentación por Watershed del módulo 2, ya que fue la que mejor delimitó la pupila. Además, sumamos un cierre morfológico, que según la última tabla del mismo módulo fue de las operaciones que menos modificó la máscara.
    # 
    # A partir de la máscara extraemos el contorno externo de la pupila. Hacemos una función con la segmentación para poder reutilizarla en el video frame a frame.

    # %%
    center_x, center_y = PUPIL_CENTER

    def segmentar_pupila(gris, cx=center_x, cy=center_y, radio=20):
        """Segmenta la pupila con Watershed. Devuelve la máscara y su contorno externo"""
        # Hago el contraste + suavizado, tal como se hizo en el módulo 2
        contraste = np.clip(gris.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        suavizada = cv2.GaussianBlur(contraste, (5, 5), 0)

        # Marcadores: 1 = pupila segura, 2 = fondo seguro (fuera de la elipse)
        marcadores = np.zeros(gris.shape, dtype=np.int32)
        cv2.circle(marcadores, (int(cx), int(cy)), radio, 1, -1)
        zona = np.zeros(gris.shape, dtype=np.uint8)
        cv2.ellipse(zona, (int(cx), int(cy)), (60, 55), 0, 0, 360, 1, -1)
        marcadores[zona == 0] = 2

        ws = cv2.watershed(cv2.cvtColor(suavizada, cv2.COLOR_GRAY2BGR), marcadores)
        mascara = (ws == 1).astype(np.uint8) * 255

        # Cierre morfológico con kernel elíptico de 5x5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

        # Contorno externo de la máscara
        contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contorno = max(contornos, key=cv2.contourArea) if contornos else None
        return mascara, contorno

    mask_pupila, contorno = segmentar_pupila(roi_gray)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_rgb)
    plt.title("ROI original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(roi_rgb)
    plt.contour(mask_pupila > 0, colors="r", linewidths=2)
    plt.title("Pupila segmentada")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # A continuación se calculará el área, que representa la cantidad de píxeles de la máscara, el perímetro, que es la longitud del borde de la región, las coordenadas del centroide, la circularidad y la excentricidad.

    # %%
    def descriptores_morfometricos(contorno):
        """Área, perímetro, centroide, circularidad y excentricidad de una máscara binaria."""
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        # Centroide a partir de los momentos del contorno
        m = cv2.moments(contorno)
        xc = m["m10"] / m["m00"]
        yc = m["m01"] / m["m00"]

        # Elipse que mejor se ajusta al contorno: de acá salen los ejes y la excentricidad
        elipse = cv2.fitEllipse(contorno)
        (_, _), (d1, d2), angulo = elipse
        eje_mayor, eje_menor = max(d1, d2), min(d1, d2)

        circularidad = 4 * np.pi * area / perimetro**2
        excentricidad = np.sqrt(1 - (eje_menor / eje_mayor) ** 2)

        return {
            "Área (px²)": area,
            "Perímetro (px)": perimetro,
            "Centroide X_c (px)": xc,
            "Centroide Y_c (px)": yc,
            "Circularidad": circularidad,
            "Excentricidad": excentricidad,
            "Eje mayor (px)": eje_mayor,
            "Eje menor (px)": eje_menor,
        }, elipse


    desc, elipse = descriptores_morfometricos(contorno)

    tabla_morfo = pd.DataFrame(desc, index=["Pupila (frame de referencia)"]).T
    display(tabla_morfo.round(4))

    # %%
    # Visualización: contorno, centroide y elipse ajustada
    img = roi_rgb.copy()
    cv2.drawContours(img, [contorno], -1, (255, 0, 0), 2)
    cv2.ellipse(img, elipse, (255, 255, 0), 1)
    cv2.circle(img, (int(round(desc["Centroide X_c (px)"])), int(round(desc["Centroide Y_c (px)"]))), 4, (0, 255, 255), -1)

    plt.figure(figsize=(9, 5))
    plt.imshow(img)
    plt.title(f"Circularidad = {desc['Circularidad']:.3f} | Excentricidad = {desc['Excentricidad']:.3f}")
    plt.axis("off")
    plt.show()

    # %% [markdown]
    # Observando los resultados, se observa que el centroide de la pupila segmentada está en (157.2, 51.1), mientras que el centro asignado para aplicar Watershed se colocó en (155, 50), por lo que hay solo una leve diferencia entre estos dos centros.
    # 
    # La circularidad obtenida no es tan cercana a 1 por los bordes irregulares de la segmentación, que igualmente es adecuada para el análisis ya que, como se observa, rodea correctamente la pupila.
    # 
    # Más alla de que la circularidad no fue tan cercana a 1, el tamaño de los ejes de la elipse ajustada indican que la pupila sigue una forma aproximadamente circular.

    # %% [markdown]
    # A continuación, se calculará la media, desvío estándar, entropía y asimetría sobre la ROI de la pupila y el iris en los 3 canales de colores (RGB) de forma independiente.
    # 
    # Para analizar la textura del iris lo segmentamos también por Watershed, con tres marcadores: iris seguro (un anillo cercano a la pupila), pupila segura y fondo seguro (todo lo que queda fuera de una elipse grande). Esta segmentación se usa solo para el análisis de textura del frame de referencia, por lo que no se aplican correcciones adicionales, y se verifica visualmente que sea aproximadamente correcta.

    # %%
    # Parámetros de los marcadores
    ANILLO_INT, ANILLO_EXT = 6, 12   # iris "seguro": anillo entre estas dos distancias (px) al borde de la pupila
    FONDO_RX, FONDO_RY = 78, 70      # semiejes (px) de la elipse: fuera de ella es fondo seguro
    Y_PARPADO = 10                   # filas de arriba de la ROI que se consideran fondo (pestañas / párpado)

    cx, cy = int(round(desc["Centroide X_c (px)"])), int(round(desc["Centroide Y_c (px)"]))

    def kernel_elipse(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    marcadores_iris = np.zeros(roi_gray.shape, dtype=np.int32)

    # 2 = fondo seguro: fuera de la elipse y en la franja superior
    zona = np.zeros(roi_gray.shape, dtype=np.uint8)
    cv2.ellipse(zona, (cx, cy), (FONDO_RX, FONDO_RY), 0, 0, 360, 1, -1)
    marcadores_iris[zona == 0] = 2
    marcadores_iris[:Y_PARPADO, :] = 2

    # 1 = iris seguro: anillo cercano a la pupila
    anillo_semilla = cv2.subtract(cv2.dilate(mask_pupila, kernel_elipse(ANILLO_EXT)),
                                  cv2.dilate(mask_pupila, kernel_elipse(ANILLO_INT)))
    marcadores_iris[anillo_semilla > 0] = 1

    # 3 = pupila segura (para que el iris no la invada)
    marcadores_iris[cv2.erode(mask_pupila, kernel_elipse(2)) > 0] = 3

    plt.figure(figsize=(10, 5))
    plt.imshow(roi_rgb)
    plt.contour(marcadores_iris == 1, colors="c", linewidths=1.5)
    plt.contour(marcadores_iris == 2, colors="b", linewidths=1.5)
    plt.contour(marcadores_iris == 3, colors="r", linewidths=1.5)
    plt.title("Marcadores para Watershed del iris")
    plt.axis("off")
    plt.show()

    # %%
    # Watershed sobre la ROI suavizada
    roi_suavizada_iris = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    ws_iris = cv2.watershed(cv2.cvtColor(roi_suavizada_iris, cv2.COLOR_GRAY2BGR), marcadores_iris.copy())
    mask_iris = (ws_iris == 1).astype(np.uint8) * 255

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(mask_iris, cmap="gray")
    plt.title("Máscara del iris (Watershed)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(roi_rgb)
    plt.contour(mask_pupila > 0, colors="r", linewidths=1.5)
    plt.contour(mask_iris > 0, colors="c", linewidths=1.5)
    plt.title("Pupila (rojo) e iris (cian)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Píxeles de la pupila:", cv2.countNonZero(mask_pupila))
    print("Píxeles del iris:    ", cv2.countNonZero(mask_iris))

    # %% [markdown]
    # Para evitar errores en los cálculos estadísticos, se deben excluir los reflejos brillantes que se observan en el iris.

    # %%
    UMBRAL_BRILLO = 205   # nivel de gris a partir del cual un píxel se considera reflejo

    # Para elegir el umbral: cómo son los valores del iris
    valores_iris = roi_gray[mask_iris > 0]
    print("Iris: mediana =", int(np.median(valores_iris)), "| percentil 99 =", int(np.percentile(valores_iris, 99)), "| máximo =", int(valores_iris.max()))

    # Reflejos: píxeles muy claros, con un pequeño margen alrededor
    _, mask_brillo = cv2.threshold(roi_gray, UMBRAL_BRILLO, 255, cv2.THRESH_BINARY)
    mask_brillo = cv2.dilate(mask_brillo, np.ones((5, 5), np.uint8))

    # Iris sin reflejos
    mask_iris_sin_brillo = cv2.bitwise_and(mask_iris, cv2.bitwise_not(mask_brillo))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_rgb)
    plt.contour(mask_iris > 0, colors="c", linewidths=1.5)
    plt.contour(mask_brillo > 0, colors="m", linewidths=1.5)
    plt.title("Iris (cian) y reflejos excluidos (magenta)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_iris_sin_brillo, cmap="gray")
    plt.title("Máscara del iris sin reflejos")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Píxeles del iris antes:  ", cv2.countNonZero(mask_iris))
    print("Píxeles del iris después:", cv2.countNonZero(mask_iris_sin_brillo))

    # %% [markdown]
    # La entropía mide la proporción de píxeles de la región en cada nivel de gris, mientras que la asimetría es positiva si la distribución tiene una cola hacia valores altos y negativa si la tiene hacia valores bajos.

    # %%
    def estadisticas_primer_orden(img_cuant, mascara, niveles=NIVELES):
        """Media, desvío, entropía y asimetría de cada canal (R, G, B), solo dentro de la máscara."""
        medias, desvios = cv2.meanStdDev(img_cuant, mask=mascara)

        filas = {}
        for i, canal in enumerate("RGB"):
            # Entropía (en bits) a partir del histograma de los niveles cuantizados
            hist = cv2.calcHist([img_cuant], [i], mascara, [niveles], [0, niveles]).ravel()
            p = hist / hist.sum()
            p = p[p > 0]
            entropia = -np.sum(p * np.log2(p))

            # Asimetría (skewness) de los valores de los píxeles dentro de la máscara
            valores = img_cuant[:, :, i][mascara > 0]
            asimetria = pd.Series(valores).skew()

            filas[canal] = {
                "Media": medias[i, 0],
                "Desvío": desvios[i, 0],
                "Entropía (bits)": entropia,
                "Asimetría": asimetria,
            }
        return pd.DataFrame(filas).T


    stats_pupila = estadisticas_primer_orden(roi_cuant, mask_pupila)
    stats_iris = estadisticas_primer_orden(roi_cuant, mask_iris_sin_brillo)

    print("PUPILA")
    display(stats_pupila.round(3))
    print("IRIS")
    display(stats_iris.round(3))

    # %% [markdown]
    # La pupila es más oscura y uniforme que el iris: sus medias están entre 2.5 y 3.2, contra cerca de 10 en el iris, y tiene menor desvío y entropía, lo cual es lo esperable, ya que el estroma iridiano presenta más variación de intensidad que la pupila.
    # 
    # La asimetría de la pupila es positiva y alta (2.3 a 2.7), probablemente por el reflejo del flash, que no se excluyó en esa región. En el iris es casi nula, por lo que la distribución de intensidades es aproximadamente simétrica.
    # 
    # Entre canales las diferencias son pequeñas. En la pupila el canal B es algo más claro, y en el iris los tres canales tienen medias, desvíos y entropías muy parecidos.

    # %% [markdown]
    # La GLCM cuenta con qué frecuencia aparece un nivel de gris `i` junto a un nivel `j` a una distancia y dirección dadas. Así describe la relación espacial entre píxeles. Los atributos se calculan por canal y se promedian sobre los 4 ángulos.

    # %%
    DISTANCIAS = [1]
    ANGULOS = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def glcm_mascara(canal, mascara, niveles=NIVELES):
        img = canal.copy()
        img[mascara == 0] = niveles                      # nivel extra = "fuera de la región"
        glcm = graycomatrix(img, distances=DISTANCIAS, angles=ANGULOS,
                            levels=niveles + 1, symmetric=True, normed=False)
        glcm = glcm[:niveles, :niveles].astype(float)    # se descarta el nivel de fondo
        glcm /= glcm.sum(axis=(0, 1), keepdims=True)     # se renormaliza
        return glcm

    def atributos_glcm(glcm):
        props = {"Contraste": "contrast", "Homogeneidad": "homogeneity",
                 "Energía": "energy", "Correlación": "correlation"}
        return {nombre: graycoprops(glcm, p).mean() for nombre, p in props.items()}

    # %%
    glcms = {}
    filas_pupila, filas_iris = {}, {}

    for idx, canal in enumerate("RGB"):
        img_canal = roi_cuant[:, :, idx]
        glcms[("Pupila", canal)] = glcm_mascara(img_canal, mask_pupila)
        glcms[("Iris", canal)] = glcm_mascara(img_canal, mask_iris_sin_brillo)
        filas_pupila[canal] = atributos_glcm(glcms[("Pupila", canal)])
        filas_iris[canal] = atributos_glcm(glcms[("Iris", canal)])

    glcm_pupila = pd.DataFrame(filas_pupila).T
    glcm_iris = pd.DataFrame(filas_iris).T

    print("GLCM – Pupila")
    display(glcm_pupila.round(4))
    print("GLCM – Iris")
    display(glcm_iris.round(4))

    # %%
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, canal in zip(axs, "RGB"):
        im = ax.imshow(glcms[("Iris", canal)][:, :, 0, 0], cmap="Reds")
        ax.set_title(f"GLCM iris – canal {canal} (0°)")
        ax.set_xlabel("Nivel j"); ax.set_ylabel("Nivel i")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # En el iris, los tres canales dan valores de GLCM muy parecidos. Las GLCM de la figura se concentran sobre la diagonal, con niveles de gris vecinos, lo que indica una textura suave, con variaciones graduales entre píxeles. Esto coincide con la correlación y homogeneidad alta.
    # 
    # Entre canales, R muestra mayor contraste (0.563) y menor homogeneidad (0.806). G tiene la energía más baja (0.2087) y la correlación más alta (0.974), y B tiene la mayor energía (0.213), es decir, es el más uniforme. Las diferencias son pcas, por lo que ningún canal aporta una riqueza textural claramente mayor. Sin embargo, podemos decir que R es el más informativo, debido al mayor contraste y menor homogeneidad.
    # 
    # La pupila es más homogénea y tiene mayor energía que el iris, como se espera de una región más uniforme y oscura. Su contraste es similar al del iris, probablemente por los bordes y el brillo que quedaron dentro de la máscara.

    # %% [markdown]
    # A continuación aplicaremos a cada frame del video el mismo procedimiento que se hizo para segmentar la pupila con un solo frame. Esto implica recorte de la ROI, conversión a gris y segmentación de la pupila por Watershed con cierre morfológico.
    # 
    # De cada contorno se obtienen el área y el diámetro.

    # %% [markdown]
    # Para segmentar la pupila en cada frame del video se parte del método del módulo 2 (Watershed sobre la ROI con contraste ×1.5 y suavizado gaussiano de 5x5, marcadores de pupila y de fondo, y cierre morfológico), pero no alcanza con aplicarlo tal cual. En el módulo 2 se trabajó con un único frame de referencia y una ROI fija, mientras que en el video la pupila se mueve, cambia de tamaño y aparecen reflejos del flash y pestañas. Por eso se hicieron algunas modificaciones.
    # 
    # - En el video, el centro de la pupila se estima frame a frame, por lo que se usa una ventana de búsqueda alrededor de la posición del frame anterior y la ROI de 320x130 px se recorta centrada en ese punto.
    # - Los píxeles muy brillantes cercanos a la pupila se detectan, se dilatan y se rellenan con los píxeles de alrededor.
    # - El marcador de pupila segura pasa de un círculo de radio 20 a uno de radio 10 como valor base, para que la semilla quede dentro de la pupila aun cuando se contrae.
    # - Con ese radio, en algunos frames (por ejemplo cuando se ve el reflejo en la pupila), el Watershed no atraviesa la franja clara y el contorno queda más chico que la pupila. Por eso, además del radio base, se prueban radios de 16 y 22 px y se compara el contraste de borde de cada contorno. Un contorno correcto tiene pupila (oscuro) adentro e iris (claro) afuera, por lo que su contraste es alto.
    # - Se mantiene el cierre de 5x5 dentro de la segmentación y después se agrega una apertura de 11x11 (corta colitas finas producidas por reflejos sobre el borde o pestañas), un cierre de 21x21 (rellena huecos) y la envolvente convexa del contorno.

    # %%
    X0, Y0, ROI_W, ROI_H = ROI_RECT     # rectángulo seleccionado en el menú

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps:.2f} | Frames: {n_frames} | Duración: {n_frames / fps:.2f} s")

    # %%
    RADIO_BUSQUEDA = 80          # ventana de búsqueda alrededor de la pupila del frame anterior (px)
    RADIO_BUSQUEDA_INICIAL = 150 # para el primer frame
    MARGEN_OSCURO = 15
    KERNEL_SUAVIZADO = 31        # el suavizado borra pestañas y reflejos
    KERNEL_REFLEJO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    def seguir_pupila(frame_bgr, centro_previo, radio_busqueda=RADIO_BUSQUEDA):
        h, w = frame_bgr.shape[:2]
        cx, cy = centro_previo
        x0, x1 = max(cx - radio_busqueda, 0), min(cx + radio_busqueda, w)
        y0, y1 = max(cy - radio_busqueda, 0), min(cy + radio_busqueda, h)

        ventana = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        ventana = cv2.morphologyEx(ventana, cv2.MORPH_OPEN, KERNEL_REFLEJO)   # borra reflejos brillantes
        ventana = cv2.GaussianBlur(ventana, (KERNEL_SUAVIZADO, KERNEL_SUAVIZADO), 0)
        valor_min, _, punto_min, _ = cv2.minMaxLoc(ventana)

        # Región oscura conectada al mínimo y su centroide
        _, binaria = cv2.threshold(ventana, int(valor_min) + MARGEN_OSCURO, 255, cv2.THRESH_BINARY_INV)
        _, etiquetas = cv2.connectedComponents(binaria)
        region = (etiquetas == etiquetas[punto_min[1], punto_min[0]]).astype(np.uint8)
        M = cv2.moments(region, binaryImage=True)
        return x0 + int(M["m10"] / M["m00"]), y0 + int(M["m01"] / M["m00"])

    centro = (X0 + center_x, Y0 + center_y)
    centros = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            centros.append((np.nan, np.nan)); continue
        radio = RADIO_BUSQUEDA_INICIAL if i == 0 else RADIO_BUSQUEDA
        centro = seguir_pupila(frame, centro, radio)
        centros.append(centro)

    centros = np.array(centros, dtype=float)

    # %%
    def recortar_roi(frame_bgr, centro):
        """Recorta una ROI de 320x130 con la pupila en (center_x, center_y), como en el módulo 1."""
        h, w = frame_bgr.shape[:2]
        x0 = int(np.clip(centro[0] - center_x, 0, w - ROI_W))
        y0 = int(np.clip(centro[1] - center_y, 0, h - ROI_H))
        roi = frame_bgr[y0:y0 + ROI_H, x0:x0 + ROI_W]
        return roi, (centro[0] - x0, centro[1] - y0)      # centro dentro de la ROI

    # %%
    UMBRAL_REFLEJO = 200
    RADIO_ZONA_REFLEJO = 45     # los reflejos se buscan solo cerca de la pupila
    KERNEL_DILATAR_REFLEJO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def quitar_reflejos(gris, cx, cy):
        """Rellena los reflejos brillantes cercanos a la pupila con los píxeles de alrededor."""
        _, brillo = cv2.threshold(gris, UMBRAL_REFLEJO, 255, cv2.THRESH_BINARY)
        brillo = cv2.dilate(brillo, KERNEL_DILATAR_REFLEJO)
        cerca = np.zeros(gris.shape, np.uint8)
        cv2.circle(cerca, (int(cx), int(cy)), RADIO_ZONA_REFLEJO, 255, -1)
        brillo = cv2.bitwise_and(brillo, cerca)
        return cv2.inpaint(gris, brillo, 5, cv2.INPAINT_TELEA)

    # %%
    RADIOS_MARCADOR = (10, 16, 22)
    MARGEN_CONTRASTE = 8
    KERNEL_COLITAS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    KERNEL_CIERRE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    def contorno_con_marcador(limpia, cx, cy, radio):
        """Watershed + apertura + cierre + envolvente convexa con un radio de marcador dado."""
        try:
            mascara, _ = segmentar_pupila(limpia, cx=cx, cy=cy, radio=radio)
        except ValueError:
            return None
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL_COLITAS)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL_CIERRE)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contornos:
            return None
        return cv2.convexHull(max(contornos, key=cv2.contourArea))

    def contraste_borde(gris, contorno, ancho=4):
        """Gris medio del anillo exterior menos el del anillo interior del contorno."""
        mascara = np.zeros(gris.shape, np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ancho + 1, 2 * ancho + 1))
        anillo_ext = cv2.subtract(cv2.dilate(mascara, k), mascara)
        anillo_int = cv2.subtract(mascara, cv2.erode(mascara, k))
        return cv2.mean(gris, mask=anillo_ext)[0] - cv2.mean(gris, mask=anillo_int)[0]

    def medir_pupila(roi_gris, cx=center_x, cy=center_y):
        """Devuelve el contorno de la pupila o None si no segmenta nada."""
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
            if c > mejor_c + MARGEN_CONTRASTE:       # solo reemplaza si mejora claramente
                mejor, mejor_c = cand, c
        return mejor

    # %%
    def circularidad(contorno):
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        return 4 * np.pi * area / perimetro**2

    # %% [markdown]
    # 
    # Como verificación de que la segmentación generaliza a todo el video, se evalúa en 12 frames.

    # %%
    indices = np.linspace(0, n_frames - 1, 12).astype(int)

    fig, axs = plt.subplots(3, 4, figsize=(18, 8))
    for ax, idx in zip(axs.ravel(), indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        centro_i = (int(centros[idx, 0]), int(centros[idx, 1]))
        roi, (cx, cy) = recortar_roi(frame, centro_i)
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        contorno = medir_pupila(gris, cx=cx, cy=cy)
        if contorno is None:
            titulo = f"frame {idx}: sin segmentación"
        else:
            cv2.drawContours(vis, [contorno], -1, (0, 255, 0), 1)
            titulo = f"frame {idx} ({idx / fps:.1f} s)\nC = {circularidad(contorno):.2f}"
        cv2.circle(vis, (cx, cy), 3, (255, 0, 0), -1)
        ax.imshow(vis); ax.set_title(titulo, fontsize=10); ax.axis("off")
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Se observa que la segmentación planteada se puede generalizar correctamente para los frames del video, con algunas excepciones, como por ejemplo el frame 955, donde posiblemente se encuentren errores de segmentación por parpadeo u otros artefactos.
    # 
    # Para poder estimar si se realizó correctamente la segmentación de un frame, se utiliza el parámetro de circularidad. A continuación se muestra un gráfico de la evolución de este parámetro a lo largo del video.

    # %%
    filas = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or np.isnan(centros[i, 0]):
            filas.append((i, i / fps, np.nan)); continue
        roi, (cx, cy) = recortar_roi(frame, (int(centros[i, 0]), int(centros[i, 1])))
        contorno = medir_pupila(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cx=cx, cy=cy)
        filas.append((i, i / fps, np.nan if contorno is None else circularidad(contorno)))

    df = pd.DataFrame(filas, columns=["frame", "t (s)", "Circularidad"])

    plt.figure(figsize=(12, 3))
    plt.plot(df["t (s)"], df["Circularidad"], linewidth=0.8)
    plt.xlabel("Tiempo (s)"); plt.ylabel("Circularidad")
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # Se observa que la circularidad del contorno tiene leves variaciones a lo largo del video. Los picos y variaciones muy grandes seguramente se deben a parpadeos o movimientos (artefactos). Para el análisis estadístico de algunos parámetros conviene excluir esos frames o interpolar los valores. Para esto, se considerarán frames invalidos a aquellos con un circularidad menor a 0.89, ya que se considera que en esos casos probablemente no se segmentó bien la pupila por algun artefacto.

    # %%
    UMBRAL_CIRC = 0.89

    df["Válido"] = df["Circularidad"] >= UMBRAL_CIRC     # NaN queda como inválido
    print(f"Frames inválidos: {(~df['Válido']).sum()} de {len(df)} "
          f"({100 * (~df['Válido']).mean():.1f} %)")

    plt.figure(figsize=(12, 3))
    plt.plot(df["t (s)"], df["Circularidad"], linewidth=0.8)
    plt.scatter(df.loc[~df["Válido"], "t (s)"], df.loc[~df["Válido"], "Circularidad"],
                color="r", s=12, zorder=3, label="Frames inválidos")
    plt.axhline(UMBRAL_CIRC, color="k", linestyle="--", linewidth=0.8, label=f"Umbral = {UMBRAL_CIRC}")
    plt.xlabel("Tiempo (s)"); plt.ylabel("Circularidad"); plt.legend()
    plt.tight_layout(); plt.show()

    # %% [markdown]
    # A continuación se calculará el área y el diametro frame a frame.

    # %%
    UMBRAL_CIRC = 0.89

    filas = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or np.isnan(centros[i, 0]):
            filas.append((i, i / fps, np.nan, np.nan, np.nan)); continue
        roi, (cx, cy) = recortar_roi(frame, (int(centros[i, 0]), int(centros[i, 1])))
        contorno = medir_pupila(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cx=cx, cy=cy)
        if contorno is None:
            filas.append((i, i / fps, np.nan, np.nan, np.nan)); continue
        area = cv2.contourArea(contorno)
        filas.append((i, i / fps, area, 2 * np.sqrt(area / np.pi), circularidad(contorno)))

    df = pd.DataFrame(filas, columns=["frame", "t (s)", "Área auto (px²)", "Diámetro auto (px)", "Circularidad"])
    df["Válido"] = df["Circularidad"] >= UMBRAL_CIRC            # NaN queda como inválido

    df["Área (px²)"] = df["Área auto (px²)"].where(df["Válido"]).interpolate(method="linear", limit_direction="both")
    df["Diámetro (px)"] = df["Diámetro auto (px)"].where(df["Válido"]).interpolate(method="linear", limit_direction="both")
    df["Interpolado"] = ~df["Válido"]

    inv = ~df["Válido"]
    rachas = inv.ne(inv.shift()).cumsum()[inv].value_counts()
    print(f"Frames interpolados: {inv.sum()} de {len(df)} ({100 * inv.mean():.1f} %) | racha más larga: {rachas.max()} frames ({rachas.max() / fps:.2f} s)")

    # %%
    import os
    ANOT_PATH = "../data/reference/anotaciones_manuales.csv"
    ESCALA = 4
    frames_manual = np.linspace(0, n_frames - 1, 25).astype(int)

    def anotar_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, frame = cap.read()
        roi, _ = recortar_roi(frame, (int(centros[idx, 0]), int(centros[idx, 1])))
        grande = cv2.resize(roi, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_CUBIC)
        puntos = []
        def click(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN: puntos.append((x, y))
        ventana = f"frame {idx} - clic en el borde (min 5) | Enter: aceptar | u: deshacer | s: saltear"
        cv2.namedWindow(ventana); cv2.setMouseCallback(ventana, click)
        while True:
            vis = grande.copy()
            for p in puntos: cv2.circle(vis, p, 3, (0, 0, 255), -1)
            if len(puntos) >= 5:
                cv2.ellipse(vis, cv2.fitEllipse(np.array(puntos, np.float32)), (0, 255, 0), 1)
            cv2.imshow(ventana, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and len(puntos) >= 5: break
            if k == ord("u") and puntos: puntos.pop()
            if k == ord("s"): puntos = None; break
        cv2.destroyAllWindows(); cv2.waitKey(1)
        if puntos is None: return None
        (_, _), (d1, d2), _ = cv2.fitEllipse(np.array(puntos, np.float32) / ESCALA)
        area = np.pi * d1 * d2 / 4
        return area, 2 * np.sqrt(area / np.pi)

    # Se retoma desde lo que ya esté anotado
    manual = pd.read_csv(ANOT_PATH) if os.path.exists(ANOT_PATH) else pd.DataFrame(columns=["frame", "Área manual (px²)", "Diámetro manual (px)"])
    for idx in frames_manual:
        if idx in manual["frame"].values: continue
        res = anotar_frame(idx)
        if res is not None:
            manual.loc[len(manual)] = [idx, *res]
            manual.to_csv(ANOT_PATH, index=False)

    manual = pd.read_csv(ANOT_PATH)
    manual = manual[manual["frame"].isin(frames_manual)]
    comp = manual.merge(df, on="frame")

    def metricas(auto, man):
        e = auto - man
        return {"n": len(e), "MAE": e.abs().mean(), "RMSE": np.sqrt((e**2).mean()),
                "Sesgo (auto−manual)": e.mean(), "Error relativo medio (%)": 100 * (e.abs() / man).mean()}

    tabla = pd.DataFrame({
        "Área (px²)":    metricas(comp["Área (px²)"], comp["Área manual (px²)"]),
        "Diámetro (px)": metricas(comp["Diámetro (px)"], comp["Diámetro manual (px)"]),
    }).T
    display(tabla.round(2))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (auto, man, nombre) in zip(axs, [("Área (px²)", "Área manual (px²)", "Área (px²)"),
                                             ("Diámetro (px)", "Diámetro manual (px)", "Diámetro (px)")]):
        ax.scatter(comp[man], comp[auto], c=np.where(comp["Interpolado"], "tab:red", "tab:blue"), s=20)
        lim = [min(comp[man].min(), comp[auto].min()), max(comp[man].max(), comp[auto].max())]
        ax.plot(lim, lim, "k--", linewidth=0.8)
        ax.set_xlabel(f"{nombre} manual"); ax.set_ylabel(f"{nombre} automático"); ax.set_title(nombre)
    plt.tight_layout(); plt.show()

    # La segmentación automática concuerda bien con la medición manual en los 25 frames anotados: el error medio es de 2.6% en el diámetro y de 5.3% en el área, con un sesgo pequeño. El único punto que se aleja de la diagonal corresponde a un frame interpolado. Se considera que la segmentación es adecuada para calcular las curvas temporales.

    # A continuación se grafican el diámetro D(t) y el área A(t) de la pupila a lo largo del video. En rojo se marcan los frames que se interpolaron.

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax, col in [(axs[0], "Diámetro (px)"),
                             (axs[1], "Área (px²)")]:
        ax.plot(df["t (s)"], df[col], linewidth=0.9, label="Automático")
        interp = df[df["Interpolado"]]
        ax.scatter(interp["t (s)"], interp[col], color="r", s=10, zorder=3, label="Interpolado")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    axs[0].legend(loc="upper right")
    axs[1].set_xlabel("Tiempo (s)")
    axs[0].set_title("Evolución temporal de la pupila")
    plt.tight_layout()
    plt.show()

    # En la curva se observa que el diámetro de la pupila se mantiene en torno a 55–60 px (área de ~2500 px²) durante los primeros ~21 s y luego disminuye de forma marcada, hasta estabilizarse en ~35 px (área de ~1000 px²) a partir de los ~23 s. Esta caída corresponde a la contracción de la pupila. Los picos aislados, probablemente se deban a artefactos u otros ruidos no detectados anteriormente por el criterio de circularidad, como parpadeos o errores puntuales de segmentación.

    # Para calcular el tiempo de latencia, buscamos el intervalo entre la aplicación del flash y el inicio de la constricción de la pupila.
    # El instante del estímulo se detecta como el salto de brillo del video, y el inicio de la constricción como el primer instante en que el diámetro cae por debajo de la línea de base previa al estímulo y se mantiene por debajo durante 5 frames consecutivos.

    T_ESTIMULO = None

    # Brillo medio de cada frame
    brillo = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            brillo.append(np.nan); continue
        chico = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        brillo.append(cv2.mean(cv2.cvtColor(chico, cv2.COLOR_BGR2GRAY))[0])
    df["Brillo"] = brillo

    b = df["Brillo"].rolling(5, center=True, min_periods=1).median()
    salto = b.diff(5).abs()
    i_pico = int(salto.idxmax())

    base = df["Brillo"].iloc[max(0, i_pico - 60): i_pico - 10]
    mu, sd = base.median(), base.std()
    umbral = mu + max(3 * sd, 0.05 * (df["Brillo"].iloc[i_pico] - mu))

    ini = max(0, i_pico - 15)
    i_estimulo = ini + int(np.argmax(df["Brillo"].iloc[ini:i_pico + 1].values > umbral))

    if T_ESTIMULO is None:
        T_ESTIMULO = df.loc[i_estimulo, "t (s)"]
    print(f"Frame del estímulo: {i_estimulo} | t = {T_ESTIMULO:.3f} s")

    # Gráfico general
    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axs[0].plot(df["t (s)"], df["Brillo"], linewidth=0.9); axs[0].set_ylabel("Brillo medio del frame")
    axs[1].plot(df["t (s)"], df["Diámetro (px)"], linewidth=0.9); axs[1].set_ylabel("Diámetro (px)")
    for ax in axs:
        ax.axvline(T_ESTIMULO, color="r", linestyle="--", linewidth=0.9); ax.grid(alpha=0.3)
    axs[1].set_xlabel("Tiempo (s)")
    plt.tight_layout(); plt.show()

    N_SOSTENIDO = 5          # frames consecutivos por debajo de la línea base
    VENTANA_BASE_S = 2.0     # segundos previos al estímulo para estimar la línea base
    K_SIGMA = 3              # cuántos desvíos por debajo de la base cuenta como "caída"

    d = df["Diámetro (px)"].values

    # Línea base de diámetro
    n_base = int(round(VENTANA_BASE_S * fps))
    base_d = d[max(0, i_estimulo - n_base): i_estimulo]
    mu_d, sd_d = np.nanmedian(base_d), np.nanstd(base_d)
    umbral_d = mu_d - K_SIGMA * sd_d

    # Primer frame desde el estímulo
    debajo = d < umbral_d
    i_constr = None
    for i in range(i_estimulo, n_frames - N_SOSTENIDO + 1):
        if debajo[i:i + N_SOSTENIDO].all():
            i_constr = i          
            break

    if i_constr is None:
        print("No se detectó constricción sostenida después del estímulo.")

    T_CONSTRICCION = df.loc[i_constr, "t (s)"]
    print(f"Línea base: {mu_d:.1f} px (desvío {sd_d:.2f}) | umbral: {umbral_d:.1f} px")
    print(f"Inicio de la constricción: frame {i_constr} ({T_CONSTRICCION:.3f} s)")

    plt.figure(figsize=(12, 3.5))
    plt.plot(df["t (s)"], df["Diámetro (px)"], linewidth=0.9)
    plt.axvline(T_CONSTRICCION, color="g", linestyle="--", linewidth=0.9, label="Inicio de la constricción")
    plt.xlabel("Tiempo (s)"); plt.ylabel("Diámetro (px)")
    plt.axvline(T_ESTIMULO, color="r", linestyle="--", linewidth=0.9, label="Estímulo")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    latencia_s = T_CONSTRICCION - T_ESTIMULO
    latencia_frames = i_constr - i_estimulo

    print(f"Estímulo:                  frame {i_estimulo} ({T_ESTIMULO:.3f} s)")
    print(f"Inicio de la constricción: frame {i_constr} ({T_CONSTRICCION:.3f} s)")
    print(f"Tiempo de latencia: {latencia_s * 1000:.0f} ms ({latencia_frames} frames)")

    # Teniendo en cuenta que la latencia pupilar a un flash en humanos es de 200 a 300 ms en promedio, corroboramos que los resultados obtenidos son lógicos.

    VENT_MIN = 15      # frames de la mediana móvil (descarta picos aislados)

    a = df["Área (px²)"]
    a_suav = a.rolling(VENT_MIN, center=True, min_periods=1).median()

    # Área basal: mediana de la serie suavizada en la ventana previa al estímulo (oscuridad)
    area_basal = a_suav.iloc[max(0, i_estimulo - n_base): i_estimulo].median()

    # Área mínima: mínimo de la serie suavizada desde el inicio de la constricción
    i_min = int(a_suav.iloc[i_constr:].idxmin())
    area_min = a_suav.iloc[i_min]
    t_min = df.loc[i_min, "t (s)"]

    # Amplitud de respuesta
    amplitud = area_basal - area_min
    amplitud_pct = 100 * amplitud / area_basal

    print(f"Área basal:  {area_basal:.0f} px²")
    print(f"Área mínima: {area_min:.0f} px² (frame {i_min}, {t_min:.2f} s)")
    print(f"Amplitud de respuesta: {amplitud:.0f} px² ({amplitud_pct:.1f} % del área basal)")

    # Gráfico
    plt.plot(df["t (s)"], a, linewidth=0.9, label="Área")
    plt.axhline(area_basal, color="gray", ls=":", label=f"Área basal")
    plt.axhline(area_min, color="gray", ls="--", label=f"Área mínima")
    plt.axvline(T_ESTIMULO, color="r", ls="--", linewidth=0.9, label="Estímulo")
    plt.axvline(T_CONSTRICCION, color="g", ls="--", linewidth=0.9, label="Inicio constricción")
    plt.xlabel("Tiempo (s)"); plt.ylabel("Área (px²)")
    plt.legend(loc="upper right", fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Cabe aclarar que todos los valores de área y diámetro están expresados en píxeles, no en unidades físicas. Por eso no es del todo seguro guiarse por los valores absolutos: si durante la toma la cámara se acercó o se alejó del ojo, o el ojo cambió de distancia respecto de ella, el tamaño aparente de la pupila en la imagen pudo variar y modificar los resultados, sin que eso corresponda a un cambio real de la pupila.
    # A continuación, calcularemos la velocidad media de constricción, que se obtiene como el cambio total entre el inicio de la constricción y el mínimo, dividido por el tiempo que tardó.

    FRAC_CAIDA = 0.9   # fracción de la caída total que marca el final de la constricción

    def constriccion(serie):
        s = serie.rolling(VENT_MIN, center=True, min_periods=1).median()
        base = s.iloc[max(0, i_estimulo - n_base): i_estimulo].median()
        minimo = s.iloc[i_constr:].min()
        nivel = base - FRAC_CAIDA * (base - minimo)
        i_fin = i_constr + int(np.argmax(s.iloc[i_constr:].values <= nivel))   # primer frame que llega al 90 % de la caída
        return s, base, minimo, i_fin

    def velocidad_media_constriccion(serie, unidad):
        s, base, minimo, i_fin = constriccion(serie)
        dur = df.loc[i_fin, "t (s)"] - T_CONSTRICCION
        vel = (base - s.iloc[i_fin]) / dur
        return pd.DataFrame({
            f"Basal ({unidad})": [base],
            f"Mínimo ({unidad})": [minimo],
            "Duración (s)": [dur],
            f"Velocidad media ({unidad}/s)": [vel],
        }, index=[""])

    display(velocidad_media_constriccion(df["Diámetro (px)"], "px").round(2))

    # Se obtiene una velocidad de constricción de 11,81 px/s.
    # Para calcular la velocidad de redilatación se utiliza otro video. Repetimos todo el procesamiento aplicado al primer video para poder obtener los datos.

    VIDEO_PATH_2 = "../data/raw/Registro B_1.MOV"

    cap2 = cv2.VideoCapture(VIDEO_PATH_2)
    if not cap2.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH_2}")
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    n_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps2:.2f} | Frames: {n_frames2} | Duración: {n_frames2 / fps2:.2f} s")

    CENTRO_INICIAL_2 = (1390, 590)
    RADIO_INICIAL_2 = 40

    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = cap2.read()
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    plt.plot(*CENTRO_INICIAL_2, "r", ms=15)
    plt.gca().add_patch(plt.Circle(CENTRO_INICIAL_2, RADIO_INICIAL_2, fill=False, color="r"))
    plt.axis("off"); plt.show()

    # %%
    UMBRAL_CIRC = 0.89

    def procesar_video(cap_v, n_v, fps_v, centro_inicial, radio_inicial):
        centros_v, centro = [], centro_inicial
        cap_v.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(n_v):
            ok, frame = cap_v.read()
            if not ok:
                centros_v.append((np.nan, np.nan)); continue
            centro = seguir_pupila(frame, centro, radio_inicial if i == 0 else RADIO_BUSQUEDA)
            centros_v.append(centro)
        centros_v = np.array(centros_v, dtype=float)

        # 2) Medición frame a frame
        filas = []
        cap_v.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(n_v):
            ok, frame = cap_v.read()
            if not ok or np.isnan(centros_v[i, 0]):
                filas.append((i, i / fps_v, np.nan, np.nan, np.nan)); continue
            roi, (cx, cy) = recortar_roi(frame, (int(centros_v[i, 0]), int(centros_v[i, 1])))
            contorno = medir_pupila(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cx=cx, cy=cy)
            if contorno is None:
                filas.append((i, i / fps_v, np.nan, np.nan, np.nan)); continue
            area = cv2.contourArea(contorno)
            filas.append((i, i / fps_v, area, 2 * np.sqrt(area / np.pi), circularidad(contorno)))

        d = pd.DataFrame(filas, columns=["frame", "t (s)", "Área auto (px²)", "Diámetro auto (px)", "Circularidad"])
        d["Válido"] = d["Circularidad"] >= UMBRAL_CIRC
        d["Área (px²)"] = d["Área auto (px²)"].where(d["Válido"]).interpolate(method="linear", limit_direction="both")
        d["Diámetro (px)"] = d["Diámetro auto (px)"].where(d["Válido"]).interpolate(method="linear", limit_direction="both")
        d["Interpolado"] = ~d["Válido"]
        return d, centros_v

    df2, centros2 = procesar_video(cap2, n_frames2, fps2, CENTRO_INICIAL_2, RADIO_INICIAL_2)

    inv = ~df2["Válido"]
    rachas = inv.ne(inv.shift()).cumsum()[inv].value_counts()
    print(f"Frames interpolados: {inv.sum()} de {len(df2)} ({100 * inv.mean():.1f} %) | "
          f"racha más larga: {rachas.max()} frames ({rachas.max() / fps2:.2f} s)")

    # %%
    indices = np.linspace(0, n_frames2 - 1, 12).astype(int)

    fig, axs = plt.subplots(3, 4, figsize=(18, 8))
    for ax, idx in zip(axs.ravel(), indices):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap2.read()
        roi, (cx, cy) = recortar_roi(frame, (int(centros2[idx, 0]), int(centros2[idx, 1])))
        vis = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        contorno = medir_pupila(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cx=cx, cy=cy)
        if contorno is None:
            titulo = f"frame {idx}: sin segmentación"
        else:
            cv2.drawContours(vis, [contorno], -1, (0, 255, 0), 1)
            titulo = f"frame {idx} ({idx / fps2:.1f} s)\nC = {circularidad(contorno):.2f}"
        ax.imshow(vis); ax.set_title(titulo, fontsize=10); ax.axis("off")
    plt.tight_layout(); plt.show()

    # %%
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, col in [(axs[0], "Diámetro (px)"), (axs[1], "Área (px²)")]:
        ax.plot(df2["t (s)"], df2[col], linewidth=0.9, label="Automático")
        interp = df2[df2["Interpolado"]]
        ax.scatter(interp["t (s)"], interp[col], color="r", s=10, zorder=3, label="Interpolado")
        ax.set_ylabel(col); ax.grid(alpha=0.3)
    axs[0].legend(loc="upper right"); axs[1].set_xlabel("Tiempo (s)")
    axs[0].set_title("Evolución temporal de la pupila")
    plt.tight_layout(); plt.show()

    # %%
    N_SOSTENIDO = 5
    VENTANA_BASE_S = 2.0
    K_SIGMA = 3
    VENT_MIN = 15
    FRAC_SUBIDA = 0.9      # fracción de la subida total que marca el final de la redilatación
    T_REF = (9, 12)

    n_base2 = int(round(VENTANA_BASE_S * fps2))
    t2 = df2["t (s)"].values
    m_ref = (t2 >= T_REF[0]) & (t2 <= T_REF[1])
    i_ref_fin = int(np.where(m_ref)[0][-1])

    # Mínimo de la serie suavizada (descarta picos aislados) y su ruido
    s_dia2 = df2["Diámetro (px)"].rolling(VENT_MIN, center=True, min_periods=1).median().values
    mu_c, sd_c = np.median(s_dia2[m_ref]), np.std(s_dia2[m_ref])
    umbral_up = mu_c + K_SIGMA * sd_c

    encima = s_dia2 > umbral_up
    i_red = None
    for i in range(i_ref_fin, n_frames2 - N_SOSTENIDO + 1):
        if encima[i:i + N_SOSTENIDO].all():
            i_red = i              # primer frame de la racha, no el último
            break
    if i_red is None:
        raise RuntimeError("No se detectó redilatación sostenida después de la ventana de referencia.")
    T_REDILATACION = df2.loc[i_red, "t (s)"]
    print(f"Nivel previo: {mu_c:.1f} px (desvío {sd_c:.2f}) | umbral: {umbral_up:.1f} px")
    print(f"Inicio de la redilatación: frame {i_red} ({T_REDILATACION:.3f} s)")

    # Velocidad media
    def redilatacion(serie):
        s = serie.rolling(VENT_MIN, center=True, min_periods=1).median()
        previo = s[m_ref].median()
        final = s.iloc[-n_base2:].median()
        if final <= previo:
            raise RuntimeError("El nivel final no es mayor que el nivel previo: revisar la señal.")
        nivel = previo + FRAC_SUBIDA * (final - previo)
        i_fin = i_red + int(np.argmax(s.iloc[i_red:].values >= nivel))   # primer frame que llega al 90 % de la subida
        return s, previo, final, i_fin

    def velocidad_media_redilatacion(serie, unidad):
        s, previo, final, i_fin = redilatacion(serie)
        dur = df2.loc[i_fin, "t (s)"] - T_REDILATACION
        vel = (s.iloc[i_fin] - previo) / dur
        return pd.DataFrame({
            f"Nivel previo ({unidad})": [previo],
            f"Nivel final ({unidad})": [final],
            "Duración (s)": [dur],
            f"Velocidad media ({unidad}/s)": [vel],
        }, index=[""])

    print("DIÁMETRO")
    display(velocidad_media_redilatacion(df2["Diámetro (px)"], "px").round(2))

    # %% [markdown]
    # Se obtiene una velocidad de redilatación de 12,05 px/s. Los dos valores de velocidades son practicamente iguales, solo hay un ~2% de diferencia entre ellos, lo que está dentro de la incertidumbre de la medición.
    # Cabe aclarar que la constricción suele ser más rápida que la redilatación. Acá no se observa esa diferencia, y puede deberse a posibles cambios de escala de la cámara entre tomas, y que se está haciendo el cálculo a partir de una sola medición, sin poder evaluar errores a partir del análisis de otros videos.

    # %% [markdown]
    # En cuanto a presencia de artefactos, se implementaron los siguientes métodos a lo largo del análisis para poder filtrarlos e interpolarlos:
    # - Movimientos oculares y de cabeza: la pupila se sigue frame a frame con una ventana de búsqueda centrada en su posición del frame anterior, y la ROI se recorta centrada en ese punto, de modo que el movimiento no desplaza la pupila fuera del análisis.
    # - Reflejos del flash: los píxeles muy brillantes cercanos a la pupila se detectan, se dilatan y se rellenan con los píxeles de alrededor. Además, se aplica una apertura morfológica y un suavizado gaussiano que eliminan reflejos y pestañas.
    # - Pestañas y sombras sobre el borde: en la segmentación por Watershed se prueban tres radios de marcador y se elige el contorno con mayor contraste de borde. Luego se aplican una apertura de 11×11, un cierre de 21×21 y la envolvente convexa.
    # 
    # - Para cada frame se calcula la circularidad del contorno. Se consideran inválidos los frames con circularidad < 0,89 y aquellos donde no se pudo segmentar la pupila, que pudo haber sido por parpadeos o errores puntuales de segmentación. Cabe aclarar que este umbral se eligió observando solo el primer video analizado.
    # - Los valores de área y diámetro de los frames inválidos se reemplazan por interpolación lineal entre los frames válidos vecinos.

def modulo_4():

    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import heapq
    from collections import Counter
    from skimage.feature import graycomatrix, graycoprops
    from skimage.metrics import (peak_signal_noise_ratio, structural_similarity)

    # %%
    # Se usa la ROI seleccionada en el menú (roi_gray y roi_rgb).

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(roi_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("ROI original en gris")

    axes[1].imshow(roi_rgb)
    axes[1].set_title("ROI original en RGB")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("Dimensiones ROI gris:", roi_gray.shape)
    print("Dimensiones ROI RGB:", roi_rgb.shape)

    # %% [markdown]
    # Se aplica el algoritmo de Huffman a las regiones de interés (ROI) en escala de grises y en color RGB. Este método de compresión sin pérdida asigna códigos binarios de longitud variable a los valores de píxel según su frecuencia de aparición, utilizando códigos más cortos para los valores más frecuentes. De esta forma se aprovecha que algunos valores de píxel aparecen más que otros para representarlos con menos bits.

    # %%
    def huffman_codificar(imagen):
        datos = np.asarray(imagen, dtype=np.uint8).ravel()
        frecuencias = Counter(datos.tolist())

        heap = []
        contador = 0

        for simbolo, frecuencia in frecuencias.items():
            heapq.heappush(heap, (frecuencia, contador, simbolo))
            contador += 1

        # Caso especial: todos los píxeles tienen el mismo valor
        if len(heap) == 1:
            codigos = {heap[0][2]: "0"}

        else:
            while len(heap) > 1:
                f1, _, n1 = heapq.heappop(heap)
                f2, _, n2 = heapq.heappop(heap)

                nodo = (n1, n2)
                heapq.heappush(heap, (f1 + f2, contador, nodo))
                contador += 1

            raiz = heap[0][2]
            codigos = {}

            def construir_codigos(nodo, prefijo=""):
                if isinstance(nodo, int):
                    codigos[nodo] = prefijo or "0"
                else:
                    izquierda, derecha = nodo
                    construir_codigos(izquierda, prefijo + "0")
                    construir_codigos(derecha, prefijo + "1")

            construir_codigos(raiz)

        bits = "".join(codigos[int(valor)] for valor in datos)

        return bits, codigos, frecuencias

    def huffman_decodificar(bits, codigos, forma):
        # Invertir el diccionario: código binario -> valor de píxel
        decodigos = {codigo: simbolo for simbolo, codigo in codigos.items()}

        datos = []
        codigo_actual = ""

        for bit in bits:
            codigo_actual += bit

            if codigo_actual in decodigos:
                datos.append(decodigos[codigo_actual])
                codigo_actual = ""

        if codigo_actual:
            raise ValueError("La secuencia binaria no se pudo decodificar por completo.")

        # Recuperar las dimensiones originales
        imagen_recuperada = np.array(datos, dtype=np.uint8).reshape(forma)

        return imagen_recuperada

    imagenes = {"ROI gris": roi_gray,"ROI RGB": roi_rgb}
    resultados_huffman = {}
    for nombre, imagen in imagenes.items():
        # 1. Codificacion
        bits, codigos, frecuencias = huffman_codificar(imagen)
        # 2. Decodificación
        reconstruida = huffman_decodificar(bits, codigos, imagen.shape)
        # 3. Tamaños
        bits_originales = imagen.size * 8
        bits_codificados = len(bits)

        # Bytes necesarios para almacenar el flujo binario empaquetado
        bytes_comprimidos = (bits_codificados + 7) // 8
        bits_empaquetados = bytes_comprimidos * 8

        # Métricas sin contar el diccionario ni otros metadatos
        reduccion = 100 * (1 - bits_codificados / bits_originales)
        ratio = bits_originales / bits_codificados

        # 4. Verificar que sea una compresión sin pérdida
        error_maximo = np.max(np.abs(imagen.astype(np.int16) - reconstruida.astype(np.int16)))

        resultados_huffman[nombre] = {
            "original": imagen,
            "reconstruida": reconstruida,
            "bits": bits,
            "codigos": codigos,
            "frecuencias": frecuencias
        }

        # 5. Mostrar imágenes
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        if imagen.ndim == 2:
            axes[0].imshow(imagen, cmap="gray", vmin=0, vmax=255)
            axes[1].imshow(reconstruida, cmap="gray", vmin=0, vmax=255)
        else:
            axes[0].imshow(imagen)
            axes[1].imshow(reconstruida)

        axes[0].set_title(f"{nombre} original")
        axes[1].set_title(f"{nombre} reconstruida")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # Mostramos la información del algoritmo y lo que logró reducir y comprimir
    imagenes = {"ROI gris": roi_gray,"ROI RGB": roi_rgb}

    for nombre, imagen in imagenes.items():
        print(f"\n--- {nombre} ---")
        print(f"Dimensiones: {imagen.shape}")
        print(f"Valores de píxel distintos: {len(frecuencias)}")
        print(f"Bits originales: {bits_originales:,}")
        print(f"Bits codificados: {bits_codificados:,}")
        print(f"Bytes del flujo empaquetado: {bytes_comprimidos:,}")
        print(f"Reducción teórica: {reduccion:.2f}%")
        print(f"Ratio de compresión: {ratio:.3f}:1")
        print(f"Error máximo de reconstrucción: {error_maximo}")
        print(f"¿Reconstrucción idéntica?: {np.array_equal(imagen, reconstruida)}")

    # El algoritmo de Huffman permite comprimir la información de una imagen sin pérdida, asignando códigos binarios de longitud variable según la frecuencia de aparición de los valores de píxel. De esta manera, es posible reducir la cantidad de bits necesarios para representar la imagen sin modificar su contenido visual. Al decodificar los datos comprimidos, se recupera la imagen original exactamente. La eficiencia del método depende de la distribución de los valores de píxel, por lo que la reducción obtenida puede variar entre la ROI en escala de grises y la ROI en color RGB.
    # En la ejecución obtenida, la reducción teórica fue del 8,08 % para la imagen denominada ROI gris y del 9,86 % para la ROI RGB. Estos porcentajes corresponden a la cantidad de bits de los códigos generados, sin incluir la información adicional necesaria para almacenar el árbol de Huffman y reconstruir la imagen.

    # Ahora aplicamos RLE en dos direcciones de recorrido (izquierda y derecha)
    # RLE también es un método de compresión sin pérdida que aprovecha la aparición consecutiva de valores iguales. En lugar de almacenar cada valor de píxel por separado, guarda el valor y la cantidad de veces que se repite.

    # Trabajamos con la ROI en escala de grises para que cada píxel tenga un único valor entre 0 y 255. Esta función codifica y decodifica los datos, y después la reutilizamos en ambos recorridos
    # ROI que vamos a comprimir
    imagen = roi_gray.astype(np.uint8)

    def rle_codificar_horizontal(imagen):
        codificacion = []

        for fila in imagen:
            if len(fila) == 0:
                continue

            valor_actual = int(fila[0])
            longitud = 1

            for valor in fila[1:]:
                valor = int(valor)

                if valor == valor_actual:
                    longitud += 1
                else:
                    codificacion.append((valor_actual, longitud))
                    valor_actual = valor
                    longitud = 1

            codificacion.append((valor_actual, longitud))

        return codificacion

    def rle_codificar_vertical(imagen):
        codificacion = []

        for columna in imagen.T:
            if len(columna) == 0:
                continue

            valor_actual = int(columna[0])
            longitud = 1

            for valor in columna[1:]:
                valor = int(valor)

                if valor == valor_actual:
                    longitud += 1
                else:
                    codificacion.append((valor_actual, longitud))
                    valor_actual = valor
                    longitud = 1

            codificacion.append((valor_actual, longitud))
        return codificacion


    def rle_decodificar_horizontal(codificacion, forma):
        alto, ancho = forma
        datos = np.zeros(alto * ancho, dtype=np.uint8)
        indice = 0

        for valor, longitud in codificacion:
            datos[indice:indice + longitud] = valor
            indice += longitud
        return datos.reshape(forma)


    def rle_decodificar_vertical(codificacion, forma):
        alto, ancho = forma
        datos = np.zeros(alto * ancho, dtype=np.uint8)
        indice = 0

        for valor, longitud in codificacion:
            datos[indice:indice + longitud] = valor
            indice += longitud

        # El flujo se guardó columna por columna
        return datos.reshape(ancho, alto).T

    # RLE horizontal: recorremos la imagen fila por fila

    # Codificar horizontalmente
    rle_horizontal = rle_codificar_horizontal(imagen)

    # Reconstruir la imagen
    reconstruida_horizontal = rle_decodificar_horizontal(
        rle_horizontal, imagen.shape
    )

    # Calcular métricas
    pixeles_originales = imagen.size
    bytes_originales = imagen.nbytes
    cantidad_corridas_h = len(rle_horizontal)

    # Estimación: 1 byte por valor + 4 bytes por longitud
    bytes_rle_h = cantidad_corridas_h * 5

    reduccion_h = 100 * (1 - bytes_rle_h / bytes_originales)
    if bytes_rle_h > 0:
        ratio_h = bytes_originales / bytes_rle_h
    else:
        ratio_h = float("inf")

    # Visualizar original y reconstruida
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(imagen, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("ROI original")

    axes[1].imshow(
        reconstruida_horizontal, cmap="gray", vmin=0, vmax=255
    )
    axes[1].set_title("ROI reconstruida - RLE horizontal")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("Corridas horizontales:", cantidad_corridas_h)
    print("Primeras 15 corridas:", rle_horizontal[:15])
    print("Tamaño original:", bytes_originales, "bytes")
    print("Tamaño RLE estimado:", bytes_rle_h, "bytes")
    print(f"Reducción estimada: {reduccion_h:.2f}%")
    print(f"Ratio de compresión: {ratio_h:.3f}:1")
    print(
        "¿Reconstrucción idéntica?:",
        np.array_equal(imagen, reconstruida_horizontal)
    )

    # RLE vertical: recorremos la imagen columna por columna
    # Codificar verticalmente
    rle_vertical = rle_codificar_vertical(imagen)

    # Reconstruir la imagen
    reconstruida_vertical = rle_decodificar_vertical(
        rle_vertical, imagen.shape
    )

    # Calcular métricas
    cantidad_corridas_v = len(rle_vertical)

    # Estimación: 1 byte por valor + 4 bytes por longitud
    bytes_rle_v = cantidad_corridas_v * 5

    reduccion_v = 100 * (1 - bytes_rle_v / bytes_originales)
    if bytes_rle_v > 0:
        ratio_v = bytes_originales / bytes_rle_v
    else:
        ratio_v = float("inf")

    # Visualizar original y reconstruida
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(imagen, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("ROI original")

    axes[1].imshow(
        reconstruida_vertical, cmap="gray", vmin=0, vmax=255
    )
    axes[1].set_title("ROI reconstruida - RLE vertical")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("Corridas verticales:", cantidad_corridas_v)
    print("Primeras 15 corridas:", rle_vertical[:15])
    print("Tamaño original:", bytes_originales, "bytes")
    print("Tamaño RLE estimado:", bytes_rle_v, "bytes")
    print(f"Reducción estimada: {reduccion_v:.2f}%")
    print(f"Ratio de compresión: {ratio_v:.3f}:1")
    print(
        "¿Reconstrucción idéntica?:",
        np.array_equal(imagen, reconstruida_vertical)
    )

    # Ahora comparamos los resultados obtenidos por ambos recorridos

    df_rle = pd.DataFrame({
        "Método": ["RLE horizontal", "RLE vertical"],
        "Cantidad de corridas": [
            cantidad_corridas_h, cantidad_corridas_v
        ],
        "Tamaño original (bytes)": [
            bytes_originales, bytes_originales
        ],
        "Tamaño RLE estimado (bytes)": [
            bytes_rle_h, bytes_rle_v
        ],
        "Reducción estimada (%)": [
            reduccion_h, reduccion_v
        ],
        "Reconstrucción exacta": [
            np.array_equal(imagen, reconstruida_horizontal),
            np.array_equal(imagen, reconstruida_vertical)
        ]
    })

    display(df_rle.round(2))

    # En ambos casos, la reconstrucción es exacta (True), lo que confirma que RLE es un método de compresión sin pérdida. Sin embargo, el tamaño estimado de los datos codificados es considerablemente mayor que el de la imagen original: pasa de 41.600 bytes a 161.180 bytes para el recorrido horizontal y a 166.405 bytes para el vertical. Esto suscede porque la imagen contiene pocos píxeles consecutivos con valores idénticos, por lo que se generan muchas corridas. Como cada corrida requiere almacenar tanto el valor del píxel como la longitud de la secuencia, la información adicional supera el ahorro obtenido al agrupar píxeles repetidos.
    # El recorrido horizontal resulta ligeramente más eficiente ya que el aumento es menor, aunque la diferencia no es suficiente para que el método consiga comprimir la imagen.

    # Luego, haremos una compresión por JPEG, un algortimo de compresión con pérdida, a comparación de los dos métodos anteriores. JPEG reduce el tamaño de la imagen descartando parte de la información visual. Al variar el factor de calidad Q, vamos a poder observar cómo cambia la apariencia de la ROI del ojo

    # Imagen original en RGB
    imagen_original = roi_rgb

    # Factores de calidad solicitados
    calidades = [90, 50, 10, 5]

    # Diccionarios para guardar los resultados
    imagenes_jpeg = {}
    archivos_jpeg = {}
    resultados_jpeg = []

    # Convertir de RGB a BGR para OpenCV
    imagen_bgr = cv2.cvtColor(
        imagen_original, cv2.COLOR_RGB2BGR
    )
    for Q in calidades:
        # Comprimir la imagen en formato JPEG
        exito, buffer = cv2.imencode(
            ".jpg",
            imagen_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, Q]
        )

        if not exito:
            raise RuntimeError(f"No se pudo comprimir con Q={Q}")

        # Guardar los bytes reales del archivo JPEG en memoria
        archivos_jpeg[Q] = buffer.tobytes()

        # Reconstruir la imagen a partir del JPEG codificado
        imagen_decodificada = cv2.imdecode(
            buffer, cv2.IMREAD_COLOR
        )

        imagen_rgb = cv2.cvtColor(
            imagen_decodificada, cv2.COLOR_BGR2RGB
        )

        imagenes_jpeg[Q] = imagen_rgb

        # Calcular métricas de calidad y tamaño
        tamano_original = len(
            cv2.imencode(".png", imagen_bgr)[1]
        )
        tamano_jpeg = len(archivos_jpeg[Q])

        error_medio = np.mean(
            np.abs(
                imagen_original.astype(np.float32)
                - imagen_rgb.astype(np.float32)
            )
        )

        reduccion = 100 * (
            1 - tamano_jpeg / tamano_original
        )

        resultados_jpeg.append({
            "Calidad JPEG": Q,
            "Tamaño JPEG (bytes)": tamano_jpeg,
            "Tamaño PNG de referencia (bytes)": tamano_original,
            "Reducción respecto a PNG (%)": reduccion,
            "Error absoluto medio": error_medio
        })

    print("Compresión JPEG completada.")

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    axes[0].imshow(imagen_original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, Q in zip(axes[1:], calidades):
        ax.imshow(imagenes_jpeg[Q])
        ax.set_title(f"JPEG - Q={Q}", fontsize=12)
        ax.axis("off")

    plt.suptitle(
        "Comparación visual de la compresión JPEG",
        fontsize=15
    )
    plt.tight_layout()
    plt.show()

    df_jpeg = pd.DataFrame(resultados_jpeg)

    display(df_jpeg.round(3))

    # Gráfico del tamaño de cada archivo JPEG
    plt.figure(figsize=(8, 4))
    plt.bar(
        [str(Q) for Q in calidades],
        [len(archivos_jpeg[Q]) for Q in calidades]
    )
    plt.xlabel("Calidad JPEG (Q)")
    plt.ylabel("Tamaño del archivo (bytes)")
    plt.title("Tamaño del archivo según la calidad JPEG")
    plt.tight_layout()
    plt.show()

    # Por lo tanto, podemos observar que a medida que Q disminuye, disminuye la calidad visual a medida que la compresión se intensifica.
    # Q = 90 presenta una mayor calidad visual y, normalmente, menor pérdida de detalles.
    # Q = 50 presenta una compresión más intensa, con pocas diferencias visibles.
    # Q = 10 ya presenta una pérdida de detalles más pronunciada y posibles artefactos.
    # Q = 5, por último, posee una compresión muy intensa, que puede producir bloques y distorsiones visibles.

    # En síntesis, a diferencia de Huffman y RLE, JPEG es un método de compresión con pérdida, por lo que la imagen reconstruida no es idéntica a la original. La comparación entre las calidades evaluadas (Q=90, 50, 10, 5) permite analizar el compromiso entre el ahorro de espacio y la conservación de los detalles de la ROI del ojo, especialmente en los bordes de la pupila y el iris.
    # Imagen original: un canal, 8 bits por píxel
    original = roi_gray.astype(np.uint8)

    alto, ancho = original.shape
    bytes_originales = original.nbytes

    # 1. Huffman
    bits_h, codigos_h, frecuencias_h = huffman_codificar(original)

    recon_h = huffman_decodificar(
        bits_h, codigos_h, original.shape
    )

    # Bytes necesarios para empaquetar los bits
    bytes_huffman = (len(bits_h) + 7) // 8

    # 2. RLE horizontal y vertical
    rle_h = rle_codificar_horizontal(original)
    rle_v = rle_codificar_vertical(original)

    recon_rle_h = rle_decodificar_horizontal(
        rle_h, original.shape
    )

    recon_rle_v = rle_decodificar_vertical(
        rle_v, original.shape
    )

    # Tamaño estimado: 1 byte para valor + 4 para longitud
    bytes_rle_h = len(rle_h) * 5
    bytes_rle_v = len(rle_v) * 5

    # 3. JPEG para cada calidad
    calidades = [90, 50, 10, 5]

    imagenes_jpeg_gris = {}
    bytes_jpeg = {}

    for Q in calidades:
        exito, buffer = cv2.imencode(
            ".jpg",
            original,
            [cv2.IMWRITE_JPEG_QUALITY, Q]
        )

        if not exito:
            raise RuntimeError(f"Falló JPEG para Q={Q}")

        reconstruida = cv2.imdecode(
            buffer, cv2.IMREAD_GRAYSCALE
        )

        bytes_jpeg[Q] = len(buffer)
        imagenes_jpeg_gris[Q] = reconstruida

    # Reunir todas las imágenes reconstruidas
    imagenes_evaluadas = {
        "Huffman": recon_h,
        "RLE horizontal": recon_rle_h,
        "RLE vertical": recon_rle_v,
        **{
            f"JPEG Q={Q}": imagenes_jpeg_gris[Q]
            for Q in calidades
        }
    }

    print("Dimensiones de la ROI:", original.shape)
    print("Tamaño original:", bytes_originales, "bytes")
    print("Cantidad de píxeles:", original.size)

    # Tamaños de las representaciones comprimidas
    tamanos_comprimidos = {
        "Huffman": bytes_huffman,
        "RLE horizontal": bytes_rle_h,
        "RLE vertical": bytes_rle_v,
        **{
            f"JPEG Q={Q}": bytes_jpeg[Q]
            for Q in calidades
        }
    }

    filas = []

    for metodo, reconstruida in imagenes_evaluadas.items():

        tamano = tamanos_comprimidos[metodo]

        # Ratio: tamaño original / tamaño comprimido
        ratio = bytes_originales / tamano

        # Reducción porcentual
        reduccion = 100 * (1 - tamano / bytes_originales)

        # Métricas de calidad
        psnr = peak_signal_noise_ratio(
            original, reconstruida, data_range=255
        )

        ssim = structural_similarity(
            original, reconstruida, data_range=255
        )

        filas.append({
            "Método": metodo,
            "Tamaño (bytes)": tamano,
            "Ratio de compresión": ratio,
            "Reducción (%)": reduccion,
            "PSNR (dB)": psnr,
            "SSIM": ssim,
            "Reconstrucción idéntica": np.array_equal(
                original, reconstruida
            )
        })

    df_comparacion = pd.DataFrame(filas)

    display(df_comparacion.round(4))

    # Ahora comparamos las métricas de PSNR y SSIM.

    df_grafico = df_comparacion.replace(
        [np.inf, -np.inf], np.nan
    )

    plt.figure(figsize=(9, 5))

    for _, fila in df_grafico.iterrows():
        if pd.notna(fila["PSNR (dB)"]):
            plt.scatter(
                fila["Ratio de compresión"],
                fila["PSNR (dB)"],
                s=70
            )
            plt.annotate(
                fila["Método"],
                (
                    fila["Ratio de compresión"],
                    fila["PSNR (dB)"]
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

    plt.xlabel("Ratio de compresión (original / comprimido)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR en función del ratio de compresión")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # El PSNR de una reconstrucción exacta es infinito pero lo representamos como NaN en el gráfico para no distorsionar la escala. Es por esto que no figura en la imagen.

    plt.figure(figsize=(9, 5))

    for _, fila in df_comparacion.iterrows():
        plt.scatter(
            fila["Ratio de compresión"],
            fila["SSIM"],
            s=70
        )
        plt.annotate(
            fila["Método"],
            (
                fila["Ratio de compresión"],
                fila["SSIM"]
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    plt.xlabel("Ratio de compresión (original / comprimido)")
    plt.ylabel("SSIM")
    plt.title("SSIM en función del ratio de compresión")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Luego realizamos la observación visual para determinar a partir de que Q emergen los artefactos de bloques en JPEG. 
    # Nuevamente imprimimos las imagenes comprimidas mediante este método para realizar la evaluación.

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")

    for ax, Q in zip(axes[1:], calidades):
        ax.imshow(
            imagenes_jpeg_gris[Q],
            cmap="gray",
            vmin=0,
            vmax=255
        )
        ax.set_title(f"JPEG Q={Q}")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Artefactos visuales según la calidad JPEG")
    plt.tight_layout()
    plt.show()

    # Para apreciar mejor los artefactos, ampliamos una región del ojo que contenga pupila, iris y esclerótica.
    # Zoom de la región pupilar para comparar los artefactos JPEG

    alto, ancho = original.shape[:2]
    if PUPIL_CENTER is None:
        raise RuntimeError(
            "No hay centro pupilar seleccionado. Volvé a elegir la ROI y hacé clic en el centro de la pupila."
        )

    cx, cy = map(int, PUPIL_CENTER)
    cx = int(np.clip(cx, 0, ancho - 1))
    cy = int(np.clip(cy, 0, alto - 1))

    # Recorte bastante más pequeño que la ROI para que el zoom sea visible.
    # Se muestra el mismo recorte para todas las calidades.
    zoom_w = min(ancho, max(24, int(ancho * 0.42)))
    zoom_h = min(alto, max(24, int(alto * 0.68)))
    x1 = max(0, min(ancho - zoom_w, cx - zoom_w // 2))
    y1 = max(0, min(alto - zoom_h, cy - zoom_h // 2))
    x2, y2 = x1 + zoom_w, y1 + zoom_h

    imagenes_zoom = [("Original", original)] + [
        (f"JPEG Q={Q}", imagenes_jpeg_gris[Q]) for Q in calidades
    ]
    fig, axes = plt.subplots(
        1, len(imagenes_zoom), figsize=(4.6 * len(imagenes_zoom), 5.5),
        squeeze=False
    )
    for ax, (nombre, imagen) in zip(axes.ravel(), imagenes_zoom):
        recorte = imagen[y1:y2, x1:x2]
        ax.imshow(
            recorte, cmap="gray", vmin=0, vmax=255,
            interpolation="nearest"
        )
        ax.set_title(nombre)
        ax.axis("off")
    plt.suptitle("Zoom de la región pupilar: comparación de artefactos JPEG")
    plt.tight_layout()
    plt.show()
    
    # Evaluación de la compresión JPEG sobre la estimación del diámetro pupilar.
    # Se conserva la imagen original y las versiones comprimidas Q=50 y Q=10.

    original = roi_gray.astype(np.uint8)

    imagenes = {"Original": original}

    for q in [50, 10]:
        ok, buffer = cv2.imencode(
            ".jpg",
            original,
            [cv2.IMWRITE_JPEG_QUALITY, q]
        )

        if not ok:
            raise RuntimeError(f"No se pudo comprimir con Q={q}")

        reconstruida = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

        if reconstruida is None:
            raise RuntimeError(f"No se pudo reconstruir la imagen Q={q}")

        imagenes[f"Q={q}"] = reconstruida

    # Verificar que se hayan creado las tres imágenes
    print("\nImágenes que se van a medir:", list(imagenes.keys()))

    if PUPIL_CENTER is None:
        raise RuntimeError(
            "No hay centro pupilar seleccionado. "
            "Usá la opción 5 del menú y seleccioná el centro de la pupila."
        )

    # El centro debe estar expresado en coordenadas de la ROI: (x, y).
    cx, cy = map(int, PUPIL_CENTER)

    # Watershed: mismos parámetros que el método utilizado en el módulo 3.
    def segmentar_y_medir(gris, centro):
        alto, ancho = gris.shape
        cx, cy = map(int, centro)

        if not (0 <= cx < ancho and 0 <= cy < alto):
            raise ValueError(
                f"El centro ({cx}, {cy}) está fuera de la ROI "
                f"de tamaño {ancho}x{alto}."
            )

        contraste = np.clip(
            gris.astype(np.float32) * 1.5, 0, 255
        ).astype(np.uint8)

        suavizada = cv2.GaussianBlur(contraste, (5, 5), 0)

        marcadores = np.zeros(gris.shape, dtype=np.int32)

        # Marcador inicial de la pupila
        cv2.circle(marcadores, (cx, cy), 20, 1, -1)

        # Marcador de fondo fuera de la región de búsqueda
        zona = np.zeros(gris.shape, dtype=np.uint8)

        cv2.ellipse(
            zona,
            (cx, cy),
            (60, 55),
            0, 0, 360,
            1, -1
        )

        marcadores[zona == 0] = 2

        ws = cv2.watershed(
            cv2.cvtColor(suavizada, cv2.COLOR_GRAY2BGR),
            marcadores.copy()
        )

        mascara = (ws == 1).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

        mascara = cv2.morphologyEx(
            mascara, cv2.MORPH_CLOSE, kernel
        )

        contornos, _ = cv2.findContours(
            mascara.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contornos:
            return None, np.nan, np.nan, mascara

        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)

        if area <= 0:
            return None, np.nan, np.nan, mascara

        diametro = 2 * np.sqrt(area / np.pi)

        return contorno, area, diametro, mascara

    # Segmentar y medir las tres imágenes
    resultados = []
    contornos = {}
    mascaras = {}

    for nombre, imagen in imagenes.items():
        contorno, area, diametro, mascara = segmentar_y_medir(
            imagen, centro=(cx, cy)
        )

        contornos[nombre] = contorno
        mascaras[nombre] = mascara

        resultados.append({
            "Imagen": nombre,
            "Área (px²)": area,
            "Diámetro equivalente (px)": diametro
        })

    tabla = pd.DataFrame(resultados).set_index("Imagen")

    # Calcular las diferencias respecto de la imagen original
    d_original = tabla.loc["Original", "Diámetro equivalente (px)"]

    if pd.notna(d_original) and d_original > 0:
        tabla["Cambio diámetro (px)"] = (
            tabla["Diámetro equivalente (px)"] - d_original
        )
        tabla["Cambio diámetro (%)"] = (
            100 * tabla["Cambio diámetro (px)"] / d_original
        )
    else:
        tabla["Cambio diámetro (px)"] = np.nan
        tabla["Cambio diámetro (%)"] = np.nan

    print("\nResultados de la segmentación y medición:")
    print(tabla.round(3).to_string())

    # Visualización: imagen con contorno y máscara binaria
    fig, axs = plt.subplots(
        2, len(imagenes),
        figsize=(5 * len(imagenes), 8),
        squeeze=False
    )
    for j, (nombre, imagen) in enumerate(imagenes.items()):
        axs[0, j].imshow(imagen, cmap="gray", vmin=0, vmax=255)

        if contornos[nombre] is not None:
            axs[0, j].contour(
                mascaras[nombre] > 0,
                levels=[0.5],
                colors="red",
                linewidths=1.5
            )

        d = tabla.loc[nombre, "Diámetro equivalente (px)"]

        texto_d = (
            f"{d:.2f} px"
            if pd.notna(d)
            else "No detectado"
        )

        axs[0, j].set_title(
            f"{nombre}\nDiámetro equivalente: {texto_d}"
        )
        axs[0, j].axis("off")

        axs[1, j].imshow(
            mascaras[nombre],
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axs[1, j].set_title("Máscara Watershed")
        axs[1, j].axis("off")

    plt.suptitle(
        "Verificación de la segmentación pupilar"
    )
    plt.tight_layout()
    plt.show()
    # Para evaluar el efecto de la compresión JPEG sobre la estimación del diámetro pupilar, se aplicó el mismo procedimiento de segmentación Watershed utilizado en el Módulo 3 a la imagen original y a sus versiones comprimidas con factores de calidad Q=50 y Q=10. 
    # A partir del área de cada contorno segmentado, se calculó el diámetro equivalente.

def menu():
    while True:
        print("""
================== PSIB: ANÁLISIS DEL OJO ==================
1. Módulo 1 — Procesamiento y análisis de imagen
2. Módulo 2 — Segmentación y morfología
3. Módulo 3 — Seguimiento y análisis temporal de la pupila
4. Módulo 4 — Compresión y calidad de imagen
5. Seleccionar nuevamente video, frame, ROI y centro pupilar
0. Salir
============================================================
""")
        opcion = input("Elegí una opción: ").strip()
        if opcion == "0":
            print("Programa finalizado.")
            break
        elif opcion == "5":
            try:
                preparar_roi()
            except Exception as exc:
                print(f"No se pudo seleccionar la ROI: {exc}")
        elif opcion in {"1", "2", "3", "4"}:
            try:
                ejecutar_modulo(int(opcion))
            except Exception as exc:
                print(f"Error en el módulo {opcion}: {exc}")
                print("Revisá las rutas, las dependencias y los parámetros para el tamaño de ROI elegido.")
        else:
            print("Opción no válida. Elegí 0, 1, 2, 3, 4 o 5.")

if __name__ == "__main__":
    menu()
