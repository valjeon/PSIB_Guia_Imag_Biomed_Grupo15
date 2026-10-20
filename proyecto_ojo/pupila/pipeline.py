"""Orquestación: ejecuta el pipeline completo sobre un video y escribe las salidas estandarizadas."""
from pathlib import Path
import traceback
import cv2
import numpy as np
import pandas as pd

from .video import preparar_referencia, guardar_vista_previa
from . import modulo1, modulo2, modulo3, modulo4

COLUMNAS_CURVA = ["frame", "tiempo_s", "area_px2", "diametro_px", "interpolado",
                  "circularidad", "centro_x_px", "centro_y_px", "brillo_medio"]

ARCHIVO_SALIDA = "resultados.csv"
# -------------------------------------------------------------- gráficos
def graficar_curva(df, P, ev, ruta=None):
    """Curva D(t) y A(t) con estímulo, inicio de constricción y mínimo. Devuelve la figura."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, col, lab in [(axs[0], "diametro_px", "Diámetro (px)"), (axs[1], "area_px2", "Área (px²)")]:
        ax.plot(df["tiempo_s"], df[col], lw=0.9, label="Automático")
        interp = df[df["interpolado"]]
        ax.scatter(interp["tiempo_s"], interp[col], c="r", s=10, zorder=3, label="Interpolado")
        for clave, color, nombre in [("i_estimulo", "r", "Estímulo"), ("i_constriccion", "g", "Inicio constricción"),
                                     ("i_redilatacion", "m", "Inicio redilatación")]:
            if ev.get(clave) is not None:
                ax.axvline(df["tiempo_s"].iloc[ev[clave]], color=color, ls="--", lw=0.9, label=nombre)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel("Tiempo (s)")
    axs[0].set_title("Evolución temporal de la pupila")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=130)
    return fig


# --------------------------------------------------------------- reporte
def generar_reporte(ctx, cfg, df, out_dir):
    """Tablas de los Módulos 1 a 4 sobre el frame de referencia, guardadas como CSV.

    Cada módulo se ejecuta aislado: si uno falla se informa y se sigue con los demás.
    """
    out_dir = Path(out_dir)
    gris, rgb = ctx["roi_gray"], ctx["roi_rgb"]
    cx, cy = ctx["centro_roi"]
    avisos = []

    def paso(nombre, f):
        try:
            f()
        except Exception as e:            # noqa: BLE001
            avisos.append(f"{nombre}: {e}")
            print(f"  [aviso] {nombre} falló: {e}")
            (out_dir / "reporte_errores.txt").open("a", encoding="utf-8").write(
                f"{nombre}\n{traceback.format_exc()}\n")

    mask_ws, cont = modulo2.segmentar_pupila(gris, cx, cy, radio=20)

    # ---- Módulo 1
    def m1():
        ruido = modulo1.agregar_ruido(gris, seed=0)
        filas = []
        for r_nombre, r_img in ruido.items():
            for f_nombre, f_img in modulo1.filtrar_espacial(r_img).items():
                filas.append({"ruido": r_nombre, "filtro": f_nombre, **modulo1.metricas_calidad(gris, f_img)})
        for forma in ("ideal", "gaussiano"):
            for tipo, d0 in (("pasa_bajos", 40), ("pasa_altos", 30)):
                img, _ = modulo1.filtro_frecuencia(gris, d0, tipo, forma)
                filas.append({"ruido": "ninguno", "filtro": f"frecuencia {forma} {tipo} D0={d0}",
                              **modulo1.metricas_calidad(gris, img)})
        # for nombre, b in modulo1.detectar_bordes(gris).items():
        #     filas.append({"ruido": "ninguno", "filtro": f"continuidad borde {nombre}",
        #                   "PSNR (dB)": np.nan,
        #                   "SSIM": modulo1.continuidad_contorno_pupilar(b, mask_ws)})
        # pd.DataFrame(filas).to_csv(out_dir / "modulo1_metricas.csv", index=False)
    paso("Módulo 1", m1)

    # ---- Módulo 2
    def m2():
        _, otsu = modulo2.segmentar_otsu(gris)
        mascaras = {"Otsu": otsu, **{f"K-Means K={k}": m for k, m in modulo2.segmentar_kmeans(rgb).items()},
                    "Watershed": mask_ws}
        try:
            mascaras["Snake"] = modulo2.segmentar_snake(gris, (cx, cy))[2]
        except Exception as e:             # noqa: BLE001
            avisos.append(f"Snake: {e}")
        # modulo2.tabla_metricas(mascaras, mask_ws).to_csv(out_dir / "modulo2_segmentacion_vs_watershed.csv")
        # modulo2.tabla_metricas(modulo2.combinaciones_morfologicas(mask_ws), mask_ws).to_csv(
        #     out_dir / "modulo2_morfologia_vs_watershed.csv")
    paso("Módulo 2", m2)

    # ---- Módulo 3 (frame de referencia)
    def m3():
        roi_cuant, _ = modulo3.cuantizar_roi(rgb, gris)
        desc, _ = modulo3.descriptores_morfometricos(cont)
        mask_iris, _ = modulo3.segmentar_iris(gris, mask_ws, (desc["Centroide X_c (px)"], desc["Centroide Y_c (px)"]))
        mask_iris_ok, _ = modulo3.mascara_sin_reflejos(gris, mask_iris)
        # pd.DataFrame(desc, index=["valor"]).T.to_csv(out_dir / "modulo3_morfometria.csv")
        # sp = modulo3.estadisticas_primer_orden(roi_cuant, mask_ws)
        # si = modulo3.estadisticas_primer_orden(roi_cuant, mask_iris_ok)
        # gp, gi, _ = modulo3.textura_glcm_por_canal(roi_cuant, mask_ws, mask_iris_ok)
        # pd.concat({"pupila_primer_orden": sp, "iris_primer_orden": si,
        #            "pupila_glcm": gp, "iris_glcm": gi}, axis=0).to_csv(out_dir / "modulo3_textura.csv")
    paso("Módulo 3", m3)

    # ---- Módulo 4
    def m4():
        tabla, _ = modulo4.comparar_compresion(gris)
        # tabla.to_csv(out_dir / "modulo4_compresion.csv", index=False)
        # modulo4.efecto_jpeg_en_diametro(gris, cx, cy)[0].to_csv(out_dir / "modulo4_efecto_jpeg_diametro.csv")
    paso("Módulo 4", m4)
    return avisos


# --------------------------------------------------------------- pipeline
def ejecutar_pipeline(cfg, progreso=True, guardar=True, df_previo=None):
    """Pipeline de punta a punta para el video indicado en `cfg['video_path']`.

    Salidas (en cfg['output_dir']):
      - curva_pupila.csv            área/diámetro frame a frame
      - parametros_fisiologicos.csv latencia, amplitud, velocidades (parametro, valor, unidad)
      - curva_pupila.png, vista_previa_roi.png y, si generar_reporte, las tablas de los Módulos 1-4.
    `df_previo` permite reutilizar el resultado de `procesar_video` (evita reprocesar el video).
    Devuelve un diccionario con df, parametros, tabla_parametros, eventos, ctx y rutas.
    """
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if progreso:
        print(f"Video: {cfg['video_path']}")

    ctx = preparar_referencia(cfg)
    info = ctx["info"]
    if progreso:
        print(f"{info['ancho']}x{info['alto']} | {info['fps']:.2f} fps | {info['n_frames']} frames | "
              f"{info['duracion_s']:.1f} s")
        origen = "automático" if ctx["centro_automatico"] else "config"
        print(f"Centro pupilar inicial ({origen}): {ctx['centro_frame']}")
        det = ctx.get("deteccion")
        if det:
            print(f"  detección: confianza {det['confianza']:.0%} ({det['frames_usados']}/{det['frames_muestreados']} frames) | "
                  f"grupos candidatos (x, y, votos): {det['grupos']}")
            if det["confianza"] < 0.5:
                print("  [aviso] confianza baja: revisá vista_previa_roi.png o fijá centro_pupila_inicial en config.yaml")
        elif ctx["centro_automatico"]:
            print("  [aviso] no se detectó una pupila creíble; se usó la zona más oscura. Revisá vista_previa_roi.png")
    guardar_vista_previa(ctx, out_dir / "vista_previa_roi.png")

    df = df_previo if df_previo is not None else modulo3.procesar_video(
        cfg["video_path"], ctx["centro_frame"], cfg, progreso=progreso)
    P, ev = modulo3.calcular_parametros(df, info["fps"], cfg)
    art = modulo3.resumen_artefactos(df, info["fps"])
    extra = {"fps": (info["fps"], "Hz"), "n_frames": (info["n_frames"], "frames"),
             "duracion_s": (info["duracion_s"], "s"),
             "frames_interpolados_pct": (art["frames_interpolados_pct"], "%")}
    tabla_par = modulo3.parametros_a_tabla(P, extra)

    rutas = {}
    if guardar:
        # CSV único: una fila por frame (curva A(t), D(t)) + los parámetros fisiológicos
        # (valores constantes del registro) repetidos en cada fila.
        salida = df[COLUMNAS_CURVA].copy()
        for nombre, valor in tabla_par.set_index("parametro")["valor"].items():
            salida[nombre] = valor
        rutas["resultados"] = out_dir / ARCHIVO_SALIDA
        salida.to_csv(rutas["resultados"], index=False)
        fig = graficar_curva(df, P, ev, out_dir / "curva_pupila.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

    validacion = None
    if cfg.get("anotaciones_manuales") and Path(cfg["anotaciones_manuales"]).exists():
        validacion, _ = modulo3.validar_contra_manual(df, cfg["anotaciones_manuales"])
        if guardar:
            validacion.to_csv(out_dir / "validacion_vs_manual.csv")

    avisos = []
    if cfg.get("generar_reporte", True) and guardar:
        if progreso:
            print("Generando tablas de los Módulos 1-4...")
        avisos = generar_reporte(ctx, cfg, df, out_dir)

    if progreso:
        print(tabla_par.to_string(index=False))
        print(f"\nListo. Resultados en: {out_dir}")
    return {"df": df, "parametros": P, "tabla_parametros": tabla_par, "eventos": ev,
            "ctx": ctx, "rutas": rutas, "validacion": validacion, "avisos": avisos}
