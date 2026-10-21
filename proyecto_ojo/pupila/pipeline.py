"""Orquestación: ejecuta el pipeline completo sobre un video y escribe las salidas estandarizadas."""
from pathlib import Path
import numpy as np
import pandas as pd

from .video import preparar_referencia, guardar_vista_previa
from . import modulo1, modulo2, modulo3, modulo4

COLUMNAS_CURVA = ["frame", "tiempo_s", "area_px2", "diametro_px", "interpolado",
                  "circularidad", "centro_x_px", "centro_y_px"]

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

# --------------------------------------------------------------- pipeline
def ejecutar_pipeline(cfg, progreso=True, guardar=True, df_previo=None):
    """Pipeline para el video indicado en `cfg['video_path']`.
    Devuelve un diccionario con df, parametros, eventos, ctx y rutas.
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

    if progreso:
        print(tabla_par.to_string(index=False))
        print(f"\nListo. Resultados en: {out_dir}")
    return {"df": df, "parametros": P, "tabla_parametros": tabla_par, "eventos": ev,
            "ctx": ctx, "rutas": rutas, "validacion": validacion}
