# PSIB — Reflejo fotomotor pupilar

## Estructura

```
config.yaml            <- único archivo a editar para procesar otro video (video_path)
run_pipeline.py        <- ejecución de punta a punta desde la terminal
analisis_pupila.ipynb  <- notebook que llama a las funciones y muestra imágenes/resultados
pupila/
  config.py            carga de config.yaml
  video.py             lectura de video, detección del centro pupilar, ROI
  modulo1.py           procesamiento de imagen, ruido, bordes, Fourier, PSD, PSNR/SSIM
  modulo2.py           segmentación (Otsu, K-Means, Snakes, Watershed) y morfología
  modulo3.py           morfometría, textura, procesamiento del video, parámetros fisiológicos
  modulo4.py           Huffman, RLE, JPEG, comparación
  pipeline.py          orquesta todo y escribe los CSV
data/                  videos de entrada
resultados/            salidas (se crea sola)
```

## Uso

```bash
pip install -r requirements.txt
  # Editar video_path en config.yaml  (o usar --video)
python run_pipeline.py
python run_pipeline.py --video "data/nombre del video.mov"
#guardar video a utilizar en la carpeta data
```

El centro de la pupila se detecta automáticamente
(si fallara, fijar `centro_pupila_inicial: [x, y]` en `config.yaml`; `resultados/vista_previa_roi.png`
muestra qué se detectó).

## Salidas (en `resultados`)

- `resultados.csv`: Las columnas son frame, tiempo_s, area_px2, diametro_px, interpolado, circularidad, centro_x_px, centro_y_px, brillo_medio, seguidas de los parámetros fisiológicos (t_estimulo_s, frame_estimulo, t_inicio_constriccion_s, latencia_ms, diametro_basal_px, diametro_minimo_px, amplitud_diametro_px, area_basal_px2, area_minima_px2, amplitud_area_px2, amplitud_area_pct, t_minimo_s, velocidad_constriccion_*, duracion_constriccion_s, t_inicio_redilatacion_s, velocidad_redilatacion_*) y de fps, n_frames, duracion_s, frames_interpolados_pct. Los parámetros son constantes del registro y se repiten en cada fila. Si alguno no se puede calcular queda en NaN.
- `curva_pupila.png`

Todas las magnitudes están en píxeles (sin calibración a mm).
