# PSIB — Reflejo fotomotor pupilar

Pipeline reproducible (Módulos 0 a 4) para medir la respuesta pupilar a un flash a partir de un video del ojo.

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
data/raw/              videos de entrada
resultados/            salidas (se crea sola)
```

## Uso

Para elegir el ojo a mano (recomendado en tus videos), una sola vez por video:

```bash
python3 seleccionar_roi.py          # dibujás el recuadro del ojo + clic en el centro de la pupila
```

Esto guarda la selección en `config.yaml`. Después:

```bash
pip install -r requirements.txt
# 1) editar video_path en config.yaml  (o usar --video)
python run_pipeline.py
python run_pipeline.py --video "data/raw/otro_registro.mov" --out resultados_otro
```

### Detección automática de la pupila (`pupila/deteccion.py`)

No busca "lo más oscuro" (la regla, el pelo o las cejas lo son más que la pupila). Genera candidatos oscuros con
umbrales de distinto nivel y puntúa cada uno por lo que distingue a una pupila: forma redonda y compacta, tamaño
plausible, anillo claro alrededor (iris), sin tocar el borde del frame, reflejo del flash adentro y esclerótica
a ambos lados. Repite en 9 frames del video y vota. Si se ven los dos ojos, `deteccion.ojo` en `config.yaml`
(`auto | izquierdo | derecho`) elige cuál seguir. El pipeline imprime la confianza de la detección y
`resultados/vista_previa_roi.png` muestra el elegido (verde) y los otros candidatos (naranja).

No requiere intervención: el centro de la pupila se detecta automáticamente
(si fallara, fijar `centro_pupila_inicial: [x, y]` en `config.yaml`; `resultados/vista_previa_roi.png`
muestra qué se detectó).

## Salidas (en `output_dir`)

- `curva_pupila.csv`: `frame, tiempo_s, area_px2, diametro_px, interpolado, circularidad, centro_x_px, centro_y_px, brillo_medio`
- `parametros_fisiologicos.csv`: `parametro, valor, unidad` — tiempo de estímulo, latencia (ms), amplitud (área y diámetro),
  velocidad de constricción y de redilatación, etc. Los parámetros que no se pueden calcular quedan en NaN (las filas son siempre las mismas).
- `curva_pupila.png`, `vista_previa_roi.png` y tablas `modulo1_*.csv … modulo4_*.csv` (si `generar_reporte: true`).

Todas las magnitudes están en píxeles (sin calibración a mm).
