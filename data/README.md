# Data Directory

Esta carpeta contiene todos los datos del proyecto. Si se usa un disco externo,
configurar `GEOTERMIA_DATA_ROOT` para que los scripts apunten a la ruta correcta.

## Estructura

- `raw/`: Imágenes ASTER descargadas sin procesar (.tif, 7 bandas) — 2,019 en v3
- `augmented/`: Imágenes con augmentación (×10 variaciones) — 22,209 en v3
- `processed/`: Archivos .npy particionados listos para entrenamiento (~30 GB en v3)

## Disco Externo (v3)

El dataset v3 (~40 GB total) se almacena en disco externo NTFS:
```
E:\geotermia_datos\
  raw\positive\    ← 997 .tif (zonas geotérmicas)
  raw\negative\    ← 1,022 .tif (zonas control)
  augmented\       ← 22,209 .tif
  processed\       ← 45 archivos .npy
```
