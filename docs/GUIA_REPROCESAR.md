# Guía para Reprocesar el Dataset (v4 — Normalización Global)

> **TL;DR:** NO necesitas volver a descargar ni volver a augmentar.
> Solo necesitas re-ejecutar `prepare_dataset.py` (con el fix de
> normalización global ya aplicado) y luego entrenar.

---

## 1. ¿Qué pasó? (El Bug #5)

### El problema

En la versión anterior de `scripts/prepare_dataset.py`, la función
`normalize_image()` aplicaba una normalización **z-score por imagen
individual por banda**:

```python
# ❌ VERSIÓN ANTERIOR (Bug #5)
for i in range(image.shape[-1]):
    band = image[:, :, i]
    mean = np.mean(band)      # ← mean de ESTA imagen, ESTA banda
    std = np.std(band)         # ← std de ESTA imagen, ESTA banda
    normalized[:, :, i] = (band - mean) / std
```

### ¿Por qué es un problema?

Cada imagen quedaba con **media ≈ 0 y desviación estándar ≈ 1 en
cada banda**. Esto **destruye toda la información absoluta** entre
imágenes:

- Una zona geotérmica con temperatura de **80°C** quedaba
  idéntica a una zona fría de **20°C** después de normalizar.
- La banda de NDVI perdía la diferencia entre vegetación
  densa (NDVI=0.8) y roca desnuda (NDVI=0.1).
- **Todas las imágenes son estadísticamente indistinguibles** →
  el modelo no puede aprender → AUC ≈ 0.51 (azar puro).

### Evidencia del diagnóstico

```
Per-image means:  min=-0.000032, max=0.000323 (todas ≈ 0)
Per-image stds:   min=0.000000, max=1.000000 (todas ≈ 1)
Cosine similarity intra-class: 0.012
Cosine similarity cross-class: 0.006
→ Clases INDISTINGUIBLES
```

### La corrección (v4)

Ahora `prepare_dataset.py` hace **2 pases**:

1. **Pase 1** (`compute_global_band_stats()`): Recorre TODOS los
   22,209 `.tif` y calcula mean/std **globales** por banda usando
   el algoritmo de Welford (online, no necesita cargar todo en RAM).

2. **Pase 2**: Normaliza cada imagen usando esos mean/std globales:

```python
# ✅ VERSIÓN CORREGIDA (v4)
for i in range(image.shape[-1]):
    normalized[:, :, i] = (
        (image[:, :, i] - global_band_means[i])
        / global_band_stds[i]
    )
```

Las estadísticas globales se guardan en `data/processed/band_stats.json`
para que `predict.py` las use en inferencia.

---

## 2. ¿Qué archivos están bien y cuáles no?

| Paso del pipeline | Script | ¿Está bien? | Acción necesaria |
|---|---|---|---|
| Descarga de GEE | `download_dataset.py` | ✅ Perfecto | Ninguna |
| Augmentación | `augment_full_dataset.py` | ✅ Perfecto | Ninguna |
| Preparación `.npy` | `prepare_dataset.py` | ❌ Bug #5 | **Re-ejecutar** |
| Entrenamiento | `train_model.py` | ✅ (ya arreglado v3) | Re-entrenar |
| Predicción | `predict.py` | ✅ (arreglado v4) | Usa band_stats.json |

### Archivos en tu disco externo

```
D:\geotermia_datos\
├── raw\                  ← 2,019 .tif originales ✅ NO TOCAR
│   ├── positive\
│   ├── negative\
│   └── labels.csv
├── augmented\            ← 22,209 .tif augmentados ✅ NO TOCAR
│   ├── positive\
│   ├── negative\
│   └── labels.csv
├── processed\            ← ❌ .npy MAL NORMALIZADOS → SE REEMPLAZAN
│   ├── X_train_part*.npy
│   ├── y_train_part*.npy
│   ├── X_val_part*.npy
│   ├── y_val_part*.npy
│   ├── X_test_part*.npy
│   ├── y_test_part*.npy
│   └── split_info.json
└── processed_backup_pre_shuffle\  ← backup antiguo, puedes borrar
```

---

## 3. Pasos para reprocesar en casa

### Requisitos previos

- Python 3.12+ instalado
- El repositorio clonado: `git clone https://github.com/crisveg24/geotermia-colombia-cnn.git`
- El disco externo **D:** con los datos (o la ruta donde estén)

### Paso 0: Actualizar el código

```powershell
cd C:\ruta\al\proyecto\geotermia-colombia-cnn
git pull origin v3
```

Verifica que `scripts/prepare_dataset.py` tiene el método
`compute_global_band_stats()` (busca esa cadena en el archivo).

### Paso 1: Crear entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si tienes GPU NVIDIA, instala `tensorflow[and-cuda]` en vez de
> `tensorflow` para acelerar el entrenamiento (~10x más rápido).

### Paso 2: Configurar ruta de datos

```powershell
# Opción A: Variable de entorno (recomendado)
$env:GEOTERMIA_DATA_ROOT = "D:\geotermia_datos"

# Opción B: Si los datos están dentro del proyecto
# (no necesitas hacer nada, usa data/ por defecto)
```

Verificar configuración:

```powershell
python -c "from config import cfg; print(cfg.summary())"
```

Debe mostrar que `augmented/` tiene ~22,209 `.tif` y que la fuente
es "env $GEOTERMIA_DATA_ROOT".

### Paso 3: Borrar los `.npy` viejos (mal normalizados)

```powershell
# ⚠️ IMPORTANTE: esto borra los .npy del disco externo
Remove-Item "D:\geotermia_datos\processed\*.npy" -Force
Remove-Item "D:\geotermia_datos\processed\split_info.json" -Force
# Si quieres también borrar la copia local:
Remove-Item ".\data\processed\*.npy" -Force
Remove-Item ".\data\processed\split_info.json" -Force
```

### Paso 4: Re-ejecutar prepare_dataset.py

```powershell
python scripts/prepare_dataset.py
```

**¿Qué va a hacer?**

1. Leer `labels.csv` del directorio augmented
2. Encontrar los 22,209 `.tif` augmentados
3. **PASE 1** (nuevo): Calcular mean/std globales por banda
   recorriendo TODOS los .tif (~15-30 min)
4. Dividir en train/val/test por GroupShuffleSplit
5. **PASE 2**: Cargar, normalizar con stats globales, guardar `.npy`
   en lotes (~30-60 min)

**Tiempo total estimado:** 45-90 minutos (dependiendo del disco).

**Output esperado:**

```
Calculando estadísticas globales por banda (22209 imágenes, 7 bandas)...
global_stats: 100%|████████████████| 22209/22209
Estadísticas globales por banda (DATASET-LEVEL):
  Band 0: mean=XX.XXXX, std=YY.YYYY   ← valores reales, NO 0 / 1
  Band 1: mean=XX.XXXX, std=YY.YYYY
  ...
  Band 6: mean=XX.XXXX, std=YY.YYYY
Estadísticas globales guardadas: .../processed/band_stats.json
```

Después de terminar, verifica:

```powershell
# Verificar que los .npy se crearon
Get-ChildItem "D:\geotermia_datos\processed\*.npy" | Measure-Object
# Debería mostrar ~48 archivos

# Verificar band_stats.json
Get-Content "D:\geotermia_datos\processed\band_stats.json"
# Debe mostrar band_means y band_stds con valores REALES (no 0/1)
```

### Paso 5: Copiar datos al SSD local (si entrenas en otra máquina)

```powershell
# Crear directorio local
New-Item -ItemType Directory -Force -Path ".\data\processed"

# Copiar los .npy nuevos
Copy-Item "D:\geotermia_datos\processed\*.npy" ".\data\processed\" -Force
Copy-Item "D:\geotermia_datos\processed\split_info.json" ".\data\processed\" -Force
Copy-Item "D:\geotermia_datos\processed\band_stats.json" ".\data\processed\" -Force
```

### Paso 6: Entrenar el modelo

```powershell
python scripts/train_model.py
```

**¿Qué esperar?**

- **Época 1-5:** Accuracy debería subir rápidamente de ~55% a ~70-80%
- **Época 5-15:** Accuracy debería estabilizarse entre ~80-90%
- **Si AUC > 0.80 en las primeras 5 épocas**, la normalización está
  funcionando correctamente.
- EarlyStopping con paciencia=15 detendrá el entrenamiento si no mejora.

**🚨 Si accuracy se queda pegada en ~50% otra vez:**

Algo salió mal. Verificar:

```powershell
python -c "
import numpy as np
X = np.load('data/processed/X_train_part0.npy')
print('Shape:', X.shape)
print('Per-image means:', X.mean(axis=(1,2)).mean(axis=0))
print('Per-image stds:', X.std(axis=(1,2)).mean(axis=0))
# Los means NO deben ser todos ~0 y los stds NO deben ser todos ~1
# Deben tener valores variados por banda
"
```

### Paso 7: Evaluar el modelo

```powershell
python scripts/evaluate_model.py
```

---

## 4. Verificación rápida post-reprocesamiento

Ejecuta este script para confirmar que la normalización es correcta:

```python
import numpy as np, json

# 1. Cargar stats globales
with open('data/processed/band_stats.json') as f:
    stats = json.load(f)
print("Band means:", stats['band_means'])
print("Band stds:", stats['band_stds'])

# 2. Cargar una partición
X = np.load('data/processed/X_train_part0.npy')
y = np.load('data/processed/y_train_part0.npy')

# 3. Verificar que las clases tienen estadísticas DIFERENTES
mask0 = y == 0
mask1 = y == 1
print(f"\nClase 0 ({mask0.sum()} imgs): mean per band = {X[mask0].mean(axis=(0,1,2)):.4f}")
print(f"Clase 1 ({mask1.sum()} imgs): mean per band = {X[mask1].mean(axis=(0,1,2)):.4f}")
# Si los means son DIFERENTES → normalización correcta ✅
# Si los means son IGUALES (~0) → bug sigue presente ❌
```

---

## 5. Resumen de tiempos estimados

| Paso | Tiempo estimado | Notas |
|---|---|---|
| Instalar dependencias | 5-10 min | Solo la primera vez |
| `prepare_dataset.py` | 45-90 min | Depende del disco (USB ≈ lento) |
| Copiar .npy al SSD | 5-10 min | Solo si entrenas en otro disco |
| `train_model.py` | 2-6 horas (CPU) | ~10 min/época × 20-50 épocas |
| `train_model.py` | 15-45 min (GPU) | Con NVIDIA + CUDA |
| `evaluate_model.py` | 2-5 min | Evaluación final |

**Total sin GPU:** ~4-8 horas  
**Total con GPU:** ~1-2 horas

---

## 6. Archivos modificados en esta versión (v4)

| Archivo | Cambio |
|---|---|
| `scripts/prepare_dataset.py` | + `compute_global_band_stats()` (Welford), `normalize_image()` usa stats globales |
| `scripts/predict.py` | Carga `band_stats.json`, normaliza con stats globales |
| `docs/GUIA_REPROCESAR.md` | Este documento |
| `docs/CHANGELOG_V3.md` | Actualizado con Bug #5 |

---

## 7. FAQ

**P: ¿Tengo que volver a descargar las imágenes de Google Earth Engine?**  
R: **NO.** Los `.tif` descargados y augmentados están perfectos. El bug
está solo en cómo se normalizaban al crear los `.npy`.

**P: ¿Puedo ejecutar prepare_dataset.py directamente sobre el disco externo?**  
R: Sí. Configura `$env:GEOTERMIA_DATA_ROOT = "D:\geotermia_datos"` y
ejecuta. Será más lento que en un SSD interno pero funciona.

**P: ¿Cuánto espacio necesito?**  
R: Los `.npy` procesados ocupan ~29 GB. Necesitas ese espacio libre en
el disco donde se guarden.

**P: ¿Qué pasa si no tengo GPU?**  
R: El entrenamiento funciona en CPU pero es ~10x más lento. Con 100
épocas y paciencia=15, espera 4-8 horas. Con GPU NVIDIA: 15-45 min.

**P: ¿Qué es `band_stats.json`?**  
R: Archivo con las medias y desviaciones estándar globales por banda,
calculadas sobre todo el dataset. Se usa en `predict.py` para normalizar
nuevas imágenes de la misma forma que el dataset de entrenamiento.
