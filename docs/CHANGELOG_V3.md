# CHANGELOG v3 — Correcciones Críticas del Pipeline de Datos y Entrenamiento

**Rama:** `v3`
**Fecha:** 2 de marzo de 2026
**Estado:** Bugs corregidos — pendiente re-entrenamiento con datos corregidos
**Bugs encontrados:** 5 críticos en el pipeline de datos/entrenamiento
**Entorno:** Python 3.12.8, TensorFlow 2.20.0, CPU (sin GPU)

---

## Resumen Ejecutivo

La v2 del modelo reportaba Accuracy 91.45% y ROC AUC 0.983 sobre un dataset
pequeño (6,200 imágenes). Al escalar a **22,209 imágenes** (v3), se
descubrieron **5 bugs críticos** en el pipeline de entrenamiento que impedían
el aprendizaje del modelo. El primer intento de entrenamiento v3 (sin corregir
la segregación de clases) produjo resultados inaceptables: accuracy oscilando
45-69%, EarlyStopping en época 20. El segundo intento (tras corregir bugs 1-4)
produjo accuracy estancado en ~50% (azar puro) debido al bug de normalización.

### Bugs Identificados y Corregidos

| # | Severidad | Bug | Impacto | Estado |
|---|-----------|-----|---------|--------|
| 1 | **CRITICAL** | Clases segregadas en particiones .npy | Batches de una sola clase → modelo no aprende | ✅ Corregido |
| 2 | **CRITICAL** | CosineDecay con decay_steps hardcodeado | LR llegaba a mínimo en época 12 en vez de 100 | ✅ Corregido |
| 3 | **HIGH** | Validación/test cargados completos en RAM | ~5 GB en RAM innecesarios, OOM en equipos limitados | ✅ Corregido |
| 4 | **HIGH** | Generador con global shuffle + LRU cache | 16 s/step por cache thrashing entre 31 partes | ✅ Corregido |
| 5 | **CRITICAL** | Normalización z-score per-image per-band | Destruye info absoluta entre imágenes → AUC ≈ 0.51 | ✅ Corregido |

---

## Bug 1: Clases Segregadas en Particiones (CRITICAL)

### Problema
Los archivos `y_train.npy`, `y_val.npy`, `y_test.npy` tenían las etiquetas
**completamente segregadas por clase**:

```
y_train[0:7683]     → TODAS clase 1 (potencial geotérmico)
y_train[7684:15452]  → TODAS clase 0 (sin potencial)
```

Como cada archivo `X_train_part*.npy` contiene ~500 muestras secuenciales:
- Parts 0–14: **100% clase 1**
- Parts 16–30: **100% clase 0**
- Part 15: mezcla en el borde

El generador cargaba una parte a la vez y producía batches de **una sola clase**.
El modelo veía, por ejemplo, 15 partes seguidas de solo clase 1, luego 15 de
solo clase 0. Esto impedía el aprendizaje de gradientes útiles.

### Evidencia
Primer entrenamiento (20 épocas con EarlyStopping):
- Accuracy: osciló entre 45% y 69%, nunca estable
- val_accuracy: osciló entre 50% y 73%, sin tendencia de mejora
- F1-score: 0.0 en varias épocas (modelo predecía solo una clase)
- EarlyStopping en época 20, restauró pesos de época 5

### Solución
**Pre-shuffle one-time de datos en disco** (seed=42 reproducible):
1. Generar permutación aleatoria global para train/val/test
2. Reescribir **todos** los 45 archivos `X_*_part*.npy` con muestras mezcladas
3. Reescribir `y_train.npy`, `y_val.npy`, `y_test.npy` con la misma permutación

Resultado después del shuffle:
```
Cada partición tiene ~50% clase 0 y ~50% clase 1
  Part  0: 500 samples | class 0: 250 (50.0%) | class 1: 250 (50.0%)
  Part  1: 500 samples | class 0: 262 (52.4%) | class 1: 238 (47.6%)
  ...
  Part 30: 453 samples | class 0: 226 (49.9%) | class 1: 227 (50.1%)
```

### Archivos modificados
- `data/processed/X_train_part0-30.npy` — reescritos con mezcla de clases
- `data/processed/X_val_part0-6.npy` — reescritos con mezcla de clases
- `data/processed/X_test_part0-6.npy` — reescritos con mezcla de clases
- `data/processed/y_train.npy` — reordenado
- `data/processed/y_val.npy` — reordenado
- `data/processed/y_test.npy` — reordenado

---

## Bug 2: CosineDecay con decay_steps Hardcodeado (CRITICAL)

### Problema
En `models/cnn_geotermia.py`, el schedule de learning rate estaba hardcodeado:

```python
# ANTES (bug):
lr_schedule = CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=100 * 60,  # 6,000 steps hardcodeado
    alpha=0.0001
)
```

Con el dataset v3 (22,209 imágenes, batch_size=32):
- `steps_per_epoch = 15,453 // 32 = 482`
- `total_steps = 100 × 482 = 48,200`

Con decay_steps=6,000, el LR llegaba a su mínimo (0.0001) en la **época 12**,
dejando 88 épocas restantes con un LR prácticamente muerto.

### Solución
Mover el CosineDecay schedule de `build_model()` al método `train()` de
`train_model.py`, donde se conoce el `steps_per_epoch` real:

```python
# DESPUÉS (correcto):
# En cnn_geotermia.py → build_model():
optimizer = AdamW(learning_rate=0.001, ...)  # LR fijo inicial

# En train_model.py → train():
total_steps = self.epochs * steps_per_epoch  # 48,200
lr_schedule = get_cosine_decay_schedule(
    initial_learning_rate=cfg.LEARNING_RATE,
    decay_steps=total_steps,  # dinámico
    alpha=0.0001
)
model.optimizer.learning_rate = lr_schedule
```

### Archivos modificados
- `models/cnn_geotermia.py` — `build_model()` ahora usa LR fijo
- `scripts/train_model.py` — `train()` aplica CosineDecay dinámicamente

---

## Bug 3: Validación/Test Cargados Completos en RAM (HIGH)

### Problema
`load_data()` cargaba X_val completo en RAM (~4.7 GB para 7 partes × 670 MB).
En máquinas con poca RAM esto causaba slowdown o OOM.

### Solución
Val y test ahora usan **generadores por partes** idénticos al train generator:
- `load_data()` retorna `val_parts` (lista de rutas) en vez de `X_val` (array)
- El generador de validación carga una parte a la vez desde disco
- Test no se carga durante entrenamiento

### Archivos modificados
- `scripts/train_model.py` — `load_data()`, `create_data_generators()`

---

## Bug 4: Generador con Global Shuffle + LRU Cache (HIGH)

### Problema
El primer intento de corregir Bug 1 fue hacer shuffle global con un cache LRU
de 3 partes. Pero con 31 partes de train, el acceso aleatorio global causaba
**cache thrashing** constante: cada batch necesitaba muestras de partes
diferentes, eviccionando y recargando archivos de 670 MB continuamente.

**Resultado:** 16 segundos por step (vs 347 ms con lectura secuencial).
Inaceptable.

### Solución
Estrategia **part-sequential con intra-part shuffle** (v3.3):
1. Los datos ya están pre-shuffleados en disco (Bug 1 fix)
2. Cada época: shuffle del **orden de las partes**
3. Cargar **una parte completa** a la vez (~670 MB, lectura secuencial)
4. Shuffle de las muestras **dentro** de esa parte
5. Yield batches secuencialmente

Esto da excelente aleatoriedad estadística (clases mezcladas en cada parte +
shuffle intra-parte) con I/O óptimo (lectura secuencial, sin cache thrashing).

### Archivos modificados
- `scripts/train_model.py` — `create_data_generators()` generador v3.3

---

## Bug 5: Normalización z-score Per-Image Per-Band (CRITICAL)

### Problema
En `scripts/prepare_dataset.py`, `normalize_image()` aplicaba z-score
**por imagen individual por banda**:

```python
# ANTES (bug):
for i in range(image.shape[-1]):
    band = image[:, :, i]
    mean = np.mean(band)      # ← mean de ESTA imagen
    std = np.std(band)        # ← std de ESTA imagen
    normalized[:, :, i] = (band - mean) / std
```

Cada imagen quedaba con mean≈0 y std≈1 en **cada banda**. Esto destruye
toda la información absoluta: una zona geotérmica (80°C) y una zona fría
(20°C) quedan idénticas tras normalizar. Las bandas de emissividad, NDVI
y temperatura pierden su escala real.

### Evidencia
Diagnóstico ejecutado sobre los 22,209 .npy (sin afectar entrenamiento):

```
Per-image means:  min=-0.000032, max=0.000323  (todas ≈ 0)
Per-image stds:   min=0.000000, max=1.000000   (todas ≈ 1)
Cosine similarity intra-class (clase 1): 0.012
Cosine similarity cross-class (0 vs 1): 0.006
→ Clases INDISTINGUIBLES estadísticamente
```

Entrenamiento tras corregir bugs 1-4 (11 épocas):
- accuracy: osciló entre 49.0% y 51.4% (azar puro para binario)
- val_accuracy: 49.7% - 51.1%
- AUC: 0.505 - 0.515
- loss: 0.694 - 0.696 (≈ -ln(0.5) = 0.693)

### Solución
Normalización **dataset-global per-band** en dos pases:

1. **Pase 1** (`compute_global_band_stats()`): Recorre TODOS los 22,209
   `.tif` usando el algoritmo de Welford (online, O(1) RAM) para calcular
   mean y std **globales** de cada una de las 7 bandas.

2. **Pase 2**: Normaliza cada imagen con esas estadísticas globales:

```python
# DESPUÉS (correcto):
for i in range(image.shape[-1]):
    normalized[:, :, i] = (
        (image[:, :, i] - global_band_means[i])
        / global_band_stds[i]
    )
```

Las estadísticas se guardan en `data/processed/band_stats.json` para
que `predict.py` las use en inferencia (no tiene que recalcularlas).

### Impacto en el pipeline
- **download_dataset.py**: NO afectado (descarga está bien)
- **augment_full_dataset.py**: NO afectado (augmentación está bien)
- **prepare_dataset.py**: CORREGIDO (requiere re-ejecutar)
- **predict.py**: ACTUALIZADO (carga band_stats.json)

### Archivos modificados
- `scripts/prepare_dataset.py` — +`compute_global_band_stats()`, `normalize_image()` usa stats globales
- `scripts/predict.py` — Carga `band_stats.json`, normaliza con stats globales
- `data/processed/band_stats.json` — Nuevo: mean/std globales por banda
- `docs/GUIA_REPROCESAR.md` — Guía paso a paso para re-ejecutar el pipeline

---

## Datos del Dataset v3

| Parámetro | v2 | v3 |
|-----------|:--:|:--:|
| Imágenes originales | 200 | 2,019 |
| Imágenes totales (con augmentación) | 6,200 | 22,209 |
| Train / Val / Test | 4,223 / 960 / 1,017 | 15,453 / 3,414 / 3,342 |
| Clase 0 (sin potencial) | 2,759 | 11,242 (50.6%) |
| Clase 1 (potencial geotérmico) | 3,441 | 10,967 (49.4%) |
| Bandas espectrales | 7 | 7 |
| Resolución | 224×224 | 224×224 |
| Tamaño en disco | ~4 GB | ~29 GB |
| Particiones train | 9 partes | 31 partes |
| Particiones val | 2 partes | 7 partes |
| Particiones test | 2 partes | 7 partes |

---

## Configuración del Modelo v3

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | Custom ResNet-inspired CNN |
| Parámetros | 5,032,385 (19.2 MB) |
| Bloques residuales | 4 (64→128→256→512 filtros) |
| Pooling | GlobalAveragePooling2D |
| Dropout | SpatialDropout2D(0.15) + Dropout(0.5) |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR Schedule | CosineDecay (1e-3 → 1e-4, 48,200 steps) |
| Loss | BinaryCrossentropy (label_smoothing=0.1) |
| Batch Size | 32 |
| Max Epochs | 100 (EarlyStopping patience=15) |
| Mixed Precision | Desactivado (CPU only) |
| Augmentation online | Desactivado (pre-augmentado offline) |

---

## Archivos Modificados (Resumen)

### Código fuente
| Archivo | Cambios |
|---------|---------|
| `models/cnn_geotermia.py` | `build_model()` usa LR fijo, schedule movido a trainer |
| `scripts/train_model.py` | Generador v3.3, val por partes, CosineDecay dinámico |
| `scripts/prepare_dataset.py` | +`compute_global_band_stats()`, normalización global per-band |
| `scripts/predict.py` | Carga `band_stats.json`, normalización con stats globales |

### Datos (no versionados en git — .npy en .gitignore)
| Archivo | Cambios |
|---------|---------|
| `data/processed/*.npy` | Todos los 48 archivos reescritos con clases mezcladas |

### Documentación
| Archivo | Cambios |
|---------|---------|
| `docs/CHANGELOG_V3.md` | Este documento |
| `docs/GUIA_REPROCESAR.md` | Guía paso a paso para re-ejecutar el pipeline |

---

## Próximos Pasos

1. **Re-ejecutar** `prepare_dataset.py` con normalización global corregida
   (ver `docs/GUIA_REPROCESAR.md`)
2. **Re-entrenar** modelo con datos correctamente normalizados
3. **Evaluar** en test set y comparar con v2
4. **Documentar** resultados finales

---

*Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín*
*Asesor: Prof. Yeison Eduardo Conejo Sandoval*
*Universidad de San Buenaventura — Bogotá*
