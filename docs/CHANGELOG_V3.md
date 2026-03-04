# CHANGELOG v3 — Pipeline Completo: Bugs, Data Leakage, Re-arquitectura y Entrenamiento Final

**Rama:** `v3`
**Fecha:** 2-3 de marzo de 2026
**Estado:** ✅ **COMPLETADO** — Modelo V3 entrenado y evaluado con éxito
**Bugs encontrados:** 6 (5 pipeline + 1 data leakage crítico)
**Resultado final:** Accuracy **92.28%**, ROC AUC **0.9737**
**Entorno:** Python 3.12.12, TensorFlow 2.20.0, NVIDIA RTX 4070 (WSL2 + Ubuntu 22.04)

---

## Resumen Ejecutivo

La v2 del modelo reportaba Accuracy 91.45% y ROC AUC 0.983 sobre un dataset
pequeño (6,200 imágenes). Al escalar a **22,209 imágenes** (v3), se
descubrieron **5 bugs críticos** en el pipeline de entrenamiento que impedían
el aprendizaje del modelo, y un **6to bug catastrófico: 62.5% de data leakage**
entre los splits de train/val/test.

Tras corregir todos los bugs, re-partir los datos con cero leakage, y migrar
a una arquitectura de Transfer Learning (EfficientNetB0), el modelo V3 alcanzó:
- **Test Accuracy: 92.28%**
- **Test ROC AUC: 0.9737**
- **Test PR AUC: 0.9693**
- **F1-Score: 0.92** (ambas clases)

### Bugs Identificados y Corregidos

| # | Severidad | Bug | Impacto | Estado |
|---|-----------|-----|---------|--------|
| 1 | **CRITICAL** | Clases segregadas en particiones .npy | Batches de una sola clase → modelo no aprende | ✅ Corregido |
| 2 | **CRITICAL** | CosineDecay con decay_steps hardcodeado | LR llegaba a mínimo en época 12 en vez de 100 | ✅ Corregido |
| 3 | **HIGH** | Validación/test cargados completos en RAM | ~5 GB en RAM innecesarios, OOM en equipos limitados | ✅ Corregido |
| 4 | **HIGH** | Generador con global shuffle + LRU cache | 16 s/step por cache thrashing entre 31 partes | ✅ Corregido |
| 5 | **CRITICAL** | Normalización z-score per-image per-band | Destruye info absoluta entre imágenes → AUC ≈ 0.51 | ✅ Corregido |
| 6 | **CRITICAL** | **62.5% data leakage entre splits** | Augmentaciones de misma zona en train+val+test → métricas infladas | ✅ Corregido |

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

## Bug 6: Data Leakage entre Splits — 62.5% (CRITICAL)

### Problema
El descubrimiento **más importante** de v3. Las 22,209 imágenes provienen de
~2,019 tiles originales, cada uno aumentado ~11 veces (flip, rotation, noise, etc.).
El split train/val/test original se hacía por **imagen individual**, no por
**zona geográfica**. Resultado: augmentaciones del **mismo tile** aparecían
simultáneamente en train, val y test.

Diagnóstico con `_check_leakage.py`:
```
Total filenames: 22209
  Train: 15453  Val: 3414  Test: 3342

Unique zones: 407
  In train: 391  (96.1%)
  In val:   366  (89.9%)
  In test:  365  (89.7%)

LEAKAGE zones (in train ∩ val): 351 → 86.2%
LEAKAGE zones (in train ∩ test): 349 → 85.8%
LEAKAGE zones (val ∩ test): 324 → 79.6%

Leaking SAMPLES:
  Val samples from leaked zones: 2136/3414 → 62.5%
  Test samples from leaked zones: 2088/3342 → 62.5%
```

### Impacto
- El modelo V2 (accuracy 91.45%) estaba **memorizando augmentaciones**
  del mismo tile, no aprendiendo features geotérmicas reales.
- Las métricas V2 son **no confiables** porque val/test comparten zonas con train.
- Todos los modelos V3-V6 (que usaban el mismo split) también están afectados.
- Explica por qué V3-V6 con datos "limpios" de normalización solo llegaban a ~64%.

### Solución
Script `scripts/resplit_data.py` — Re-partición completa por zonas geográficas:

1. **Función `extract_group(filename)`**: Extrae la zona geográfica base
   de cada archivo, eliminando sufijos de augmentación y grid:
   ```python
   # "volcan_ruiz_aug_flip_h_grid_0_1.npy" → "volcan_ruiz"
   # Strips: _aug_*, _grid_*, _part*
   ```

2. **GroupShuffleSplit**: Split por **zona** (no por imagen), asegurando que
   TODAS las augmentaciones de una zona van al **mismo split**.

3. **Memory-efficient**: Usa `np.memmap` + escritura por partes (500 imgs/parte)
   para manejar ~29 GB sin cargar todo en RAM.

### Resultado del re-split
```
Zones: 407 total
  Train: 284 zones (69.8%) → 15,037 samples (31 parts)
  Val:    61 zones (15.0%) → 3,553 samples (8 parts)
  Test:   62 zones (15.2%) → 3,619 samples (8 parts)

Zone overlap train↔val:  0 (ZERO LEAKAGE ✅)
Zone overlap train↔test: 0 (ZERO LEAKAGE ✅)
Zone overlap val↔test:   0 (ZERO LEAKAGE ✅)
```

### Archivos creados/modificados
- `scripts/resplit_data.py` — Script de re-partición (nuevo)
- `scripts/_check_leakage.py` — Script de diagnóstico (nuevo)
- `/home/cristian/geotermia/data/processed_v2/` — Datos limpios (nuevo directorio)
- `data/processed_v2/split_info.json` — Metadata del split (nuevo)
- `data/processed_v2/band_stats.json` — Estadísticas de normalización (nuevo)

---

## Migración a GPU y WSL2

### Motivación
El entrenamiento en CPU (v3 original) era impracticable: ~16 s/step ×
482 steps × 100 epochs = **214 horas** por experimento. Se configuró un
entorno GPU completo.

### Entorno configurado
| Componente | Valor |
|------------|-------|
| GPU | NVIDIA GeForce RTX 4070, 12 GB VRAM |
| Driver | 591.74 |
| CUDA | 13.1 |
| cuDNN | 91900 |
| OS | WSL2 Ubuntu 22.04.5 LTS |
| TensorFlow | 2.20.0 (GPU + XLA) |
| Python | 3.12.12 |
| Mixed Precision | float16 (activado) |
| Velocidad | ~347 ms/step (45× más rápido que CPU) |

### Datos en filesystem nativo Linux
Los datos se copiaron al filesystem nativo de Linux para evitar penalización
de I/O del filesystem de Windows montado en WSL:
```
/home/cristian/geotermia/data/processed_v2/  ← datos limpios (zero leakage)
```

---

## Modelos Experimentales V3-V6 (Fallidos)

Antes de encontrar el data leakage, se probaron múltiples arquitecturas:

| Modelo | Arquitectura | Mejor val_acc | Problema |
|--------|-------------|:---:|---------|
| V3 (custom) | Custom ResNet 5M params | 64% | Data leakage |
| V4 | Custom CNN reducido | 56% | Data leakage |
| V5 | EfficientNetB0 (primer intento) | 62% | Data leakage + adapter pobre |
| V6 | MobileNetV2 | 58% | Data leakage |

Todos estos modelos fueron entrenados **antes** de descubrir el leakage del 62.5%.
Tras la corrección del leakage y re-split, el modelo V7 (re-nombrado V3 final)
alcanzó 92.28% en el **primer entrenamiento** con datos limpios.

---

## Modelo V3 Final — Arquitectura y Entrenamiento

### Arquitectura: EfficientNetB0 + Channel Adapter

```
Input (224, 224, 7)
  │
  ├── Conv2D 3×3 (7→16, no bias) + BatchNorm + ReLU     ← Adapter Stage 1
  ├── Conv2D 1×1 (16→3, no bias) + BatchNorm + ReLU     ← Adapter Stage 2
  │
  ├── EfficientNetB0 (ImageNet, include_top=False, pooling=avg)  ← Backbone
  │
  ├── Dense 256 + BatchNorm + ReLU + Dropout(0.5)       ← Classification Head
  ├── Dense 64  + BatchNorm + ReLU + Dropout(0.3)
  └── Dense 1   + Sigmoid                               ← Output
```

**Parámetros totales:** 4,396,112
- Phase 1 (backbone congelado): 345,863 entrenables
- Phase 2 (39 capas backbone descongeladas): 2,633,367 entrenables

### Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | EfficientNetB0 + Channel Adapter (7→16→3) |
| Parámetros totales | 4,396,112 (16.8 MB) |
| Optimizer | AdamW (weight_decay=1e-3) |
| Loss | BinaryCrossentropy (label_smoothing=0.1) |
| Batch Size | 32 |
| Mixed Precision | float16 (GPU RTX 4070) |
| Regularización | MixUp (α=0.2) + Online augmentation (flip, rot, noise) |
| **Phase 1** | 30 epochs, LR=1e-3, backbone FROZEN, patience=12 |
| **Phase 2** | 50 epochs, LR=1e-4, 39 capas backbone unfrozen, patience=15 |
| ReduceLROnPlateau | factor=0.5, patience=4, min_lr=1e-7 |
| ModelCheckpoint | Mejor val_auc (mode=max) |

### Resultados del Entrenamiento

#### Phase 1 — Backbone Congelado (30 epochs)

| Epoch | val_accuracy | val_auc | val_loss |
|:-----:|:------------:|:-------:|:--------:|
| 1 | 61.78% | 0.6680 | 0.6571 |
| 10 | 73.23% | 0.8401 | 0.5372 |
| 20 | 77.51% | 0.8820 | 0.4845 |
| **27** | **79.28%** | **0.9000** | **0.4582** |
| 30 | 78.38% | 0.8936 | 0.4633 |

Test Phase 1: accuracy=76.36%, auc=0.8987

#### Phase 2 — Fine-tuning (50 epochs, 39 capas backbone)

| Epoch | val_accuracy | val_auc | val_loss | Evento |
|:-----:|:------------:|:-------:|:--------:|--------|
| 1 | 78.44% | 0.8953 | 0.4540 | Inicio fine-tuning |
| 6 | 82.09% | 0.9203 | 0.3984 | Mejora significativa |
| 15 | 84.35% | 0.9372 | 0.3547 | — |
| 21 | 85.30% | 0.9429 | 0.3362 | — |
| 30 | 88.49% | 0.9584 | 0.2822 | — |
| 37 | 90.01% | 0.9656 | 0.2493 | — |
| 45 | 91.78% | 0.9709 | 0.2197 | — |
| 49 | 92.09% | 0.9718 | 0.2143 | — |
| **50** | **92.34%** | **0.9725** | **0.2075** | **BEST (guardado)** |

### Resultados Finales en Test Set

```
═══════════════════════════════════════════════════════════
  EVALUACIÓN FINAL — TEST SET (3,619 muestras, ZERO LEAKAGE)
═══════════════════════════════════════════════════════════

  Accuracy:     92.28%
  ROC AUC:      0.9737
  PR AUC:       0.9693
  Precision:    91.27%
  Recall:       93.17%

  Classification Report:
                  precision  recall  f1-score  support
  No Geotermico      0.93    0.91     0.92     1844
  Geotermico         0.91    0.93     0.92     1772
  ─────────────────────────────────────────────────────
  accuracy                            0.92     3616
  macro avg          0.92    0.92     0.92     3616
  weighted avg       0.92    0.92     0.92     3616

  Confusion Matrix:
          Pred 0  Pred 1
  Real 0  [1686    158]
  Real 1  [ 121   1651]

  TN=1686  FP=158  FN=121  TP=1651
═══════════════════════════════════════════════════════════
```

### Modelos Guardados

| Archivo | Descripción | Tamaño |
|---------|-------------|--------|
| `geotermia_v7_phase1_best.keras` | Mejor modelo Phase 1 (E27) | 22.9 MB |
| `geotermia_v7_phase2_best.keras` | **Mejor modelo final** (E50, val_auc=0.9725) | 57.9 MB |
| `geotermia_v7_final.keras` | Modelo al finalizar Phase 2 | 57.9 MB |
| `D:\geotermia_datos\models\geotermia_v3_best.keras` | Copia en disco externo | 57.9 MB |

---

## Datos del Dataset v3

| Parámetro | v2 | v3 (original) | v3 (post-leakage fix) |
|-----------|:--:|:--:|:--:|
| Imágenes originales | 200 | 2,019 | 2,019 |
| Imágenes totales (con augmentación) | 6,200 | 22,209 | 22,209 |
| Train / Val / Test | 4,223 / 960 / 1,017 | 15,453 / 3,414 / 3,342 | **15,037 / 3,553 / 3,619** |
| Clase 0 (sin potencial) | 2,759 | 11,242 (50.6%) | ~50% |
| Clase 1 (potencial geotérmico) | 3,441 | 10,967 (49.4%) | ~50% |
| Bandas espectrales | 7 | 7 | 7 |
| Resolución | 224×224 | 224×224 | 224×224 |
| Tamaño en disco | ~4 GB | ~29 GB | ~29 GB |
| Zonas geográficas únicas | ? | 407 | 407 |
| Data leakage | no verificado | **62.5%** | **0% ✅** |
| Split method | random | random | **GroupShuffleSplit by zone** |
| Particiones train | 9 partes | 31 partes | 31 partes |
| Particiones val | 2 partes | 7 partes | 8 partes |
| Particiones test | 2 partes | 7 partes | 8 partes |

---

## Archivos Modificados (Resumen)

### Código fuente
| Archivo | Cambios |
|---------|---------|
| `models/cnn_geotermia.py` | `build_model()` usa LR fijo, schedule movido a trainer |
| `scripts/train_model.py` | Generador v3.3, val por partes, CosineDecay dinámico |
| `scripts/train_model_v7.py` | **Nuevo**: EfficientNetB0 + adapter, two-phase training |
| `scripts/resplit_data.py` | **Nuevo**: Re-partición por zonas con GroupShuffleSplit |
| `scripts/_check_leakage.py` | **Nuevo**: Diagnóstico de data leakage entre splits |
| `scripts/prepare_dataset.py` | +`compute_global_band_stats()`, normalización global per-band |
| `scripts/predict.py` | Carga `band_stats.json`, normalización con stats globales |
| `app.py` | Actualizado para cargar modelo V3 (EfficientNetB0) |

### Modelos
| Archivo | Descripción |
|---------|-------------|
| `models/saved_models/geotermia_v7_phase2_best.keras` | Mejor modelo V3 (57.9 MB) |
| `models/saved_models/geotermia_v7_final.keras` | Modelo final V3 |
| `D:\geotermia_datos\models\geotermia_v3_best.keras` | Backup en disco externo |

### Datos (no versionados en git — .npy en .gitignore)
| Archivo | Cambios |
|---------|---------|
| `data/processed/*.npy` | Todos los 48 archivos reescritos con clases mezcladas |
| `/home/cristian/geotermia/data/processed_v2/` | **Nuevo**: Datos re-partidos con ZERO leakage |

### Documentación
| Archivo | Cambios |
|---------|---------|
| `docs/CHANGELOG_V3.md` | Este documento (actualizado con resultados finales) |
| `docs/GUIA_REPROCESAR.md` | Guía paso a paso para re-ejecutar el pipeline |
| `results/metrics/evaluation_metrics.json` | Métricas V3 actualizadas |

### Logs de entrenamiento
| Archivo | Contenido |
|---------|-----------|
| `logs/geotermia_v7_phase1.csv` | Historial Phase 1 (30 epochs) |
| `logs/geotermia_v7_phase2.csv` | Historial Phase 2 (50 epochs) |
| `logs/history_v7.json` | Historial completo (JSON) |

---

## Comparación v2 vs v3

| Métrica | v2 | v3 | Cambio |
|---------|:--:|:--:|:------:|
| Accuracy | 91.45% | **92.28%** | +0.83% |
| ROC AUC | 0.983 | **0.9737** | -0.009 |
| Precision | 97.94% | **91.27%** | -6.67% |
| Recall | 86.05% | **93.17%** | +7.12% |
| F1-Score | 91.61% | **92.21%** | +0.60% |
| MCC | 0.837 | **0.846** | +0.009 |
| Test samples | 1,017 | **3,619** | +256% |
| Data leakage | no verificado | **0%** | ✅ |
| Imágenes | 6,200 | **22,209** | +258% |
| Arquitectura | Custom CNN 5M | **EfficientNetB0 4.4M** | Transfer Learning |
| GPU | CPU | **RTX 4070** | ~45× más rápido |

**Nota:** La ROC AUC de v2 (0.983) probablemente estaba inflada por el data leakage
no verificado. La ROC AUC de v3 (0.9737) es más confiable porque se mide sobre
un test set con **cero contaminación** de datos de entrenamiento.

---

## Conclusiones

1. **El data leakage era la causa raíz de todos los problemas V3-V6**.
   Con 62.5% de contaminación, las métricas de V2 no eran confiables.

2. **Tras eliminar el leakage**, el modelo V3 (EfficientNetB0) alcanzó
   92.28% de accuracy en un test set **3.5× más grande** que V2.

3. **Transfer Learning** (EfficientNetB0 + adapter 7→16→3) superó a todas
   las arquitecturas custom probadas (ResNet, CNN reducida, MobileNetV2).

4. **El two-phase training** fue clave: Phase 1 entrena la cabeza (79% val_acc),
   Phase 2 fine-tunea el backbone y lleva a 92%.

5. **MixUp + online augmentation** proporcionaron regularización efectiva
   sin necesidad de más datos.

---

*Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín*
*Asesor: Prof. Yeison Eduardo Conejo Sandoval*
*Universidad de San Buenaventura — Bogotá*
