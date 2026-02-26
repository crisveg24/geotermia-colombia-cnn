# CHANGELOG v2 — Auditoria, Correcciones y Resultados

**Rama:** `development`
**Fecha de inicio:** 18 de febrero de 2026
**Estado:** Completado — modelo v2 entrenado y evaluado
**Auditoria:** 28 bugs identificados, todos corregidos
**Resultado:** Accuracy 91.45%, ROC AUC 0.983

---

## Resumen Ejecutivo

La v1 del modelo (baseline en rama `main`) contenia **28 bugs** de diversa severidad detectados en una auditoria exhaustiva del codigo. La v2 los corrige todos, expande el dataset de 85 a 200 imagenes originales, y logra un rendimiento significativamente superior.

| Severidad | Cantidad | Estado |
|-----------|:--------:|--------|
| **CRITICAL** | 4 | Corregidos |
| **HIGH** | 4 | Corregidos |
| **MEDIUM** | 10 | Corregidos |
| **LOW** | 10 | Corregidos |

---

## Resultados v2 — Metricas Reales en Test Set

### Comparativo v1 vs v2

| Metrica | v1 (baseline) | v2 (actual) | Cambio |
|---------|:-------------:|:-----------:|:------:|
| Accuracy | 68.43% | **91.45%** | +23.02 pp |
| Precision | 86.32% | **97.94%** | +11.62 pp |
| Recall | 48.10% | **86.05%** | +37.95 pp |
| F1-Score | 61.77% | **91.61%** | +29.84 pp |
| ROC AUC | 0.8198 | **0.983** | +0.163 |
| MCC | — | **0.837** | Nuevo |

> **Nota:** Las metricas de v1 estaban infladas por data leakage (BUG 3). El rendimiento real de v1 era probablemente peor.

### Matriz de Confusion v2

```
              Predicho Neg  Predicho Pos
Real Neg         455          10
Real Pos          77         475
```

- **True Negative Rate:** 97.85% (455/465)
- **True Positive Rate (Recall):** 86.05% (475/552)
- **False Positive Rate:** 2.15% (10/465)
- **False Negative Rate:** 13.95% (77/552)

### Objetivos vs Resultado

| Metrica | Objetivo CHANGELOG_V2 | Resultado real | Estado |
|---------|:---------------------:|:--------------:|:------:|
| Accuracy | >75% | 91.45% | Superado |
| Precision | >80% | 97.94% | Superado |
| Recall | >65% | 86.05% | Superado |
| F1-Score | >72% | 91.61% | Superado |
| ROC AUC | >0.85 | 0.983 | Superado |
| MCC | >0.50 | 0.837 | Superado |

**Todos los objetivos fueron superados ampliamente.**

### Datos del Entrenamiento v2

| Parametro | Valor |
|-----------|-------|
| Imagenes originales | 200 (111 positivas + 89 negativas) |
| Imagenes augmentadas | 6,200 (3,441 pos + 2,759 neg) |
| Bandas | 7 (emissivity_band10-14 + temperature + ndvi) |
| Split | GroupShuffleSplit (sin data leakage) |
| Train / Val / Test | 4,223 / 960 / 1,017 |
| Epocas | 22 (EarlyStopping, mejor epoca 8) |
| Mejor val_acc | 94.17% (epoca 8) |
| Modelo | 5,032,385 parametros |
| Archivo | `geotermia_cnn_custom_best.keras` (52.87 MB) |

---

## Bugs CRITICAL (Invalidaban resultados)

### BUG 1: NoData no filtrado
**Archivos:** `prepare_dataset.py`, `augment_full_dataset.py`, `predict.py`, `app.py`

Los datos ASTER GED usan **-9999** como NoData. Ningun script lo filtraba.
- La normalizacion z-score incluia -9999 → media y std completamente distorsionadas.
- Las augmentaciones de brillo/contraste usaban `image.min()` → rango dinamico de 10,989 en vez de ~140.

**Fix:** Filtrado de NoData en todos los puntos de carga: deteccion, descarte si >50%, interpolacion con mediana por banda.

### BUG 2: Doble normalizacion (z-score + Rescaling 1/255)
**Archivo:** `models/cnn_geotermia.py`

El pipeline aplicaba z-score en prepare_dataset.py y luego `Rescaling(1./255)` en el modelo, aplastando la señal a ~[-0.008, +0.008].

**Fix:** Eliminada capa `Rescaling(1./255)`. La normalizacion z-score es la unica.

### BUG 3: Data Leakage en train/val/test split
**Archivo:** `scripts/prepare_dataset.py`

`train_test_split` se ejecutaba sobre las ~2,635 imagenes augmentadas sin agrupar por imagen original. Augmentaciones de la MISMA imagen podian caer en train Y test → metricas infladas.

**Fix:** Reemplazado `train_test_split` por `GroupShuffleSplit` que agrupa por imagen original. Verificacion automatica de que no hay grupos compartidos entre splits.

### BUG 4: Doble regularizacion L2 (kernel_regularizer + AdamW weight_decay)
**Archivo:** `models/cnn_geotermia.py`

Todos los `Conv2D` y `Dense` tenian `kernel_regularizer=l2(0.0001)` Y AdamW tenia `weight_decay=0.0001`. Doble fuerza de L2.

**Fix:** Eliminado `kernel_regularizer` de TODAS las capas. Solo se usa `weight_decay` de AdamW (L2 decoupled).

---

## Bugs HIGH (Degradaban rendimiento significativamente)

### BUG 5: RandomContrast sobre datos z-score
**Archivo:** `scripts/train_model.py`

`layers.RandomContrast(0.2)` espera datos en [0,1]. Nuestros datos estan normalizados z-score (media~0, rango [-3,3]).

**Fix:** Eliminado `RandomContrast` del pipeline de augmentation online.

### BUG 6: Doble augmentacion (offline 30x + online Keras layers)
**Archivo:** `scripts/train_model.py`

El dataset ya estaba augmentado 30x offline, y luego `train_model.py` aplicaba OTRA ronda online.

**Fix:** `use_augmentation=False` por defecto cuando se entrena con dataset ya augmentado.

### BUG 7: ReduceLROnPlateau conflicta con AdamW
**Archivo:** `scripts/train_model.py`

ReduceLROnPlateau modifica el LR externamente. Con AdamW, desbalancea la relacion gradiente/weight_decay.

**Fix:** ReduceLROnPlateau eliminado. Se usa `CosineDecay` schedule integrado en el optimizador.

### BUG 8: Excepciones silenciosas en prediccion y carga de modelo
**Archivo:** `app.py`

`except Exception` devolvia `{"ok": False}` sin loguear el error.

**Fix:** Logging completo de errores con traceback.

---

## Bugs MEDIUM (Afectaban calidad/precision)

| # | Bug | Fix |
|---|-----|-----|
| 9 | Bare `except:` en Earth Engine init | → `except Exception:` |
| 10 | Hardcoded GEE project ID en 2 archivos | → Centralizado en `config.py` |
| 11 | Augmentaciones producen float64 (doble tamaño) | → `.astype(np.float32)` |
| 12 | Todas las imagenes cargadas en RAM (~6.5 GB) | → Particionado por lotes (500 imgs/parte) |
| 13 | Distancia Euclidea en lat/lon en vez de Haversine | → Formula de Haversine |
| 14 | Curva ROC sintetica en UI | → Datos reales de evaluation_metrics.json |
| 15 | Metricas hardcodeadas en UI | → Lectura dinamica de `cargar_metricas()` |
| 16 | Keywords de geotermia incompletas en labels | → Inferencia por directorio parent |
| 17 | BatchNorm faltante en shortcut del bloque residual | → BN agregada al shortcut |
| 18 | `training=False` hardcodeado en Transfer Learning | → `training=not freeze_base` |

---

## Bugs LOW (Mejoras de calidad)

| # | Bug | Fix |
|---|-----|-----|
| 19 | Doc dice "Normalizacion: 0-1 (Rescaling)" | → "Z-score por banda" |
| 20 | "5,518 imagenes" en UI | → Valor real del dataset |
| 21 | Hardcoded `logs/` relativo a CWD | → `PROJECT_ROOT / 'logs'` |
| 22 | `config.labels_csv` apunta a ruta no usada | Informativo |
| 23 | `augment_noise` no clampeaba al rango valido | → `np.clip` |
| 24 | Dimensiones incorrectas en tabla de arquitectura | Informativo |
| 25 | Sin retry logic en descargas EE | → Retry con backoff exponencial (3 intentos) |
| 26 | R2 Score para clasificacion binaria | → MCC (Matthews Correlation Coefficient) |
| 27 | Transfer Learning usaba Adam en vez de AdamW | → Consistencia con modelo custom |
| 28 | Metricas `None` mostraban "0.0000" | → "N/A" |

---

## Archivos Modificados

| Archivo | Bugs corregidos | Cambios principales |
|---------|:---------------:|---------------------|
| `models/cnn_geotermia.py` | 2, 4, 17, 18, 27 | Sin Rescaling, sin kernel_regularizer, BN en shortcut |
| `scripts/prepare_dataset.py` | 1, 3, 12, 16 | NoData filtering, GroupShuffleSplit, particionado FAT32 |
| `scripts/augment_full_dataset.py` | 1, 11, 23 | NoData filtering, float32, clip en ruido |
| `scripts/train_model.py` | 5, 6, 7 | Sin RandomContrast, augmentation=False, CosineDecay, part-aware generator |
| `scripts/evaluate_model.py` | 26, 28 | MCC en vez de R2, partitioned data loader |
| `scripts/download_dataset.py` | 9, 10, 21, 25 | except Exception, GEE_PROJECT, retry 3x, 200 imagenes |
| `scripts/predict.py` | 1 | NoData filtering |
| `app.py` | 1, 8, 10, 13, 14, 15, 19, 20 | NoData, logging, Haversine, metricas dinamicas, Sliding Window |
| `config.py` | 10 | GEE_PROJECT, NUM_BANDS=7, 7 BAND_NAMES |

---

## Mejoras Implementadas en v2

Ademas de los 28 bugs, se implementaron las siguientes mejoras (previamente en MEJORAS_MODELO.md):

| Mejora | Estado | Impacto |
|--------|:------:|---------|
| SpatialDropout2D | Implementada | Mas efectivo para datos espaciales |
| AdamW (weight_decay=1e-4) | Implementada | L2 desacoplado, mejor generalizacion |
| Label Smoothing (0.1) | Implementada | Reduce sobreconfianza |
| PR-AUC como metrica | Implementada | Mejor monitoreo en datos desbalanceados |
| F1Score nativo | Implementada | Monitoreo directo precision-recall |
| CosineDecay LR | Implementada | Decay suave sinusoidal |
| Class Weights | Implementada | Compensa desbalance de clases |
| Sliding Window Inferencia | Implementada | Preserva escala real (~90m/px) |
| Expansion a 200 imagenes | Implementada | 2.35x mas datos originales |
| 7 bandas (temp + ndvi) | Implementada | Mas informacion espectral |
| Particionado FAT32 | Implementada | Compatible con USB, no desborda RAM |
| Part-aware training | Implementada | Carga 1 particion a la vez |

### Mejoras Futuras (opcionales)

| Mejora | Impacto esperado | Prioridad |
|--------|------------------|-----------|
| Mixup/CutMix augmentation | Medio | Baja (ya tenemos 91.45%) |
| Attention Mechanism | Medio | Baja |
| Focal Loss | Medio | Baja (recall ya es 86%) |
| Grad-CAM (interpretabilidad) | Bajo (para tesis) | Media |
| Transfer Learning (EfficientNet) | Alto | Baja (modelo custom ya es bueno) |
| Mas datos (>500 originales) | Alto | Futura version |

---

## Espacio en Disco

### Estado actual (D:\geotermia_datos — USB 15 GB, FAT32)

| Directorio | Tamaño aprox. | Detalle |
|------------|:------------:|---------|
| raw/ | ~6 MB | 200 imagenes originales (.tif) |
| augmented/ | ~1.2 GB | 6,200 imagenes (float32) |
| processed/ | ~5.9 GB | 14 archivos .npy particionados |
| **Total usado** | **~7.1 GB** | |
| **Libre** | **~7.9 GB** | |

---

## Compatibilidad

El modelo v2 **NO es compatible** con los `.npy` de v1 porque:
1. Los `.npy` de v1 se generaron sin filtrar NoData.
2. El modelo v2 no tiene `Rescaling(1./255)`.
3. El split v1 tenia data leakage.
4. v2 usa 200 imagenes (vs 85) con GroupShuffleSplit.

Se re-ejecuto todo el pipeline desde descarga.

---

*Documento actualizado: 25 de febrero de 2026*
*Rama: development*
*Autores: Cristian Camilo Vega Sanchez, Daniel Santiago Arevalo Rubiano, Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martin*
