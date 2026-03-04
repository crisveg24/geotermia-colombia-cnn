# ANALISIS DEL ENTRENAMIENTO — CNN Geotermia Colombia (v2 + v3)

**Fecha del entrenamiento v2:** 19 de febrero de 2026
**Fecha del entrenamiento v3:** 2-3 de marzo de 2026
**Autores:** Cristian Camilo Vega Sanchez, Daniel Santiago Arevalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martin
**Modelo v2:** GeotermiaCNN (Custom ResNet-inspired, 5,032,385 parametros)
**Modelo v3:** EfficientNetB0 + Channel Adapter (4,396,112 parametros)
**Estado:** Entrenamiento v2 y v3 **COMPLETADOS**

---

## RESUMEN EJECUTIVO

El entrenamiento v2 del modelo CNN se completo el 19 de febrero de 2026, con un dataset expandido (200 imagenes originales → 6,200 augmentadas) y los 28 bugs de v1 corregidos. El modelo entreno durante **22 epocas**, con EarlyStopping seleccionando la **mejor epoca 8** (val_accuracy 94.17%).

### Metricas de Evaluacion en Test Set (1,017 imagenes)
```
 Accuracy:   91.45%
 Precision:  97.94%
 Recall:     86.05%
 F1-Score:   91.61%
 ROC AUC:    0.983
 MCC:        0.837
```

### Metricas de la Mejor Epoca (Epoca 8, validacion)
```
 val_accuracy:   94.17%
 train_accuracy: (mejora continua)
```

### Configuracion del Entrenamiento v2
```
 Dataset:         6,200 imagenes (200 originales augmentadas)
 Split:           4,223 train / 960 val / 1,017 test (GroupShuffleSplit)
 Bandas:          7 (emissivity_band10-14 + temperature + ndvi)
 Batch size:      32
 Epocas maximas:  100 (detenido en 22 por EarlyStopping)
 Optimizer:       AdamW (weight_decay=1e-4) con CosineDecay
 Loss:            BinaryCrossentropy (label_smoothing=0.1)
 Metricas:        Accuracy, Precision, Recall, AUC, PR-AUC, F1Score
 Hardware:        CPU Intel i5-10300H (sin GPU)
 Particionado:    9 partes train, 2 val, 3 test (FAT32 compatible)
```

---

## COMPARATIVO v1 vs v2

| Aspecto | v1 (baseline) | v2 (actual) |
|---------|:-------------:|:-----------:|
| Imagenes originales | 85 | 200 |
| Imagenes augmentadas | 2,635 | 6,200 |
| Test images | 396 | 1,017 |
| Split method | train_test_split (leakage) | GroupShuffleSplit (sin leakage) |
| Epocas entrenadas | 23 | 22 |
| Mejor epoca | 8 (val_acc 70.96%) | 8 (val_acc 94.17%) |
| Normalizacion | z-score + Rescaling | Solo z-score |
| NoData filtering | No | Si |
| **Test Accuracy** | 68.43% | **91.45%** |
| **Test Precision** | 86.32% | **97.94%** |
| **Test Recall** | 48.10% | **86.05%** |
| **Test F1** | 61.77% | **91.61%** |
| **ROC AUC** | 0.8198 | **0.983** |
| **MCC** | -0.2673 (era R2) | **0.837** |

---

## RESULTADOS DEL TEST SET v2

### Matriz de Confusion (Test: 1,017 imagenes)

```
                    Predicho
                 Neg    Pos
Real Neg  |   455  |   10   |  -> Specificity: 97.85%
Real Pos  |    77  |  475   |  -> Recall: 86.05%
```

### Analisis de la Matriz

- **True Negatives (455):** El modelo identifica correctamente el 97.85% de las zonas sin potencial.
- **True Positives (475):** El modelo detecta el 86.05% de las zonas geotermicas reales.
- **False Positives (10):** Solo 10 zonas no-geotermicas fueron incorrectamente clasificadas como positivas. Esto da una precision de 97.94%.
- **False Negatives (77):** 77 zonas geotermicas no fueron detectadas (13.95% de los positivos reales).

### Interpretacion

El modelo v2 es **altamente preciso** (97.94% precision) y tiene **buen recall** (86.05%). El balance precision-recall se refleja en el F1-Score de 91.61%. El ROC AUC de 0.983 indica capacidad discriminativa cercana a la perfeccion.

**Mejora principal respecto a v1:** El recall paso de 48.10% a 86.05% (+37.95 pp). En v1, el modelo dejaba de detectar mas de la mitad de las zonas geotermicas. En v2, detecta el 86%.

---

## ANALISIS DE TENDENCIAS DEL ENTRENAMIENTO v2

### 1. Overfitting Controlado

A diferencia de v1 donde se observo overfitting severo (train 91.75% vs val 44.70%), en v2 el overfitting fue significativamente menor gracias a:
- Mas datos (200 vs 85 originales)
- GroupShuffleSplit (sin data leakage que inflara metricas)
- Solo z-score (sin doble normalizacion)
- CosineDecay en vez de ReduceLROnPlateau

La mejor epoca 8 alcanzo val_acc de 94.17%, indicando que el modelo generaliza bien.

### 2. Factores de Mejora

| Factor | Impacto estimado | Evidencia |
|--------|-----------------|-----------|
| **Mas datos (85→200)** | Alto | Principal causa de mejora en generalizacion |
| **NoData filtering** | Alto | Eliminacion de ruido -9999 permitio señales limpias |
| **Sin data leakage** | Medio-Alto | Metricas ahora reflejan rendimiento real |
| **Sin Rescaling doble** | Medio | La señal z-score llega intacta al modelo |
| **Sin doble L2** | Medio | Modelo no sobre-regularizado |
| **CosineDecay** | Bajo-Medio | Convergencia mas suave |

### 3. Precision vs Recall

```
v1: Precision 86.32% | Recall 48.10%  -> modelo conservador, pierde +50% positivos
v2: Precision 97.94% | Recall 86.05%  -> modelo equilibrado y altamente preciso
```

En v2, el modelo es tanto mas preciso como mas sensible. La precision subio de 86% a 98% (falsos positivos casi eliminados) y el recall subio de 48% a 86%.

---

## OBJETIVOS vs RESULTADOS

| Metrica | Objetivo minimo | Objetivo ideal | Resultado v2 | Estado |
|---------|:--------------:|:--------------:|:------------:|:------:|
| Accuracy | >85% | >90% | **91.45%** | Superado |
| Precision | >80% | >85% | **97.94%** | Superado |
| Recall | >80% | >85% | **86.05%** | Superado |
| F1-Score | >80% | >85% | **91.61%** | Superado |
| ROC AUC | >0.90 | >0.95 | **0.983** | Superado |
| MCC | >0.50 | >0.70 | **0.837** | Superado |

**Conclusion: Todas las metricas superan los objetivos ideales.**

---

## ARCHIVOS GENERADOS

### Modelo (en `models/saved_models/`)
```
geotermia_cnn_custom_best.keras   — 52.87 MB (mejor epoca 8)
```

### Metricas y Logs (en `results/metrics/` y `logs/`)
```
results/metrics/evaluation_metrics.json   — Metricas v2 de evaluacion en test
results/metrics/training_history.json     — Historial de entrenamiento v2
logs/history_custom.json                  — Historial original v2
```

### Datos del Test Set
```
Particionados en D:\geotermia_datos\processed\:
X_test_part0.npy, X_test_part1.npy, X_test_part2.npy
y_test.npy
Total: 1,017 imagenes de prueba
```

---

## ENTRENAMIENTO v3: EfficientNetB0 + Channel Adapter (GPU)

### Configuracion del Entrenamiento v3

```
 Arquitectura:    EfficientNetB0 + Channel Adapter (7→16→3 canales)
 Parametros:      4,396,112 (12.6% menos que v2)
 Dataset:         22,209 imagenes (2,019 originales, 4 paises)
 Split:           15,037 train / 3,553 val / 3,619 test (407 zonas, cero leakage)
 Bandas:          7 (emissivity_band10-14 + temperature + ndvi)
 Batch size:      32
 Hardware:        NVIDIA RTX 4070 12 GB VRAM, WSL2 Ubuntu 22.04
 Precision mixta: float16 (mixed_float16)
```

**Fase 1 — Backbone congelado (30 epocas):**
```
 Capas entrenables: 345,863 (Channel Adapter + Head)
 Optimizer:         AdamW (weight_decay=1e-4) con CosineDecay
 Learning rate:     1e-3
 MixUp:             alpha=0.2
 Label Smoothing:   0.1
 Mejor epoca:       27 (val_auc=0.9000)
```

**Fase 2 — Fine-tuning (50 epocas):**
```
 Capas descongeladas: ultimas 39 del backbone (BatchNorm congelado)
 Optimizer:           AdamW (weight_decay=1e-4) con CosineDecay
 Learning rate:       1e-4 (10x menor que Fase 1)
 MixUp:               alpha=0.2
 Mejor epoca:         50 (val_auc=0.9725)
```

### Metricas v3 en Test Set (3,619 imagenes)

```
 Accuracy:   92.28%
 Precision:  91.27%
 Recall:     93.17%
 F1-Score:   92.21%
 ROC AUC:    0.9737
 PR AUC:     0.9693
 MCC:        0.8458
```

### Matriz de Confusion v3 (Test: 3,619 imagenes)

```
                    Predicho
                 Neg      Pos
Real Neg  |  1,686  |   158   |  -> Specificity: 91.43%
Real Pos  |    121  |  1,651  |  -> Recall: 93.17%
```

### Analisis Comparativo v2 → v3

| Aspecto | v2 | v3 | Cambio |
|---------|:--:|:--:|:------:|
| Test set | 1,017 imgs (1 pais) | **3,619 imgs (4 paises)** | x3.6 |
| Accuracy | 91.45% | **92.28%** | +0.83 pp |
| Precision | 97.94% | 91.27% | -6.67 pp¹ |
| **Recall** | 86.05% | **93.17%** | **+7.12 pp** |
| F1-Score | 91.61% | **92.21%** | +0.60 pp |
| ROC AUC | 0.983 | 0.9737 | -0.009² |
| MCC | 0.837 | **0.8458** | +0.009 |
| Parametros | 5,032,385 | **4,396,112** | -12.6% |
| Hardware | CPU | **GPU RTX 4070** | — |
| Epocas | 22 | **80 (30+50)** | — |

¹ Precision menor porque el test v3 es 3.6x mas grande y geograficamente diverso.
² AUC evaluado sobre un dataset mas desafiante; sigue siendo excelente (>0.97).

**Conclusion v3:** El modelo v3 mejora significativamente el recall (+7.12 pp) con un test set 3.6 veces mayor, demostrando que Transfer Learning con EfficientNetB0 y entrenamiento en GPU produce resultados superiores y mas generalizables.

---

## NOTA HISTORICA: RESULTADOS v1

Para referencia, los resultados del entrenamiento v1 (baseline) fueron:

| Metrica | v1 | Problema |
|---------|:--:|---------|
| Accuracy | 68.43% | No alcanza objetivo |
| Precision | 86.32% | Unica metrica lograda |
| Recall | 48.10% | Pierde +50% de positivos |
| F1-Score | 61.77% | Bajo por recall |
| ROC AUC | 0.8198 | Decente pero inflado por leakage |

**Problemas principales de v1:**
- Overfitting severo (train 91.75% vs val 44.70% en epoca 23)
- Data leakage inflaba metricas artificialmente
- NoData (-9999) sin filtrar distorsionaba normalizacion
- Solo 85 imagenes originales

Estos problemas fueron documentados en detalle en [CHANGELOG_V2.md](CHANGELOG_V2.md).

---

**Ultima actualizacion:** 3 de marzo de 2026
**Estado:** Entrenamiento v2 y v3 completados — Evaluacion y analisis finalizados
