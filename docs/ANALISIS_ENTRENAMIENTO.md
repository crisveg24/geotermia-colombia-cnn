# ANÁLISIS DEL ENTRENAMIENTO - CNN Geotermia Colombia

**Fecha del entrenamiento:** 18 de febrero de 2026  
**Autores:** Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín  
**Modelo:** GeotermiaCNN (Custom ResNet-inspired)  
**Estado:** Entrenamiento **COMPLETADO** — 23 épocas con EarlyStopping (patience=15)

---

## RESUMEN EJECUTIVO

El entrenamiento del modelo CNN se completó exitosamente el 18 de febrero de 2026, ejecutado en CPU (Intel i5-10300H) debido a que TensorFlow 2.20.0 en Windows no soporta CUDA. El modelo entrenó durante **23 épocas** (~35 minutos), con EarlyStopping deteniendo el entrenamiento cuando la val_loss dejó de mejorar. La **mejor época fue la 8** con val_accuracy de 70.96%.

### Métricas de Evaluación en Test Set (396 imágenes)
```
 Accuracy:   68.43%
 Precision:  86.32%
 Recall:     48.10%
 F1-Score:   61.77%
 ROC AUC:    0.8198
 R²:        -0.2673
```

### Métricas de la Mejor Época (Época 8, validación)
```
 val_loss:       0.7710 (mínimo alcanzado)
 val_accuracy:   70.96%
 train_loss:     0.6268
 train_accuracy: 83.40%
```

### Configuración del Entrenamiento
```
 Dataset:         2,635 imágenes (85 originales augmentadas)
 Split:           1,843 train / 396 val / 396 test
 Batch size:      32
 Épocas máximas:  100 (detenido en 23 por EarlyStopping)
 Optimizer:       Adam con ReduceLROnPlateau
 Hardware:        CPU Intel i5-10300H (sin GPU disponible)
 Tiempo/época:    ~90 segundos
 Tiempo total:    ~35 minutos
```

---

## PROGRESO DETALLADO POR ÉPOCA

### Tabla Completa de Métricas

| Época | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|-------|-----------|-----------|----------|---------|-----|
| 1 | 0.9749 | 59.09% | 0.9008 | 53.03% | 0.001000 |
| 2 | 0.8932 | 65.93% | 0.9338 | 53.03% | 0.001000 |
| 3 | 0.8459 | 69.78% | 1.0223 | 53.03% | 0.001000 |
| 4 | 0.7803 | 73.41% | 1.0352 | 53.03% | 0.001000 |
| 5 | 0.7270 | 77.05% | 1.1625 | 53.03% | 0.001000 |
| 6 | 0.6761 | 80.03% | 0.8735 | 53.28% | 0.001000 |
| 7 | 0.6552 | 81.77% | 0.8723 | 53.79% | 0.001000 |
| **8** | **0.6268** | **83.40%** | **0.7710** | **70.96%** | **0.001000** |
| 9 | 0.6106 | 83.83% | 1.5124 | 46.97% | 0.001000 |
| 10 | 0.5788 | 85.46% | 0.8155 | 69.19% | 0.001000 |
| 11 | 0.5645 | 86.33% | 1.3460 | 39.65% | 0.001000 |
| 12 | 0.5629 | 86.60% | 1.1631 | 46.97% | 0.001000 |
| 13 | 0.5562 | 85.84% | 1.6180 | 46.21% | 0.001000 |
| 14 | 0.5241 | 87.47% | 4.5237 | 46.97% | 0.001000 |
| 15 | 0.5072 | 88.93% | 1.4094 | 45.71% | 0.001000 |
| 16 | 0.5041 | 88.55% | 3.6186 | 46.97% | 0.001000 |
| 17 | 0.4898 | 89.58% | 2.5557 | 46.97% | 0.001000 |
| 18 | 0.4915 | 89.09% | 5.4512 | 46.21% | 0.001000 |
| 19 | 0.4697 | 90.23% | 4.7367 | 46.97% | 0.001000 |
| 20 | 0.4578 | 90.29% | 3.1227 | 46.21% | 0.001000 |
| 21 | 0.4475 | 91.26% | 2.2683 | 46.72% | 0.001000 |
| 22 | 0.4382 | 91.48% | 4.5883 | 45.45% | 0.001000 |
| 23 | 0.4393 | 91.75% | 4.8363 | 44.70% | 0.001000 |

> **Nota:** La época 8 (resaltada) fue seleccionada como mejor modelo por mínimo val_loss (0.7710).

---

## ANÁLISIS DE TENDENCIAS

### 1. Overfitting Severo Detectado

El hallazgo más importante del entrenamiento es la presencia de **overfitting significativo**:

```
Época 8 (mejor):  Train Acc = 83.40%  |  Val Acc = 70.96%  |  Gap = 12.44%
Época 23 (final): Train Acc = 91.75%  |  Val Acc = 44.70%  |  Gap = 47.05%
```

**Evidencia:**
- Train loss disminuyó consistentemente: 0.9749 → 0.4393 (-55%)
- Val loss pasó de un mínimo de 0.7710 (época 8) a 4.8363 (época 23) — **aumento de 527%**
- Train accuracy subió constantemente: 59.09% → 91.75%
- Val accuracy colapsó después de época 8: 70.96% → 44.70%

**Causas probables:**
- Dataset relativamente pequeño (1,843 imágenes de entrenamiento, solo 85 originales)
- Las imágenes augmentadas comparten la misma fuente, limitando diversidad real
- La capacidad del modelo (bloques residuales con 32→512 filtros) supera la complejidad del dataset

### 2. Comportamiento de la Validation Loss

La val_loss muestra alta volatilidad después de la época 8:

```
Épocas 1-8:   Disminución gradual (0.9008 → 0.7710) ✓
Épocas 9-23:  Explosión errática (1.51, 0.82, 1.35, 1.16, 1.62, 4.52, ..., 4.84) ✗
```

Esto indica que el modelo está memorizando patrones del training set que no generalizan al validation set.

### 3. Train Loss y Accuracy

El modelo aprende eficientemente del training set:

```
Fase 1 (Épocas 1-5):   Loss 0.97→0.73  Acc 59%→77%  (aprendizaje rápido)
Fase 2 (Épocas 6-10):  Loss 0.68→0.58  Acc 80%→85%  (refinamiento)
Fase 3 (Épocas 11-23): Loss 0.56→0.44  Acc 86%→92%  (memorización/overfitting)
```

### 4. Precision vs Recall (Evaluación en Test)

```
Precision: 86.32% — Alta confiabilidad en predicciones positivas
Recall:    48.10% — Solo detecta ~48% de las zonas geotérmicas reales
```

El modelo es **conservador**: cuando predice "geotérmico" suele acertar (86%), pero pierde más de la mitad de las zonas geotérmicas reales. Esto produce:
- **170 verdaderos negativos** y **101 verdaderos positivos**
- **16 falsos positivos** (bueno) y **109 falsos negativos** (problema)

### 5. ROC AUC = 0.8198

A pesar del overfitting, el ROC AUC de 0.8198 en test es un resultado **aceptable**:
- Indica que el modelo tiene capacidad real de discriminación
- Supera significativamente el azar (0.5)
- La mejor época (8) fue seleccionada correctamente por EarlyStopping

---

## ANÁLISIS TÉCNICO

### Matriz de Confusión (Test: 396 imágenes)

```
                    Predicho
                 Neg    Pos
Real Neg  |   170  |   16   |  → Specificity: 91.40%
Real Pos  |   109  |  101   |  → Recall: 48.10%
```

- **True Negative Rate:** 91.40% (excelente)
- **True Positive Rate (Recall):** 48.10% (bajo)
- **Precision:** 86.32%
- **El modelo tiende a predecir "negativo"** → sesgo conservador

### Factores que Afectan el Rendimiento

1. **Dataset limitado:** Solo 85 imágenes originales (45 positivas + 40 negativas). Aunque se augmentaron a 2,635, la diversidad real es limitada.

2. **Desbalance post-augmentación:** Puede haber más imágenes de una clase que otra, afectando el entrenamiento.

3. **Entrenamiento en CPU:** TensorFlow 2.20.0 en Windows no tiene soporte CUDA. El entrenamiento en CPU limita la capacidad de experimentar con hiperparámetros.

4. **Capacidad del modelo vs datos:** La arquitectura ResNet-inspired con 4 bloques residuales (hasta 512 filtros) tiene alta capacidad, facilitando el overfitting con datos limitados.

---

## COMPARATIVO: OBJETIVOS vs RESULTADOS REALES

| Métrica | Objetivo Mínimo | Objetivo Ideal | Resultado Real | Estado |
|---------|-----------------|----------------|----------------|--------|
| **Accuracy** | >85% | >90% | 68.43% | No alcanzado |
| **Precision** | >80% | >85% | 86.32% | **Logrado** ✓ |
| **Recall** | >80% | >85% | 48.10% | No alcanzado |
| **F1-Score** | >80% | >85% | 61.77% | No alcanzado |
| **AUC-ROC** | >0.90 | >0.95 | 0.8198 | Parcial (~91% del mín.) |

**Conclusión:** Solo la **Precision** alcanzó el objetivo mínimo. El ROC AUC está cerca pero no lo logra. Las demás métricas quedan lejos, principalmente por el bajo recall.

---

## RECOMENDACIONES PARA MEJORAR

### Prioridad Alta — Combatir Overfitting

1. **Aumentar regularización:**
   - Incrementar dropout rate (actualmente SpatialDropout2D)
   - Agregar L2 regularización a las capas Dense
   - Considerar reducir la capacidad del modelo (menos filtros)

2. **Mejorar data augmentation:**
   - Agregar técnicas: Mixup, CutMix, random erasing
   - Aumentar variabilidad en rotaciones, scales y flips
   - Considerar augmentación online (en tiempo de entrenamiento)

3. **Obtener más datos originales:**
   - Expandir la búsqueda de imágenes ASTER a más regiones
   - Incluir zonas geotérmicas de otros países andinos para transfer
   - Más zonas negativas con paisajes variados

### Prioridad Media — Mejorar Recall

4. **Ajustar threshold de clasificación:**
   - Threshold actual: 0.5 → probar 0.3 o 0.4 para aumentar recall
   - Analizar curva ROC para punto óptimo de operación
   - Trade-off: aumentar recall reducirá precision

5. **Class weights más agresivos:**
   - Dar más peso a la clase positiva para que el modelo penalice más los falsos negativos

6. **Focal Loss:**
   - Reemplazar binary crossentropy por focal loss
   - Pone más énfasis en ejemplos difíciles de clasificar

### Prioridad Baja — Arquitectura

7. **Transfer Learning:**
   - Usar EfficientNet-B0 o ResNet50 preentrenado en ImageNet
   - Fine-tuning con nuestro dataset

8. **Entrenar con GPU:**
   - Usar Google Colab, Kaggle, o una máquina con GPU
   - Permite experimentar más rápido con hiperparámetros

---

## ARCHIVOS GENERADOS

### Modelo (en `models/saved_models/`)
```
geotermia_cnn_custom_best.keras   — 57.69 MB (mejor época 8)
geotermia_cnn_custom_final.keras  — 57.69 MB (última época 23)
```

### Métricas y Logs (en `results/metrics/` y `logs/`)
```
results/metrics/evaluation_metrics.json   — Métricas de evaluación en test
results/metrics/training_history.json     — Historial de 23 épocas
results/metrics/metrics_table.csv         — Tabla de métricas en CSV
logs/history_custom.json                  — Historial original
logs/geotermia_cnn_custom_*.csv           — Logs CSV por época
```

### Visualizaciones (en `results/figures/`)
```
training_history.png     — Curvas de loss y accuracy por época
confusion_matrix.png     — Matriz de confusión del test set
roc_curve.png            — Curva ROC con AUC
metrics_comparison.png   — Barras comparativas de métricas
```

### Reporte PDF
```
results/reporte_entrenamiento_completo.pdf — Reporte de 12 páginas
```

---

**Última actualización:** 18 de febrero de 2026  
**Estado:** Entrenamiento completado — Evaluación y análisis finalizados  
**Próxima acción:** Implementar mejoras para combatir overfitting (ver Recomendaciones)
