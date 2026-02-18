# CHANGELOG v2 — Mejoras al Pipeline de Datos y Modelo

**Rama:** `development`  
**Fecha de inicio:** 18 de febrero de 2026  
**Estado:** En desarrollo — requiere reentrenamiento

---

## Resumen de Cambios

La v1 del modelo (baseline en rama `main`) tiene tres bugs críticos en el pipeline de datos que afectan significativamente el rendimiento. La v2 los corrige y sienta las bases para un modelo mucho más preciso.

---

## Bug 1: NoData no filtrado (CRÍTICO)

### Problema
Los datos ASTER GED (AG100) usan el valor **-9999** para indicar píxeles sin datos válidos (nubes, bordes, cuerpos de agua). **Ningún script del pipeline v1 filtraba estos valores.**

**Impacto:**
- La **normalización z-score** incluía los -9999 en el cálculo de media y desviación estándar
- Ejemplo: si una banda tiene valores de emisividad ~[950, 990] pero 10% de píxeles son -9999, la media baja a ~-100 y la std sube a ~3000 → la normalización produce basura
- Las **augmentaciones** de brillo y contraste usaban `image.min()` y `image.max()` → con -9999 como mínimo, toda la normalización interna se corrompía
- Esto explica los falsos negativos en las pruebas v1 (Nevado del Ruiz: 40.1%, Paipa-Iza: 29.1%)

### Solución v2
Filtrado de NoData en **todos los puntos del pipeline** donde se cargan datos ASTER:

| Archivo | Cambio |
|---------|--------|
| `scripts/prepare_dataset.py` | `load_tif_image()`: detecta NoData, descarta imágenes con >50% NoData, reemplaza NoData con mediana de valores válidos por banda |
| `scripts/augment_full_dataset.py` | `load_image()`: filtra NoData antes de aplicar augmentaciones (brillo, contraste, etc.) |
| `scripts/predict.py` | `load_tif_image()`: filtra NoData antes de preprocesar |
| `app.py` | `predecir_con_modelo_cnn()`: filtra NoData antes de calcular estadísticas y normalizar, reporta % NoData por banda |

**Estrategia de interpolación:** Reemplazo de píxeles NoData con la **mediana** de valores válidos de la misma banda. Se eligió mediana sobre media porque es más robusta a outliers.

---

## Bug 2: Doble normalización (CRÍTICO)

### Problema
El pipeline v1 aplicaba **dos normalizaciones** secuenciales:

1. **z-score** en `prepare_dataset.py` → datos guardados en `.npy` con distribución ~N(0,1)
2. **`Rescaling(1./255)`** dentro del modelo (capa de Keras) → dividía entre 255 datos que ya estaban normalizados

Resultado: valores que eran ~[-2, +2] pasaban a ser ~[-0.008, +0.008] — señal aplastada casi a cero.

La "buena noticia" era que esto era **consistente** entre entrenamiento e inferencia (ambos sufrían lo mismo), por lo que el modelo aprendió algo, pero con información severamente degradada.

### Solución v2
**Eliminada la capa `Rescaling(1./255)`** del modelo en `models/cnn_geotermia.py`:

```python
# ANTES (v1):
x = layers.Rescaling(1./255, name='rescaling')(inputs)

# DESPUÉS (v2):
x = inputs  # Los datos ya vienen normalizados por z-score
```

La normalización z-score en `prepare_dataset.py` (y `app.py` para inferencia en tiempo real) es la única normalización. Es la correcta para datos de emisividad ASTER.

### Por qué `Rescaling(1./255)` era incorrecto
- `Rescaling(1./255)` está diseñado para imágenes **RGB de 8 bits** donde los valores están en [0, 255]
- Los datos ASTER de emisividad térmica son valores de emisividad escalados (típicamente ~[850, 1000] para emisividad ×1000), **NO son imágenes RGB**
- Aplicar `1/255` a datos que ya están en z-score no tiene sentido físico ni estadístico

---

## Bug 3: Augmentación corrupta por NoData (ALTO)

### Problema
`augment_brightness()` y `augment_contrast()` en `augment_full_dataset.py` usaban:
```python
image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
```

Si `image.min()` era -9999 (NoData), el rango dinámico se expandía a ~10,989 en vez de ~140, comprimiendo todos los valores reales de emisividad a una franja estrecha ~[0.91, 1.0].

### Solución v2
NoData se filtra en `load_image()` **antes** de que cualquier augmentación se aplique. Los valores NoData se reemplazan con la mediana válida por banda.

---

## Archivos Modificados

| Archivo | Líneas cambiadas | Cambio |
|---------|:----------------:|--------|
| `models/cnn_geotermia.py` | ~5 | Eliminada capa `Rescaling(1./255)`, añadido comentario explicativo |
| `scripts/prepare_dataset.py` | ~18 | Filtrado NoData en `load_tif_image()`: detección, descarte >50%, interpolación mediana |
| `scripts/augment_full_dataset.py` | ~15 | Filtrado NoData en `load_image()`: detección e interpolación mediana |
| `scripts/predict.py` | ~12 | Filtrado NoData en `load_tif_image()`: detección, log, interpolación mediana |
| `app.py` | ~20 | Filtrado NoData en predicción: estadísticas limpias, % NoData por banda, interpolación |

---

## Impacto Esperado

### Métricas v1 (baseline — rama main)
| Métrica | Valor |
|---------|-------|
| Accuracy | 68.43% |
| Precision | 86.32% |
| Recall | 48.10% |
| F1-Score | 61.77% |
| ROC AUC | 0.8198 |

### Métricas objetivo v2 (después de reentrenar)
| Métrica | Objetivo | Razón |
|---------|----------|-------|
| Accuracy | >75% | Datos limpios permiten mejor generalización |
| Precision | >80% | Mantener baja tasa de falsos positivos |
| Recall | >65% | Mejora principal: datos limpios reducen falsos negativos |  
| F1-Score | >72% | Balance precision/recall |
| ROC AUC | >0.85 | Mejor separación de clases |

### Predicciones de prueba esperadas
| Ubicación | v1 (corrupto) | v2 (esperado) |
|-----------|:-------------:|:-------------:|
| Nevado del Ruiz | 40.1% ❌ | >60% |
| Paipa-Iza | 29.1% ❌ | >50% |
| Bogotá (control) | 26.2% ✅ | <30% |
| Chocontá | 44.0% | ~45% |

---

## Pasos para Completar v2

### Realizados ✅
- [x] Filtrar NoData (-9999) en `prepare_dataset.py`
- [x] Filtrar NoData en `augment_full_dataset.py`
- [x] Filtrar NoData en `predict.py`
- [x] Filtrar NoData en `app.py`
- [x] Eliminar `Rescaling(1./255)` del modelo

### Pendientes (requieren ejecución)
- [ ] Re-ejecutar augmentación: `python scripts/augment_full_dataset.py`
- [ ] Re-ejecutar preparación: `python scripts/prepare_dataset.py`
- [ ] Re-entrenar modelo (idealmente con GPU)
- [ ] Evaluar con `evaluate_model.py`
- [ ] Comparar métricas v1 vs v2
- [ ] Ejecutar las 4 predicciones de prueba y comparar
- [ ] Documentar resultados en `PREDICCIONES_PRUEBA_V2.md`
- [ ] Merge a main cuando las métricas mejoren

---

## Notas Técnicas

### Sobre la interpolación de NoData
Se usa **mediana** en vez de **media** porque:
- La mediana es robusta a outliers
- En datos de emisividad ASTER, los valores válidos tienen distribución relativamente simétrica
- Si una imagen tiene >50% NoData, se descarta completamente (no tiene sentido interpolar más de la mitad)

### Sobre Transfer Learning
El modelo de Transfer Learning (`build_transfer_learning_model`) usa un adaptador `Conv2D 1×1` de 5→3 canales antes de EfficientNetB0/ResNet50V2. Este adaptador **no tenía** `Rescaling` — usaba la normalización interna del modelo base. Por lo tanto, el cambio de v2 **solo afecta al modelo custom**, que es el que estamos usando actualmente.

### Compatibilidad
⚠️ **El modelo v2 NO es compatible con los `.npy` de v1** porque:
1. Los `.npy` de v1 se generaron sin filtrar NoData → datos corruptos
2. El modelo v2 no tiene `Rescaling(1./255)` → espera datos z-score puros

Se debe re-ejecutar todo el pipeline desde augmentación.

---

*Documento creado: 18 de febrero de 2026*
*Rama: development*
*Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano, Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín*
