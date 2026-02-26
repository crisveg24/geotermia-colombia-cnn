# RESUMEN DEL PROYECTO — CNN Geotermia Colombia (v2)

**Proyecto:** Sistema CNN para Identificacion de Zonas Geotermicas en Colombia
**Institucion:** Universidad de San Buenaventura - Bogota
**Fecha de inicio:** Noviembre 2025
**Ultima actualizacion:** 25 de febrero de 2026
**Rama activa:** `development`
**Repositorio:** https://github.com/crisveg24/geotermia-colombia-cnn

---

## 1. Estado General del Proyecto

| Componente | Progreso | Notas |
|-----------|----------|-------|
| Documentacion tecnica | 100% | Todos los docs en `docs/` actualizados a v2 |
| Scripts de pipeline | 100% | Descarga, augmentacion, preparacion, entrenamiento, evaluacion |
| Dataset original | 100% | 200 imagenes ASTER (111 positivas + 89 negativas) |
| Dataset augmentado | 100% | 6,200 imagenes (~31x factor de aumento) |
| Entrenamiento v2 | 100% | 22 epocas, mejor epoca 8 (val_acc 94.17%) |
| Evaluacion v2 test | 100% | Accuracy 91.45%, ROC AUC 0.983, F1 91.61% |
| Interfaz grafica | 100% | Streamlit con Folium, Plotly — metricas v2 integradas |
| Auditoria de codigo | 100% | 28 bugs corregidos (ver CHANGELOG_V2.md) |
| Sliding Window | 100% | Inferencia preserva escala real (~90m/px) |

---

## 2. Logros de la Version 2

### 2.1 Expansion del Dataset
- **200 imagenes ASTER** descargadas desde Google Earth Engine (NASA/ASTER_GED/AG100_003).
- **111 positivas** de zonas volcanicas/geotermicas de Colombia.
- **89 negativas** de zonas de control (Llanos, Amazonia, Costa Caribe, Altiplano, Choco).
- **6,200 imagenes** tras augmentacion (3,441 positivas + 2,759 negativas).
- **7 bandas**: emissivity_band10, emissivity_band11, emissivity_band12, emissivity_band13, emissivity_band14, temperature, ndvi.
- **Division con GroupShuffleSplit** (sin data leakage): Train 4,223 / Val 960 / Test 1,017.
- **Datos particionados** para FAT32: ~500 imgs por archivo .npy.

### 2.2 Auditoria y Correccion de 28 Bugs
Se realizo una auditoria exhaustiva que revelo 28 bugs en v1 (4 CRITICAL, 4 HIGH, 10 MEDIUM, 10 LOW). Todos corregidos. Los mas importantes:
- **NoData no filtrado**: Valores -9999 de ASTER distorsionaban normalizacion.
- **Doble normalizacion**: z-score + Rescaling(1/255) aplastaba la señal.
- **Data leakage**: Augmentaciones de la misma imagen en train Y test.
- **Doble L2**: kernel_regularizer + AdamW weight_decay sobre-regularizaban.

Ver detalle completo en [CHANGELOG_V2.md](CHANGELOG_V2.md).

### 2.3 Modelo CNN v2
- Arquitectura ResNet-inspired personalizada: 5,032,385 parametros.
- Input: (224, 224, 7) — 7 bandas ASTER.
- Output: clasificacion binaria (sigmoid).
- Optimizaciones: SpatialDropout2D, AdamW con CosineDecay, Label Smoothing (0.1), PR-AUC, F1Score, Class Weights.
- Sin Rescaling layer (solo z-score).
- BatchNorm en shortcuts de bloques residuales.

### 2.4 Resultados v2 en Test Set (1,017 imagenes)

| Metrica | Valor v2 | Valor v1 | Mejora |
|---------|----------|----------|--------|
| Accuracy | **91.45%** | 68.43% | +23.02 pp |
| Precision | **97.94%** | 86.32% | +11.62 pp |
| Recall | **86.05%** | 48.10% | +37.95 pp |
| F1-Score | **91.61%** | 61.77% | +29.84 pp |
| ROC AUC | **0.983** | 0.8198 | +0.163 |
| MCC | **0.837** | — | Nuevo en v2 |

**Matriz de Confusion v2 (Test: 1,017 imagenes):**
```
              Predicho Neg  Predicho Pos
Real Neg         455          10
Real Pos          77         475
```

### 2.5 Interfaz Grafica (Streamlit)
- `app.py` con 5 paginas: Inicio, Prediccion por coordenadas, Metricas, Arquitectura, Acerca de.
- Mapas interactivos con Folium, graficos con Plotly.
- Metricas v2 integradas (lectura dinamica de `evaluation_metrics.json`).
- Sliding Window para prediccion a escala real.
- Todas las deprecaciones de Streamlit 1.54 corregidas (`width="stretch"`).

---

## 3. Comparativo v1 vs v2

| Aspecto | v1 (main) | v2 (development) |
|---------|-----------|-------------------|
| Imagenes originales | 85 | 200 |
| Imagenes augmentadas | 2,635 | 6,200 |
| Bandas | 7 (con bugs) | 7 (limpias) |
| Split | train_test_split (con leakage) | GroupShuffleSplit (sin leakage) |
| Normalizacion | z-score + Rescaling(1/255) | Solo z-score |
| NoData | No filtrado (-9999) | Filtrado + interpolacion |
| Regularizacion | L2 + AdamW (doble) | Solo AdamW weight_decay |
| LR Schedule | ReduceLROnPlateau | CosineDecay |
| Test Accuracy | 68.43% | **91.45%** |
| Test Recall | 48.10% | **86.05%** |
| ROC AUC | 0.8198 | **0.983** |
| Parametros | 5,025,409 | 5,032,385 |

---

## 4. Metricas Objetivo vs Resultado

| Metrica | Objetivo minimo | Objetivo ideal | Resultado v2 | Estado |
|---------|----------------|---------------|--------------|--------|
| Accuracy | >85% | >90% | **91.45%** | Logrado |
| Precision | >80% | >85% | **97.94%** | Superado |
| Recall | >80% | >85% | **86.05%** | Logrado |
| F1-Score | >80% | >85% | **91.61%** | Superado |
| ROC AUC | >0.90 | >0.95 | **0.983** | Superado |
| MCC | >0.50 | >0.70 | **0.837** | Superado |

**Todas las metricas superan los objetivos ideales.**

---

## 5. Tecnologias

| Categoria | Herramientas |
|-----------|-------------|
| Deep Learning | TensorFlow 2.20.0, Keras 3.12.1 |
| Procesamiento | NumPy, pandas, scikit-learn, scikit-image, OpenCV, SciPy, rasterio |
| Datos geoespaciales | Google Earth Engine API, NASA ASTER GED AG100_003 |
| Visualizacion | Matplotlib, Seaborn, TensorBoard, Plotly, Folium |
| Interfaz | Streamlit 1.54.0, streamlit-folium |
| Reportes | FPDF2 |
| Control de versiones | Git, GitHub |

---

## 6. Estructura del Repositorio

```
geotermia-colombia-cnn/
|-- README.md                    # README principal
|-- app.py                       # Interfaz grafica Streamlit (~2074 lineas)
|-- config.py                    # Configuracion centralizada (rutas, bandas, hiperparametros)
|-- setup.py                     # Configuracion del entorno
|-- requirements.txt             # Dependencias Python
|-- .gitignore
|
|-- models/
|   |-- cnn_geotermia.py         # Arquitectura CNN (5,032,385 params)
|   |-- __init__.py
|   +-- saved_models/            # Modelos entrenados (.keras)
|
|-- scripts/
|   |-- download_dataset.py      # Descarga imagenes desde GEE (200 imgs, 7 bandas)
|   |-- augment_full_dataset.py  # Augmentacion del dataset (30 tecnicas)
|   |-- prepare_dataset.py       # Preparacion para entrenamiento (particionado FAT32)
|   |-- train_model.py           # Entrenamiento (part-aware generator)
|   |-- evaluate_model.py        # Evaluacion en test set (particionado)
|   |-- predict.py               # Prediccion con coordenadas
|   |-- visualize_results.py     # Visualizaciones de resultados
|   |-- visualize_architecture.py# Diagrama de arquitectura
|   +-- miniprueba/              # Pipeline de validacion (mini-dataset)
|
|-- data/                        # (o disco externo via GEOTERMIA_DATA_ROOT)
|   |-- raw/                     # 200 imagenes originales + CSVs de metadata
|   |-- augmented/               # 6,200 imagenes augmentadas
|   +-- processed/               # .npy particionados para entrenamiento
|
|-- docs/                        # Documentacion tecnica
|   |-- README.md                # Indice de documentos
|   |-- RESUMEN_PROYECTO.md      # Este documento (vision general + bitacora)
|   |-- MODELO_PREDICTIVO.md     # Documento tecnico principal de la tesis
|   |-- ANALISIS_ENTRENAMIENTO.md# Analisis de metricas por epoca
|   |-- GUIA_PASO_A_PASO.md      # Guia completa paso a paso
|   |-- CONTEXTO_GEOTERMICO.md   # Contexto cientifico/geologico
|   |-- PREDICCIONES_PRUEBA.md   # Baseline v1 + comparativa v2
|   +-- CHANGELOG_V2.md          # Auditoria: 28 bugs + mejoras implementadas
|
|-- logs/                        # Logs de TensorBoard
|-- results/                     # Metricas y figuras
+-- notebooks/                   # Notebooks de exploracion
```

---

## 7. Bitacora Cronologica del Proyecto

### Fase 1: Configuracion y Documentacion (Nov 2025)
- Documentacion tecnica inicial (`MODELO_PREDICTIVO.md`, 2700+ lineas).
- Script de visualizacion de arquitectura (`visualize_architecture.py`).
- Entorno virtual Python 3.10.11, TensorFlow 2.20.0, Keras 3.12.

### Fase 2: Adquisicion de Datos v1 (Nov 2025)
- Autenticacion Google Earth Engine (proyecto `alpine-air-469115-f0`).
- Descarga de 85 imagenes ASTER (45 positivas + 40 negativas).
- Dataset: NASA/ASTER_GED/AG100_003, 7 bandas.

### Fase 3: Augmentacion v1 (Nov 2025)
- 30 tecnicas de augmentacion (geometricas, intensidad, ruido, combinaciones).
- 85 originales → 2,635 augmentadas (~31x).

### Fase 4: Preparacion v1 (Nov 2025)
- Redimensionamiento a 224x224, normalizacion z-score, division 70/15/15.
- Splits: Train 1,843 / Val 396 / Test 396.

### Fase 5: Entrenamiento v1 (Nov 2025 - Feb 2026)
- 23 epocas en CPU (~35 min), EarlyStopping selecciono mejor epoca 8.
- Overfitting severo: train 91.75% vs val 44.70% en epoca 23.
- Test: Accuracy 68.43%, Precision 86.32%, Recall 48.10%, ROC AUC 0.8198.

### Fase 6: Evaluacion y Visualizacion v1 (Feb 18, 2026)
- 4 graficos PNG (300 DPI) + reporte PDF de 12 paginas.
- 4 predicciones de prueba (todas <50% — baseline v1).

### Fase 7: Recuperacion del Repositorio (Feb 5, 2026)
- Repo local perdido, recuperado desde GitHub.
- Validacion end-to-end con mini-dataset (20 imagenes).

### Fase 8: Optimizacion del Modelo (Feb 5, 2026)
- SpatialDropout2D, AdamW, Label Smoothing, PR-AUC, F1Score.
- Interfaz Streamlit con Folium y Plotly.

### Fase 9: Auditoria y Limpieza (Feb 9, 2026)
- 39+ archivos revisados. 6 archivos eliminados (redundantes).
- 3 docs fusionados en RESUMEN_PROYECTO.md.
- `.gitignore` actualizado.

### Fase 10: Configuracion Centralizada (Feb 9, 2026)
- `config.py` con soporte para disco externo (`GEOTERMIA_DATA_ROOT`).
- Todos los scripts actualizados para usar `config.py`.

### Fase 11: Auditoria Profunda — 28 Bugs (Feb 18, 2026)
- 4 CRITICAL: NoData, doble normalizacion, data leakage, doble L2.
- 4 HIGH: RandomContrast en z-score, doble augmentacion, ReduceLR+AdamW, excepciones silenciosas.
- 10 MEDIUM + 10 LOW corregidos. Ver CHANGELOG_V2.md.

### Fase 12: Expansion a 200 Imagenes (Feb 18-19, 2026)
- Descarga de 200 imagenes con 7 bandas (111 positivas + 89 negativas).
- Upgrade de 5 a 7 bandas (temperatura + ndvi) en TODOS los archivos.
- Augmentacion a 6,200 imagenes. Preparacion con particionado FAT32.

### Fase 13: Entrenamiento v2 (Feb 19, 2026)
- 22 epocas, mejor epoca 8 con val_acc 94.17%.
- Entrenado con part-aware generator (1 particion a la vez en RAM).
- Modelo guardado: `geotermia_cnn_custom_best.keras` (52.87 MB).

### Fase 14: Evaluacion v2 y Correccion del Modelo (Feb 19-25, 2026)
- Evaluacion en test (1,017 imagenes): **Accuracy 91.45%, ROC AUC 0.983, F1 91.61%**.
- Fix de compatibilidad Keras (`quantization_config` removido del .keras).
- Fix de deprecaciones Streamlit 1.54 (18 `use_container_width` → `width="stretch"`).
- Metricas v2 integradas en `app.py`.

### Fase 15: Actualizacion de Documentacion (Feb 25, 2026)
- Consolidacion de docs: REGISTRO_PROCESO y MEJORAS_MODELO absorbidos.
- Todos los .md actualizados a datos v2.
- Indice de docs actualizado.

---

## 8. Hardware y Entorno

| Componente | Detalle |
|-----------|---------|
| CPU | Intel i5-10300H |
| RAM | 12 GB |
| GPU | No disponible (TF 2.20.0 sin CUDA en Windows) |
| Almacenamiento | USB FAT32 15 GB (`D:\geotermia_datos`) |
| SO | Windows |
| Python | 3.10.11 |
| TensorFlow | 2.20.0 |
| Keras | 3.12.1 |
| Streamlit | 1.54.0 |

---

## 9. Equipo

**Estudiantes:**
- Cristian Camilo Vega Sanchez (Lead Developer)
- Daniel Santiago Arevalo Rubiano
- Yuliet Katerin Espitia Ayala
- Laura Sophie Rivera Martin

**Asesor Academico:**
- Prof. Yeison Eduardo Conejo Sandoval

**Institucion:**
- Universidad de San Buenaventura - Bogota
- Facultad de Ingenieria
- Programa de Ingenieria de Sistemas

---

## 10. Notas Tecnicas

### Reproducibilidad
- **Random seed:** 42 fijo en todos los scripts.
- **GroupShuffleSplit** agrupa por imagen original (sin data leakage).
- **`requirements.txt`** con versiones exactas de dependencias.

### Prevencion de Overfitting (v2)
- SpatialDropout2D en bloques convolucionales.
- EarlyStopping con patience=15.
- Data Augmentation offline (30 tecnicas).
- AdamW weight_decay (L2 desacoplado — sin kernel_regularizer).
- Label Smoothing (0.1) en la funcion de perdida.
- CosineDecay para learning rate.

### Particionado de Datos (FAT32)
- Archivos .npy divididos en partes de ~500 imagenes (~670 MB cada uno).
- Train: 9 partes, Val: 2 partes, Test: 3 partes.
- Scripts de carga con `_load_partitioned_or_single()`.

### Balance de Clases
- 111 positivas + 89 negativas → pesos de clase calculados automaticamente.
- Class weights usados en `model.fit()`.

---

**Ultima actualizacion:** 25 de febrero de 2026
**Documento fusionado de:** RESUMEN_PROYECTO.md (v1) + REGISTRO_PROCESO.md (bitacora)
