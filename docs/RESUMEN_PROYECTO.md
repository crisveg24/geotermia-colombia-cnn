# RESUMEN DEL PROYECTO — CNN Geotermia Colombia (v3)

**Proyecto:** Sistema CNN para Identificacion de Zonas Geotermicas en Colombia
**Institucion:** Universidad de San Buenaventura - Bogota
**Fecha de inicio:** Noviembre 2025
**Ultima actualizacion:** 3 de marzo de 2026
**Rama activa:** `v3`
**Repositorio:** https://github.com/crisveg24/geotermia-colombia-cnn

---

## 1. Estado General del Proyecto

| Componente | Progreso | Notas |
|-----------|----------|-------|
| Documentacion tecnica | 100% | Todos los docs en `docs/` actualizados a v3 |
| Scripts de pipeline | 100% | Descarga paralela, augmentacion, preparacion, entrenamiento, evaluacion |
| Dataset v2 (Colombia) | 100% | 200 imagenes ASTER (111 positivas + 89 negativas) |
| Dataset v3 (Region Andina) | 100% | 2,019 imagenes (997 positivas + 1,022 negativas, 4 paises) |
| Dataset augmentado v3 | 100% | 22,209 imagenes (x10 variaciones por imagen) |
| Dataset preparado v3 | 100% | Train 15,037 / Val 3,553 / Test 3,619 (407 zonas, anti-leakage) |
| Entrenamiento v2 | 100% | 22 epocas CPU, mejor epoca 8 (val_acc 94.17%) |
| Evaluacion v2 test | 100% | Accuracy 91.45%, ROC AUC 0.983, F1 91.61% |
| **Entrenamiento v3** | **100%** | **EfficientNetB0 + Channel Adapter, 2 fases, 80 epocas GPU** |
| **Evaluacion v3 test** | **100%** | **Accuracy 92.28%, ROC AUC 0.9737, F1 92.21%** |
| Interfaz grafica | 100% | Streamlit con Folium, Plotly — metricas v3 integradas |
| Auditoria de codigo | 100% | 28 bugs corregidos (ver CHANGELOG_V2.md) |
| Prediccion CLI | 100% | predict.py con resize bicubico alineado a entrenamiento |

---

## 2. Logros de la Version 2

### 2.1 Expansion del Dataset (v2 → v3: Region Andina)

**v2 (solo Colombia):**
- 200 imagenes ASTER (111 positivas + 89 negativas).
- 6,200 imagenes tras augmentacion (x30 variaciones).

**v3 (Region Andina: Colombia + Ecuador + Peru + Chile):**
- **2,019 imagenes ASTER** descargadas desde Google Earth Engine (descarga paralela, 3 hilos).
- **997 positivas** de zonas volcanicas/geotermicas de 4 paises andinos.
- **1,022 negativas** de zonas de control (llanos, costa, amazonia, etc.).
- **22,209 imagenes** tras augmentacion (x10 variaciones por imagen original).
- **7 bandas**: emissivity_band10–14, temperature, ndvi.
- **Division con GroupShuffleSplit** (anti-leakage geografico): Train 15,037 / Val 3,553 / Test 3,619.
- **407 zonas geograficas base** (cero solapamiento entre subconjuntos).
- **Datos particionados** en partes de ~500 imgs: Train 31 partes, Val 7 partes, Test 7 partes.
- **Expansion por grilla**: Cada zona base se expande en 9 tiles (center + 8 direcciones) para mayor cobertura.
- **Balance casi perfecto**: Class weights 0.9945 (pos) / 1.0055 (neg).

> Ver catalogo completo de campos geotermicos en [CAMPOS_GEOTERMICOS_REGION_ANDINA.md](CAMPOS_GEOTERMICOS_REGION_ANDINA.md).

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

### 2.4 Modelo v3: EfficientNetB0 + Channel Adapter
- **Transfer Learning** con EfficientNetB0 preentrenado en ImageNet.
- **Channel Adapter**: Conv2D(16, 3x3) + Conv2D(3, 1x1) para proyectar 7 bandas ASTER a 3 canales RGB.
- **4,396,112 parametros** (12.6% menos que v2).
- Entrenamiento en **2 fases** con GPU (RTX 4070, WSL2, Mixed Precision float16):
  - Fase 1 (30 epocas): backbone congelado, LR=1e-3.
  - Fase 2 (50 epocas): fine-tuning ultimas 39 capas, LR=1e-4.
- Regularizacion: MixUp (α=0.2), Label Smoothing (0.1), AdamW, Dropout(0.3).
- Script: `train_model_v7.py`.

### 2.5 Resultados v2 en Test Set (1,017 imagenes)

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

### 2.6 Resultados v3 en Test Set (3,619 imagenes)

| Metrica | Valor v3 | Valor v2 | Cambio |
|---------|----------|----------|--------|
| Accuracy | **92.28%** | 91.45% | +0.83 pp |
| Precision | **91.27%** | 97.94% | -6.67 pp¹ |
| Recall | **93.17%** | 86.05% | +7.12 pp |
| F1-Score | **92.21%** | 91.61% | +0.60 pp |
| ROC AUC | **0.9737** | 0.983 | -0.009² |
| MCC | **0.8458** | 0.837 | +0.009 |

¹ Precision menor por test set 3.6x mayor y mas diverso (4 paises). ² AUC sobre dataset mas desafiante.

**Matriz de Confusion v3 (Test: 3,619 imagenes):**
```
              Predicho Neg  Predicho Pos
Real Neg        1,686         158
Real Pos          121       1,651
```

### 2.7 Interfaz Grafica (Streamlit)
- `app.py` con 5 paginas: Inicio, Prediccion por coordenadas, Metricas, Arquitectura, Acerca de.
- Mapas interactivos con Folium, graficos con Plotly.
- Metricas v2 integradas (lectura dinamica de `evaluation_metrics.json`).
- Mapa de calor (heatmap) con FeatureGroups togglables y LayerControl.
- Seleccion de zona por clic en mapa con sincronizacion de coordenadas.
- Todas las deprecaciones de Streamlit 1.54 corregidas (`width="stretch"`).

---

## 3. Comparativo v1 vs v2 vs v3

| Aspecto | v1 (main) | v2 (development) | v3 (v3) |
|---------|-----------|-------------------|----------|
| Imagenes originales | 85 | 200 | **2,019** |
| Imagenes augmentadas | 2,635 | 6,200 | **22,209** |
| Augmentaciones/img | ~30 | ~30 | **10** |
| Paises | Colombia | Colombia | **4 (CO+EC+PE+CL)** |
| Bandas | 7 (con bugs) | 7 (limpias) | 7 (limpias) |
| Split | train_test_split (con leakage) | GroupShuffleSplit (sin leakage) | GroupShuffleSplit (anti-leakage) |
| Normalizacion | z-score + Rescaling(1/255) | Solo z-score | Solo z-score |
| NoData | No filtrado (-9999) | Filtrado + interpolacion | Filtrado + interpolacion |
| Regularizacion | L2 + AdamW (doble) | Solo AdamW weight_decay | Solo AdamW weight_decay |
| LR Schedule | ReduceLROnPlateau | CosineDecay | CosineDecay |
| Disco | USB FAT32 15 GB | USB FAT32 15 GB | **Disco externo NTFS 931 GB** |
| Descarga | Secuencial | Secuencial | **Paralela (3 hilos)** |
| Test Accuracy | 68.43% | **91.45%** | **92.28%** |
| Test Recall | 48.10% | **86.05%** | **93.17%** |
| ROC AUC | 0.8198 | **0.983** | **0.9737** |
| Parametros | 5,025,409 | 5,032,385 | **4,396,112** |
| Arquitectura | CNN custom | ResNet-inspired | **EfficientNetB0 + Adapter** |
| Entrenamiento | CPU | CPU | **GPU RTX 4070 (WSL2)** |
| Epocas | 23 | 22 | **80 (30+50, 2 fases)** |

---

## 4. Metricas Objetivo vs Resultado

| Metrica | Objetivo minimo | Objetivo ideal | Resultado v2 | **Resultado v3** | Estado |
|---------|----------------|---------------|--------------|-----------------|--------|
| Accuracy | >85% | >90% | 91.45% | **92.28%** | ✅ Superado |
| Precision | >80% | >85% | 97.94% | **91.27%** | ✅ Superado |
| Recall | >80% | >85% | 86.05% | **93.17%** | ✅ Superado |
| F1-Score | >80% | >85% | 91.61% | **92.21%** | ✅ Superado |
| ROC AUC | >0.90 | >0.95 | 0.983 | **0.9737** | ✅ Superado |
| MCC | >0.50 | >0.70 | 0.837 | **0.8458** | ✅ Superado |

**Todas las metricas de la v3 superan los objetivos ideales, validadas sobre un test set 3.6x mayor y geograficamente mas diverso.**

---

## 5. Tecnologias

| Categoria | Herramientas |
|-----------|-------------|
| Deep Learning | TensorFlow 2.20.0, Keras 3.12.1 |
| GPU / Aceleracion | NVIDIA RTX 4070 12 GB VRAM, CUDA 12.x, cuDNN, Mixed Precision float16 |
| Procesamiento | NumPy, pandas, scikit-learn, scikit-image, OpenCV, SciPy, rasterio |
| Datos geoespaciales | Google Earth Engine API, NASA ASTER GED AG100_003 |
| Visualizacion | Matplotlib, Seaborn, TensorBoard, Plotly, Folium |
| Interfaz | Streamlit 1.54.0, streamlit-folium |
| Reportes | FPDF2 |
| Control de versiones | Git, GitHub |
| Lenguaje | Python 3.12.12 (WSL2), Python 3.10.11 (Windows) |
| SO | Windows 11 (desarrollo), WSL2 Ubuntu 22.04 (entrenamiento GPU) |

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
|   |-- cnn_geotermia.py         # Arquitectura CNN v2 (5M params) + v3 EfficientNetB0 (4.4M params)
|   |-- __init__.py
|   +-- saved_models/            # Modelos entrenados (.keras)
|
|-- scripts/
|   |-- download_dataset.py      # Descarga imagenes desde GEE (2,019 imgs, 3 hilos)
|   |-- augment_full_dataset.py  # Augmentacion del dataset (10 variaciones/img)
|   |-- prepare_dataset.py       # Preparacion con anti-leakage geografico
|   |-- train_model.py           # Entrenamiento v2 (part-aware generator, CPU)
|   |-- train_model_v7.py       # Entrenamiento v3 (EfficientNetB0, 2 fases, GPU)
|   |-- evaluate_model.py        # Evaluacion en test set (particionado)
|   |-- predict.py               # Prediccion con coordenadas
|   |-- visualize_results.py     # Visualizaciones de resultados
|   |-- visualize_architecture.py# Diagrama de arquitectura
|   +-- miniprueba/              # Pipeline de validacion (mini-dataset)
|
|-- data/                        # (o disco externo via GEOTERMIA_DATA_ROOT)
|   |-- raw/                     # 2,019 imagenes originales + CSVs de metadata
|   |-- augmented/               # 22,209 imagenes augmentadas
|   +-- processed/               # .npy particionados para entrenamiento
|
|-- docs/                        # Documentacion tecnica
|   |-- README.md                # Indice de documentos
|   |-- RESUMEN_PROYECTO.md      # Este documento (vision general + bitacora)
|   |-- MODELO_PREDICTIVO.md     # Documento tecnico principal de la tesis
|   |-- CAMPOS_GEOTERMICOS_REGION_ANDINA.md # Catalogo geotermico (4 paises)
|   |-- CONTEXTO_GEOTERMICO.md   # Contexto cientifico/geologico
|   |-- ANALISIS_ENTRENAMIENTO.md# Analisis de metricas por epoca
|   |-- GUIA_PASO_A_PASO.md      # Guia completa paso a paso
|   |-- PREDICCIONES_PRUEBA.md   # Baseline v1 + comparativa v2
|   +-- CHANGELOG_V2.md          # Auditoria: 28 bugs + mejoras implementadas
|   +-- CHANGELOG_V3.md          # Cambios v3: EfficientNetB0, GPU, anti-leakage
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

### Fase 16: Expansion a Region Andina — v3 (Feb 26-27, 2026)
- **Branch `v3`** creada desde development.
- Expansion del dataset de Colombia (200 imgs) a Region Andina (2,019 imgs, 4 paises).
- Catalogo completo de campos geotermicos: `CAMPOS_GEOTERMICOS_REGION_ANDINA.md`.
- Descarga paralela desde GEE: 3 hilos concurrentes, 0.5s de delay.
- Expansion por grilla: cada zona genera 9 tiles (center + 8 direcciones).
- `NUM_AUGMENTATIONS` ajustado de 30 a 10 (optimo para 2,019 imagenes base).
- Augmentacion: 2,019 → 22,209 imagenes en 40.90 minutos.
- Preparacion con **anti-leakage geografico** (GroupShuffleSplit): 407 zonas base.
- Splits (tras resplit): Train 15,037 / Val 3,553 / Test 3,619 (cero solapamiento).
- Almacenamiento migrado de USB FAT32 15 GB a disco externo NTFS 931 GB.
- Documentacion completa actualizada a v3 con referencias academicas.
- **Siguiente paso**: Entrenamiento v3 y evaluacion.

### Fase 17: Entrenamiento y Evaluacion v3 — EfficientNetB0 (Mar 2-3, 2026)
- Correccion de **fuga de datos del 62.5%** en divisiones originales: `resplit_data.py` con GroupShuffleSplit por zona geografica base (407 zonas, cero solapamiento).
- Configuracion de **WSL2 Ubuntu 22.04** con GPU **NVIDIA RTX 4070** (CUDA 12.x, cuDNN).
- Implementacion de modelo v3: **EfficientNetB0 + Channel Adapter** (7→16→3 canales), 4,396,112 parametros.
- Entrenamiento en **dos fases** con Mixed Precision float16:
  - Fase 1 (30 epocas): backbone congelado, LR=1e-3, mejor E27 val_auc=0.9000.
  - Fase 2 (50 epocas): fine-tuning ultimas 39 capas, LR=1e-4, mejor E50 val_auc=0.9725.
- Regularizacion: **MixUp** (α=0.2), Label Smoothing (0.1), AdamW weight_decay.
- Evaluacion en test set (3,619 imagenes, 4 paises):
  - **Accuracy: 92.28%**, Precision: 91.27%, **Recall: 93.17%**, F1: 92.21%.
  - **ROC AUC: 0.9737**, PR AUC: 0.9693, **MCC: 0.8458**.
  - Confusion Matrix: [[1686,158],[121,1651]].
- Modelo guardado: `geotermia_v3_efficientnet_best.keras`.
- Actualizacion de `app.py` para cargar modelo v3.
- Documentacion completa actualizada (TESIS_CONTENIDO_APA.md, CHANGELOG_V3.md, etc.).

---

## 8. Hardware y Entorno

| Componente | Detalle |
|-----------|--------|
| CPU | Intel i5-10300H |
| RAM | 12 GB |
| **GPU** | **NVIDIA RTX 4070 12 GB VRAM** |
| Almacenamiento v2 | USB FAT32 15 GB (`D:\geotermia_datos`) |
| Almacenamiento v3 | Disco externo NTFS Toshiba 931 GB (`D:\geotermia_datos`) |
| SO (desarrollo) | Windows 11 |
| **SO (entrenamiento)** | **WSL2 Ubuntu 22.04** |
| Python | 3.12.12 (WSL2), 3.10.11 (Windows) |
| TensorFlow | 2.20.0 (con soporte GPU en WSL2) |
| Keras | 3.12.1 |
| Streamlit | 1.54.0 |
| Precision mixta | float16 (politica mixed_float16) |

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
- **GroupShuffleSplit** agrupa por zona geografica base (sin data leakage).
  - En v3: strips de sufijos de augmentacion Y grilla (_center, _N, _S, etc.).
- **`requirements.txt`** con versiones exactas de dependencias.

### Prevencion de Overfitting (v2/v3)
- SpatialDropout2D en bloques convolucionales.
- EarlyStopping con patience=15.
- Data Augmentation offline (10 variaciones en v3, 30 en v2).
- AdamW weight_decay (L2 desacoplado — sin kernel_regularizer).
- Label Smoothing (0.1) en la funcion de perdida.
- CosineDecay para learning rate.

### Particionado de Datos
- Archivos .npy divididos en partes de ~500 imagenes.
- v2: Train 9 partes, Val 2 partes, Test 3 partes (FAT32).
- **v3: Train 31 partes, Val 7 partes, Test 7 partes (NTFS)**.
- Scripts de carga con `_load_partitioned_or_single()`.

### Balance de Clases
- v2: 111 positivas + 89 negativas.
- **v3: 997 positivas + 1,022 negativas → 22,209 augmentadas**.
- Class weights: 0.9945 / 1.0055 (balance casi perfecto en v3).
- Class weights usados en `model.fit()`.

---

**Ultima actualizacion:** 3 de marzo de 2026
**Documento fusionado de:** RESUMEN_PROYECTO.md (v1) + REGISTRO_PROCESO.md (bitacora)
