# RESUMEN DEL PROYECTO - CNN Geotermia Colombia

**Proyecto:** Sistema CNN para Identificación de Zonas Geotérmicas en Colombia 
**Institución:** Universidad de San Buenaventura - Bogotá 
**Fecha de inicio:** Noviembre 2025 
**Última actualización:** 18 de febrero de 2026 
**Repositorio:** https://github.com/crisveg24/geotermia-colombia-cnn

---

## 1. Estado General del Proyecto

| Componente | Progreso | Notas |
|-----------|----------|-------|
| Documentación técnica | 100% | `MODELO_PREDICTIVO.md` (~1,294 líneas) |
| Scripts de pipeline | 100% | Descarga, augmentación, preparación, entrenamiento, evaluación |
| Dataset original | 100% | 85 imágenes ASTER descargadas desde GEE |
| Dataset augmentado | 100% | 2,635 imágenes (~31x factor de aumento) |
| Entrenamiento completo | 100% | 23 épocas en CPU (EarlyStopping, mejor época 8) |
| Interfaz gráfica | 100% | Streamlit con Folium, Plotly |
| Optimizaciones del modelo | 100% | SpatialDropout2D, AdamW, Label Smoothing, PR-AUC, F1Score |
| Evaluación final | 100% | Test: Accuracy 68.43%, ROC AUC 0.8198 |
| Visualizaciones | 100% | 4 PNGs + Reporte PDF de 12 páginas |

---

## 2. Logros Completados

### 2.1 Documentación Técnica
- **`MODELO_PREDICTIVO.md`** — Fundamentos teóricos de CNNs, arquitectura detallada (52 capas), pipeline de procesamiento, métricas con ecuaciones LaTeX, 11 referencias académicas.
- **`REGISTRO_PROCESO.md`** — Cronograma de 9 fases con estadísticas completas.
- **`ANALISIS_ENTRENAMIENTO.md`** — Tabla época-por-época (23 épocas), análisis de overfitting, recomendaciones.
- **`MEJORAS_MODELO.md`** — Roadmap de optimizaciones aplicadas y futuras.
- **`GUIA_PASO_A_PASO.md`** — Guía completa para reproducir el pipeline (incluye sección para máquinas con GPU).

### 2.2 Adquisición y Procesamiento de Datos
- **85 imágenes ASTER** descargadas desde Google Earth Engine (NASA/ASTER_GED/AG100_003).
- **45 positivas** de 9 zonas volcánicas/geotérmicas (Nevado del Ruiz, Puracé, Galeras, Paipa, Tolima, Cumbal, Sotará, Azufral, termales).
- **40 negativas** de 5 zonas de control (Llanos, Amazonas, Costa Caribe, Zona Andina Oriental, Chocó).
- **2,635 imágenes** tras augmentación con 30 transformaciones (geométricas, intensidad, ruido, combinaciones).
- **División estratificada:** 1,843 train (70%) / 396 val (15%) / 396 test (15%).

### 2.3 Modelo CNN
- Arquitectura ResNet-inspired personalizada: 52 capas, 5,025,409 parámetros.
- Input: (224, 224, 7) — 7 bandas ASTER (5 térmicas + temp + ndvi).
- Output: clasificación binaria (sigmoid).
- Optimizado con SpatialDropout2D, AdamW, Label Smoothing (0.1), PR-AUC, F1Score.

### 2.4 Interfaz Gráfica (Streamlit)
- **`app.py`** con 5 páginas: Inicio, Predicción por coordenadas, Métricas, Arquitectura, Acerca de.
- Mapas interactivos con Folium, gráficos con Plotly.

---

## 3. Problemas Resueltos

| Problema | Causa | Solución |
|----------|-------|----------|
| `prepare_dataset.py` no encontraba imágenes | Rutas por defecto apuntaban a `data/raw` en vez de `data/augmented` | Actualización de parámetros en `main()` |
| ValueError: "inhomogeneous shape" al crear array | Imágenes con 3-5 bandas (augmentación generó RGB en algunos casos) | Normalización automática de bandas en `load_tif_image()` |
| `train_model.py` no encontraba archivos desde otra carpeta | Uso de rutas relativas | Cambio a rutas absolutas con `Path(__file__).parent.parent` |

---

## 4. Historial de Commits Clave

```
33343c8 - "docs: Agregar documento técnico completo del modelo predictivo CNN"
 → MODELO_PREDICTIVO.md

71b4627 - "feat: Agregar script de visualización de arquitectura CNN"
 → scripts/visualize_architecture.py

1aa8334 - "docs: Agregar documentación completa del proceso de desarrollo"
 → REGISTRO_PROCESO.md, MONITOREO, RESUMEN, scripts corregidos

f8692e0 - "docs: Actualizar documentación con análisis de 30 épocas"
 → ANALISIS_ENTRENAMIENTO.md, actualizaciones de métricas

e39c698 - "feat: Agregar scripts y guía para entrenamiento externo"
 → ENTRENAMIENTO_EXTERNO.md, 6 scripts, 3 CSVs metadata

7660081 - "feat: Optimizaciones del modelo y nueva interfaz Streamlit"
 → SpatialDropout2D, AdamW, Label Smoothing, app.py
```

---

## 5. Guía de Monitoreo del Entrenamiento

### 5.1 Opciones de Monitoreo

**Terminal (salida directa):**
```
Epoch 1/23
58/58 [==============================] - 90s - loss: 0.8912 - accuracy: 0.6534 - val_loss: 0.8210 - val_accuracy: 0.6515
```

**TensorBoard (recomendado):**
```bash
tensorboard --logdir=logs
# Abrir navegador en: http://localhost:6006
```

**CSV de historial:**
```python
import pandas as pd
df = pd.read_csv('logs/geotermia_cnn_custom_20260218-120941.csv')
print(df.tail())
```

**PowerShell (verificación rápida):**
```powershell
# Últimas líneas del CSV
Get-Content logs\geotermia_cnn_custom_*.csv -Tail 5

# Verificar que el proceso esté corriendo
Get-Process python

# Tamaño del modelo guardado
Get-ChildItem models\saved_models\*.keras | Select-Object Name, Length, LastWriteTime
```

### 5.2 Callbacks Configurados

| Callback | Configuración | Función |
|----------|--------------|---------|
| **EarlyStopping** | patience=15, monitor=val_loss, restore_best_weights=True | Detiene si no mejora por 15 épocas; restaura mejores pesos |
| **ModelCheckpoint** | save_best_only=True, monitor=val_loss | Guarda solo el mejor modelo en `models/saved_models/geotermia_cnn_custom_best.keras` |
| **ReduceLROnPlateau** | factor=0.5, patience=5, min_lr=1e-5 | Reduce LR a la mitad si val_loss no mejora en 5 épocas |
| **TensorBoard** | update_freq='epoch' | Registra métricas para visualización en tiempo real |
| **CSVLogger** | append=False | Guarda métricas por época en CSV |

### 5.3 Señales de Alerta

| Señal | Síntomas | Acción |
|-------|----------|--------|
| **Overfitting** | val_loss ↑ mientras loss ↓; val_accuracy ≪ accuracy | EarlyStopping lo detiene automáticamente |
| **Underfitting** | Ambos loss y val_loss altos; accuracy < 75% | Esperar más épocas o revisar modelo |
| **Loss explosiva** | loss → NaN; accuracy → 0% o 100% | ReduceLROnPlateau reduce LR automáticamente |
| **Estancamiento** | Métricas no cambian por muchas épocas | ReduceLROnPlateau intervendrá |

### 5.4 Interpretación de Métricas

| Nivel | Accuracy | val_accuracy | val_loss |
|-------|----------|-------------|----------|
| Excelente | > 90% | > 85% | < 0.3 |
| Bueno | > 85% | > 80% | < 0.4 |
| Aceptable | > 80% | > 75% | < 0.5 |
| Necesita mejora | < 80% | < 75% | > 0.5 |

---

## 6. Resultados del Entrenamiento Completado (18 de febrero de 2026)

### Evaluación en Test Set (396 imágenes)

| Métrica | Valor |
|---------|-------|
| Accuracy | 68.43% |
| Precision | 86.32% |
| Recall | 48.10% |
| F1-Score | 61.77% |
| ROC AUC | 0.8198 |
| R² | -0.2673 |

### Mejor Época (Época 8 de 23)

| Métrica | Train | Validation |
|---------|-------|------------|
| Loss | 0.6268 | 0.7710 |
| Accuracy | 83.40% | 70.96% |

**Diagnóstico:** Se detectó **overfitting significativo** a partir de la época 9. El train accuracy llegó a 91.75% mientras la val accuracy cayó a 44.70% en la época 23. EarlyStopping seleccionó correctamente la época 8 como mejor modelo. El ROC AUC de 0.8198 indica capacidad real de discriminación.

> Para el análisis detallado por época, consultar `ANALISIS_ENTRENAMIENTO.md`.

---

## 7. Métricas Objetivo

| Métrica | Mínimo | Ideal |
|---------|--------|-------|
| Accuracy | > 85% | > 90% |
| Precision | > 80% | > 85% |
| Recall | > 80% | > 85% |
| F1-Score | > 0.80 | > 0.85 |
| ROC AUC | > 0.90 | > 0.95 |

---

## 8. Flujo de Trabajo

### Fase 1 — Configuración (Completada)
Documentación, scripts, datos de metadata, repositorio en GitHub.

### Fase 2 — Entrenamiento (Completada — 18 de febrero de 2026)
Descarga de 85 imágenes → Augmentación a 2,635 → Preparación de splits → Entrenamiento 23 épocas en CPU → Evaluación en test → Visualizaciones generadas → Reporte PDF.

### Fase 3 — Mejoras y Finalización (En progreso)
Combatir overfitting → Mejorar recall → Actualizar documentación → Presentación de tesis.

---

## 9. Tecnologías

| Categoría | Herramientas |
|-----------|-------------|
| Deep Learning | TensorFlow 2.20.0, Keras 3.12.0 |
| Procesamiento | NumPy, pandas, scikit-learn, scikit-image, OpenCV, SciPy, rasterio |
| Datos geoespaciales | Google Earth Engine API, NASA ASTER GED AG100_003 |
| Visualización | Matplotlib, Seaborn, TensorBoard, Plotly, Folium |
| Interfaz | Streamlit, streamlit-folium |
| Reportes | FPDF2 |
| Control de versiones | Git, GitHub |

---

## 10. Estructura del Repositorio

```
geotermia-colombia-cnn/
├── README.md # README principal
├── app.py # Interfaz gráfica Streamlit
├── setup.py # Configuración del entorno
├── requirements.txt # Dependencias Python
├── .gitignore # Archivos excluidos
│
├── models/
│ ├── cnn_geotermia.py # Arquitectura del modelo CNN
│ ├── __init__.py
│ └── saved_models/ # Modelos entrenados (se generan)
│
├── scripts/
│ ├── download_dataset.py # Descarga imágenes desde GEE
│ ├── augment_full_dataset.py # Augmentación del dataset
│ ├── prepare_dataset.py # Preparación para entrenamiento
│ ├── train_model.py # Entrenamiento del modelo
│ ├── evaluate_model.py # Evaluación en test set
│ ├── predict.py # Predicción con coordenadas
│ ├── visualize_results.py # Visualizaciones de resultados
│ ├── visualize_architecture.py # Diagrama de arquitectura
│ └── miniprueba/ # Pipeline de validación (mini-dataset)
│
├── data/
│ ├── raw/ # Imágenes originales + CSVs de metadata
│ ├── augmented/ # Se genera con augment_full_dataset.py
│ └── processed/ # Se genera con prepare_dataset.py
│
├── docs/ # Documentación técnica
│ ├── MODELO_PREDICTIVO.md # Documento técnico principal
│ ├── REGISTRO_PROCESO.md # Bitácora cronológica
│ ├── ANALISIS_ENTRENAMIENTO.md # Análisis de métricas por época
│ ├── MEJORAS_MODELO.md # Roadmap de optimizaciones
│ ├── GUIA_PASO_A_PASO.md # Guía completa paso a paso
│ └── RESUMEN_PROYECTO.md # Este documento
│
├── logs/ # Logs de TensorBoard (se generan)
├── results/ # Métricas y figuras (se generan)
└── notebooks/ # Notebooks de exploración
```

---

## 11. Equipo

**Estudiantes:**
- Cristian Camilo Vega Sánchez (Lead Developer)
- Daniel Santiago Arévalo Rubiano
- Yuliet Katerin Espitia Ayala
- Laura Sophie Rivera Martín

**Asesor Académico:**
- Prof. Yeison Eduardo Conejo Sandoval

**Institución:**
- Universidad de San Buenaventura - Bogotá
- Facultad de Ingeniería
- Programa de Ingeniería de Sistemas

---

## 12. Notas Técnicas

### Reproducibilidad
- **Random seed:** 42 fijo en todos los scripts.
- **División estratificada** mantiene proporción de clases en train/val/test.
- **`requirements.txt`** con versiones exactas de dependencias.

### Prevención de Overfitting
- Dropout y SpatialDropout2D en la arquitectura.
- EarlyStopping con patience=15.
- Data Augmentation en tiempo real durante entrenamiento.
- Regularización L2 en capas densas.
- Label Smoothing (0.1) en la función de pérdida.

### Balance de Clases
- Pesos de clase: {0: 2.2247, 1: 0.6450}.
- Mayor peso a clase minoritaria (negativo) para evitar sesgo.

### Hardware Requerido
- **Mínimo:** CPU con 8 GB RAM (entrenamiento lento, ~90s/época).
- **Recomendado:** GPU NVIDIA con CUDA (~5-10s/época estimado).
- **Usado:** Intel i5-10300H (CPU), 12 GB RAM, TensorFlow 2.20.0 (sin CUDA en Windows).

---

**Última actualización:** 18 de febrero de 2026 
**Documento fusionado desde:** CONFIGURACION_COMPLETA.md, RESUMEN_EJECUTIVO.md, MONITOREO_ENTRENAMIENTO.md
