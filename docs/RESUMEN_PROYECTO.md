# 📊 RESUMEN DEL PROYECTO - CNN Geotermia Colombia

**Proyecto:** Sistema CNN para Identificación de Zonas Geotérmicas en Colombia  
**Institución:** Universidad de San Buenaventura - Bogotá  
**Fecha de inicio:** Noviembre 2025  
**Última actualización:** 9 de febrero de 2026  
**Repositorio:** https://github.com/crisveg24/geotermia-colombia-cnn

---

## 1. Estado General del Proyecto

| Componente | Progreso | Notas |
|-----------|----------|-------|
| Documentación técnica | 100% | `MODELO_PREDICTIVO.md` (1,269 líneas) |
| Scripts de pipeline | 100% | Descarga, augmentación, preparación, entrenamiento, evaluación |
| Dataset original | 100% | 85 imágenes ASTER descargadas desde GEE |
| Dataset augmentado | 100% | 5,518 imágenes (64.9x factor de aumento) |
| Entrenamiento parcial | 30% | 30/100 épocas en CPU |
| Interfaz gráfica | 100% | Streamlit con Folium, Plotly |
| Optimizaciones del modelo | 100% | SpatialDropout2D, AdamW, Label Smoothing, PR-AUC, F1Score |
| Entrenamiento completo | Pendiente | Requiere GPU (RTX 5070 objetivo) |
| Evaluación final | Pendiente | Tras completar entrenamiento |

---

## 2. Logros Completados

### 2.1 Documentación Técnica
- **`MODELO_PREDICTIVO.md`** — Fundamentos teóricos de CNNs, arquitectura detallada (52 capas), pipeline de procesamiento, métricas con ecuaciones LaTeX, 11 referencias académicas.
- **`REGISTRO_PROCESO.md`** — Cronograma de 9 fases con estadísticas completas.
- **`ANALISIS_ENTRENAMIENTO.md`** — Tabla época-por-época (30 épocas), análisis de tendencias, proyecciones.
- **`MEJORAS_MODELO.md`** — Roadmap de optimizaciones aplicadas y futuras.
- **`ENTRENAMIENTO_EXTERNO.md`** — Guía paso a paso para máquina con GPU.

### 2.2 Adquisición y Procesamiento de Datos
- **85 imágenes ASTER** descargadas desde Google Earth Engine (NASA/ASTER_GED/AG100_003).
- **45 positivas** de 9 zonas volcánicas/geotérmicas (Nevado del Ruiz, Puracé, Galeras, Paipa, Tolima, Cumbal, Sotará, Azufral, termales).
- **40 negativas** de 5 zonas de control (Llanos, Amazonas, Costa Caribe, Zona Andina Oriental, Chocó).
- **5,518 imágenes** tras augmentación con 30 transformaciones (geométricas, intensidad, ruido, combinaciones).
- **División estratificada:** 3,862 train (70%) / 828 val (15%) / 828 test (15%).

### 2.3 Modelo CNN
- Arquitectura ResNet-inspired personalizada: 52 capas, 5,025,409 parámetros.
- Input: (224, 224, 5) — 5 bandas térmicas ASTER (bandas 10-14).
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
Epoch 1/100
120/120 [==============================] - 85s - loss: 0.6543 - accuracy: 0.7234 - val_loss: 0.5432 - val_accuracy: 0.7823
```

**TensorBoard (recomendado):**
```bash
python -m tensorboard --logdir=logs/tensorboard
# Abrir navegador en: http://localhost:6006
```

**CSV de historial:**
```python
import pandas as pd
df = pd.read_csv('models/training_history.csv')
print(df.tail())
```

**PowerShell (verificación rápida):**
```powershell
# Últimas líneas del CSV
Get-Content models/training_history.csv -Tail 5

# Verificar que el proceso esté corriendo
Get-Process python

# Tamaño del modelo guardado
Get-ChildItem models/best_model.keras | Select-Object Name, Length, LastWriteTime
```

### 5.2 Callbacks Configurados

| Callback | Configuración | Función |
|----------|--------------|---------|
| **EarlyStopping** | patience=15, monitor=val_loss, restore_best_weights=True | Detiene si no mejora por 15 épocas; restaura mejores pesos |
| **ModelCheckpoint** | save_best_only=True, monitor=val_loss | Guarda solo el mejor modelo en `models/best_model.keras` |
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

## 6. Métricas del Entrenamiento Parcial (Época 30/100)

| Métrica | Valor | Tendencia |
|---------|-------|-----------|
| Accuracy | 65.26% | Mejorando (+0.07%/época) |
| AUC | 0.6252 | Crecimiento sostenido (+39.5% desde época 1) |
| Loss | 0.9241 | Disminuyendo (-6.6% total) |
| Precision | 84.61% | Excelente |
| Recall | 68.27% | Moderado, margen de mejora |
| F1-Score | ~75.54% | Balance aceptable |

**Diagnóstico:** No hay overfitting. Convergencia estable. El modelo necesita completar las 100 épocas con GPU para alcanzar los objetivos.

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

### Fase 2 — Entrenamiento (Pendiente)
Clonar en máquina con GPU → regenerar datos → entrenar 100 épocas → evaluar → push resultados.

```bash
# En máquina con GPU:
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn
# Seguir docs/ENTRENAMIENTO_EXTERNO.md
```

### Fase 3 — Finalización (Pendiente)
Pull resultados → merge → actualizar documentación → presentación de tesis.

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
├── README.md                         # README principal
├── app.py                            # Interfaz gráfica Streamlit
├── setup.py                          # Configuración del entorno
├── requirements.txt                  # Dependencias Python
├── .gitignore                        # Archivos excluidos
│
├── models/
│   ├── cnn_geotermia.py              # Arquitectura del modelo CNN
│   ├── __init__.py
│   └── saved_models/                 # Modelos entrenados (se generan)
│
├── scripts/
│   ├── download_dataset.py           # Descarga imágenes desde GEE
│   ├── augment_full_dataset.py       # Augmentación del dataset
│   ├── prepare_dataset.py            # Preparación para entrenamiento
│   ├── train_model.py                # Entrenamiento del modelo
│   ├── evaluate_model.py             # Evaluación en test set
│   ├── predict.py                    # Predicción con coordenadas
│   ├── visualize_results.py          # Visualizaciones de resultados
│   ├── visualize_architecture.py     # Diagrama de arquitectura
│   └── miniprueba/                   # Pipeline de validación (mini-dataset)
│
├── data/
│   ├── raw/                          # Imágenes originales + CSVs de metadata
│   ├── augmented/                    # Se genera con augment_full_dataset.py
│   └── processed/                    # Se genera con prepare_dataset.py
│
├── docs/                             # Documentación técnica
│   ├── MODELO_PREDICTIVO.md          # Documento técnico principal
│   ├── REGISTRO_PROCESO.md           # Bitácora cronológica
│   ├── ANALISIS_ENTRENAMIENTO.md     # Análisis de métricas por época
│   ├── MEJORAS_MODELO.md             # Roadmap de optimizaciones
│   ├── ENTRENAMIENTO_EXTERNO.md      # Guía para entrenar con GPU
│   └── RESUMEN_PROYECTO.md           # Este documento
│
├── logs/                             # Logs de TensorBoard (se generan)
├── results/                          # Métricas y figuras (se generan)
└── notebooks/                        # Notebooks de exploración
```

---

## 11. Equipo

**Estudiantes:**
- Cristian Camilo Vega Sánchez (Lead Developer)
- Daniel Santiago Arévalo Rubiano
- Yuliet Katerin Espitia Ayala
- Laura Sophie Rivera Martin

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
- **Mínimo:** CPU con 8 GB RAM (entrenamiento lento, ~117s/época).
- **Recomendado:** GPU NVIDIA con CUDA (RTX 5070 objetivo, ~5-10s/época estimado).

---

**Última actualización:** 9 de febrero de 2026  
**Documento fusionado desde:** CONFIGURACION_COMPLETA.md, RESUMEN_EJECUTIVO.md, MONITOREO_ENTRENAMIENTO.md
