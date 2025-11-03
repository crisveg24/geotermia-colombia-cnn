# 📋 REGISTRO DEL PROCESO DE DESARROLLO
## Sistema CNN para Identificación de Zonas Geotérmicas en Colombia

**Proyecto:** Modelo Predictivo CNN - Geotermia Colombia  
**Institución:** Universidad de San Buenaventura - Bogotá  
**Autores:** Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano  
**Asesor:** Prof. Yeison Eduardo Conejo Sandoval  
**Fecha Inicio:** Noviembre 2025  
**Repositorio:** https://github.com/crisveg24/geotermia-colombia-cnn

---

## 📅 CRONOGRAMA DE ACTIVIDADES

### **FASE 1: CONFIGURACIÓN Y DOCUMENTACIÓN (Completada)**
**Fecha:** 3 de noviembre de 2025

#### 1.1 Documentación Técnica
- **Archivo creado:** `MODELO_PREDICTIVO.md` (2,700+ líneas)
- **Contenido:**
  - Resumen ejecutivo del proyecto
  - Fundamentos teóricos de CNNs
  - Arquitectura detallada del modelo (52 capas)
  - Pipeline de procesamiento
  - Estrategias de entrenamiento
  - Métricas de evaluación con ecuaciones LaTeX
  - Sistema de predicción
  - Optimizaciones
  - Casos de uso para Colombia
  - 11 referencias académicas
- **Commit:** `33343c8` - "docs: Agregar documento técnico completo del modelo predictivo CNN"

#### 1.2 Herramientas de Visualización
- **Archivo creado:** `scripts/visualize_architecture.py` (483 líneas)
- **Funcionalidad:**
  - Clase `ArchitectureVisualizer`
  - Generación de diagramas de arquitectura (PNG 300 DPI)
  - Tablas LaTeX para tesis
  - Resúmenes en JSON y TXT
  - Comparación con transfer learning
- **Commit:** `71b4627` - "feat: Agregar script de visualización de arquitectura CNN"

---

### **FASE 2: CONFIGURACIÓN DEL ENTORNO (Completada)**
**Fecha:** 3 de noviembre de 2025

#### 2.1 Instalación de Dependencias
- **Python:** 3.10.11
- **Entorno virtual:** `.venv` en `C:/Users/crsti/proyectos/`
- **Librerías instaladas:**
  ```
  TensorFlow: 2.20.0 (331.7 MB)
  Keras: 3.12.0
  scikit-learn: 1.7.2
  scikit-image: 0.25.2
  opencv-python: 4.12.0
  scipy: 1.15.3
  numpy: 2.2.6
  pandas: última versión
  rasterio: para lectura de GeoTIFF
  earthengine-api: para Google Earth Engine
  ```

#### 2.2 Verificación del Modelo
- **Modelo:** CNN personalizado con arquitectura ResNet-inspired
- **Parámetros totales:** 5,025,409
- **Capas totales:** 52
- **Input shape:** (None, 224, 224, 5) - 5 bandas térmicas ASTER
- **Output shape:** (None, 1) - clasificación binaria con sigmoid
- **Características:** Batch normalization, dropout, mixed precision
- **Estado:** ✅ Verificado y funcional

#### 2.3 Configuración Hardware
- **CPU:** Optimizaciones oneDNN habilitadas
- **Instrucciones:** SSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA
- **GPU:** No disponible (modo CPU)
- **Modo:** Entrenamiento en CPU con precisión mixta

---

### **FASE 3: ADQUISICIÓN DE DATOS (Completada)**
**Fecha:** 3 de noviembre de 2025

#### 3.1 Autenticación Google Earth Engine
- **Proyecto Google Cloud:** `alpine-air-469115-f0` (My First Project)
- **Método:** OAuth 2.0
- **Dataset:** NASA/ASTER_GED/AG100_003 (ASTER Global Emissivity Dataset)
- **Estado:** ✅ Autenticado y operacional

#### 3.2 Descarga de Imágenes Satelitales
- **Script:** `scripts/download_dataset.py`
- **Tiempo de ejecución:** ~4 minutos
- **Imágenes descargadas:** 85 imágenes ASTER
  
  **Imágenes Positivas (45):** Zonas con actividad geotérmica
  - Nevado del Ruiz: 5 ubicaciones (center, north, south, east, west)
  - Volcán Puracé: 5 ubicaciones
  - Volcán Galeras: 5 ubicaciones
  - Paipa-Iza: 5 ubicaciones
  - Nevado del Tolima: 5 ubicaciones
  - Volcán Cumbal: 5 ubicaciones
  - Volcán Sotará: 5 ubicaciones
  - Volcán Azufral: 5 ubicaciones
  - Zonas termales: Manizales, Coconuco, Santa Rosa de Cabal, Herveo, Villa María
  
  **Imágenes Negativas (40):** Zonas de control sin actividad volcánica
  - Llanos Orientales: 10 ubicaciones (Casanare, Arauca, Meta, Vichada)
  - Amazonas: 8 ubicaciones (Leticia, Puerto Nariño, Florencia, etc.)
  - Costa Caribe: 8 ubicaciones (Barranquilla, Santa Marta, etc.)
  - Zona Andina Oriental: 5 ubicaciones (Bucaramanga, Cúcuta, etc.)
  - Valle del Cauca: 5 ubicaciones (Cali, Palmira, etc.)
  - Chocó: 2 ubicaciones (Quibdó, Bahía Solano)

- **Tamaño total:** 2.49 MB
- **Balance inicial:** 88.9% (45/85 positivas)
- **Formato:** GeoTIFF con 5 bandas térmicas
- **Resolución espacial:** Variable según zona
- **Salida:** `data/raw/` con subdirectorios `positive/` y `negative/`

#### 3.3 Metadata Generada
- `data/raw/dataset_metadata.json`: Información del dataset
- `data/raw/dataset_images.csv`: Lista de imágenes con coordenadas
- `data/raw/labels.csv`: Etiquetas binarias (0: negativo, 1: positivo)

---

### **FASE 4: AUGMENTACIÓN DEL DATASET (Completada)**
**Fecha:** 3 de noviembre de 2025

#### 4.1 Prueba Inicial de Augmentación
- **Script de prueba:** `scripts/augment_dataset.py`
- **Imágenes de prueba:** 3 imágenes originales
- **Resultado:** 93 imágenes (31 por original)
- **Tiempo:** 0.83 segundos
- **Estado:** ✅ Validación exitosa

#### 4.2 Augmentación Completa del Dataset
- **Script:** `scripts/augment_full_dataset.py` (350+ líneas)
- **Clase:** `FullDatasetAugmenter`
- **Tiempo de ejecución:** 27 segundos

**Técnicas de Augmentación Aplicadas (30 por imagen):**

1. **Transformaciones Geométricas:**
   - Rotación 90° (rotation_90)
   - Rotación 180° (rotation_180)
   - Rotación 270° (rotation_270)
   - Rotación 45° (rotation_45)
   - Rotación -45° (rotation_neg45)
   - Flip horizontal (flip_horizontal)
   - Flip vertical (flip_vertical)

2. **Transformaciones de Intensidad:**
   - Brillo +20% (brightness_1.2)
   - Brillo -20% (brightness_0.8)
   - Contraste +30% (contrast_1.3)
   - Contraste -30% (contrast_0.7)

3. **Técnicas de Ruido y Suavizado:**
   - Ruido gaussiano pequeño (noise_small)
   - Ruido gaussiano medio (noise_medium)
   - Desenfoque gaussiano ligero (blur_light)
   - Desenfoque gaussiano medio (blur_medium)

4. **Recorte:**
   - Crop 90% (crop_0.9)
   - Crop 85% (crop_0.85)

5. **Combinaciones Complejas (13 técnicas):**
   - rot90_flip_h: Rotación 90° + flip horizontal
   - rot180_bright: Rotación 180° + brillo
   - flip_v_contrast: Flip vertical + contraste
   - rot45_noise: Rotación 45° + ruido
   - crop_blur: Recorte + desenfoque
   - bright_blur: Brillo + desenfoque
   - contrast_noise: Contraste + ruido
   - rot90_crop: Rotación 90° + recorte
   - rot180_contrast: Rotación 180° + contraste
   - flip_h_bright: Flip horizontal + brillo
   - rot270_blur: Rotación 270° + desenfoque
   - crop_contrast_noise: Recorte + contraste + ruido
   - rot45_bright_blur: Rotación 45° + brillo + desenfoque

**Resultados de la Augmentación:**
- **Input:** 85 imágenes originales
- **Output:** 5,518 imágenes totales
- **Distribución:**
  - Positivas: 4,278 imágenes (77.5%)
  - Negativas: 1,240 imágenes (22.5%)
- **Tamaño:** 1.24 GB (1,240.27 MB)
- **Ubicación:** `data/augmented/positive/` y `data/augmented/negative/`

#### 4.3 Corrección de Metadata
- **Problema detectado:** Labels.csv sin subdirectorios en rutas
- **Script corrector:** `scripts/fix_labels.py`
- **Acción:** Actualización de 5,518 rutas con prefijos `positive/` o `negative/`
- **Resultado:** Labels.csv corregido para lectura correcta

---

### **FASE 5: PREPARACIÓN DEL DATASET (Completada)**
**Fecha:** 3 de noviembre de 2025 - 18:33

#### 5.1 Procesamiento de Imágenes
- **Script:** `scripts/prepare_dataset.py` (418 líneas)
- **Clase:** `GeoDataPreparator`
- **Tiempo de procesamiento:** ~2 minutos (5,518 imágenes)

**Correcciones Realizadas:**
1. Actualización de rutas por defecto:
   - `raw_data_path='data/augmented'` (antes: 'data/raw')
   - `labels_path='data/augmented'` (antes: 'data/labels')

2. Normalización de bandas espectrales:
   - Detección automática de número de bandas
   - Expansión a 5 bandas si < 5 (duplicación de última banda)
   - Recorte a 5 bandas si > 5 (tomar primeras 5)
   - Garantiza consistencia: todas las imágenes tienen exactamente 5 bandas

**Procesamiento Aplicado:**
- Carga de imágenes GeoTIFF con rasterio
- Redimensionamiento a 224x224 píxeles
- Normalización de valores de píxel (0-1)
- Preservación de 5 bandas térmicas ASTER

#### 5.2 División del Dataset
- **Estrategia:** Estratificada para mantener proporción de clases
- **Random state:** 42 (reproducibilidad)

**Distribución Final:**
```
Training Set:   3,862 imágenes (70.0%)
  - Clase 0 (negativo): 868 imágenes
  - Clase 1 (positivo): 2,994 imágenes

Validation Set: 828 imágenes (15.0%)
  - Clase 0 (negativo): 186 imágenes
  - Clase 1 (positivo): 642 imágenes

Test Set:       828 imágenes (15.0%)
  - Clase 0 (negativo): 186 imágenes
  - Clase 1 (positivo): 642 imágenes

TOTAL:          5,518 imágenes (100%)
```

#### 5.3 Pesos de Clase para Balanceo
**Objetivo:** Compensar desbalance entre clases durante entrenamiento

```python
Clase 0 (negativo): peso = 2.2247
Clase 1 (positivo): peso = 0.6450

Cálculo: peso_clase = n_samples / (n_classes * n_samples_clase)
```

- Mayor peso a clase minoritaria (negativos)
- Menor peso a clase mayoritaria (positivos)
- Evita sesgo hacia clase dominante

#### 5.4 Archivos Generados
**Ubicación:** `data/processed/`

```
X_train.npy      - Imágenes de entrenamiento:  (3862, 224, 224, 5) ~1.5 GB
y_train.npy      - Etiquetas de entrenamiento: (3862,)

X_val.npy        - Imágenes de validación:     (828, 224, 224, 5)  ~320 MB
y_val.npy        - Etiquetas de validación:    (828,)

X_test.npy       - Imágenes de prueba:         (828, 224, 224, 5)  ~320 MB
y_test.npy       - Etiquetas de prueba:        (828,)

dataset_info.json - Metadata completa del dataset procesado
```

**Formato:** NumPy arrays con dtype=float32
**Acceso rápido:** Carga directa con `np.load()`

---

### **FASE 6: ENTRENAMIENTO DEL MODELO (En Progreso)**
**Fecha:** 3 de noviembre de 2025 - 18:35 (inicio programado)

#### 6.1 Configuración del Entrenamiento
- **Script:** `scripts/train_model.py`
- **Modelo:** CNN personalizado (52 capas, 5,025,409 parámetros)
- **Hardware:** CPU con optimizaciones oneDNN
- **Precision:** Mixed precision (float16/float32)
- **Tiempo estimado:** 2-3 horas

#### 6.2 Hiperparámetros
```python
Batch size:        32
Épocas máximas:    100
Learning rate:     0.001
Optimizer:         Adam
Loss function:     Binary Crossentropy
Métricas:          Accuracy, Precision, Recall, AUC
```

#### 6.3 Callbacks Configurados
1. **EarlyStopping:**
   - Monitor: validation loss
   - Patience: 15 épocas
   - Restaura mejores pesos

2. **ModelCheckpoint:**
   - Guarda mejor modelo según val_loss
   - Formato: Keras (.keras)
   - Ubicación: `models/best_model.keras`

3. **ReduceLROnPlateau:**
   - Reduce learning rate si no mejora
   - Factor: 0.5
   - Patience: 5 épocas

4. **TensorBoard:**
   - Logs de entrenamiento
   - Ubicación: `logs/tensorboard/`
   - Visualización de métricas en tiempo real

5. **CSVLogger:**
   - Registro de métricas por época
   - Archivo: `models/training_history.csv`

#### 6.4 Data Augmentation en Tiempo Real
**Aplicado solo durante entrenamiento:**
- Random rotation: ±10°
- Random horizontal flip
- Random vertical flip
- Random zoom: ±10%
- Random brightness: ±10%

#### 6.5 Salidas Esperadas
```
models/
  ├── best_model.keras           - Mejor modelo guardado
  ├── training_history.json      - Historial completo
  └── training_history.csv       - Métricas por época

logs/
  └── tensorboard/               - Logs para TensorBoard
      └── [timestamp]/
```

---

### **FASE 7: EVALUACIÓN DEL MODELO (Pendiente)**

#### 7.1 Métricas a Calcular
- **Script:** `scripts/evaluate_model.py`
- **Dataset:** Test set (828 imágenes)

**Métricas Principales:**
1. **Accuracy:** Precisión general del modelo
2. **Precision:** TP / (TP + FP)
3. **Recall (Sensibilidad):** TP / (TP + FN)
4. **F1-Score:** Media armónica de precision y recall
5. **ROC AUC:** Área bajo la curva ROC
6. **R² Score:** Coeficiente de determinación
7. **Confusion Matrix:** Matriz de confusión 2x2

#### 7.2 Análisis por Clase
- Precision, Recall, F1 para cada clase
- Support (número de muestras)
- Análisis de falsos positivos y negativos

#### 7.3 Archivos de Salida
```
results/metrics/
  ├── evaluation_metrics.json    - Todas las métricas
  ├── metrics_table.csv          - Tabla para tesis
  ├── confusion_matrix.png       - Visualización (300 DPI)
  └── roc_curve.png              - Curva ROC (300 DPI)
```

---

### **FASE 8: VISUALIZACIÓN DE RESULTADOS (Pendiente)**

#### 8.1 Gráficos de Entrenamiento
- **Script:** `scripts/visualize_results.py`
- **Resolución:** 300 DPI (calidad publicación)

**Visualizaciones a Generar:**
1. **Training History:**
   - Loss (train vs validation)
   - Accuracy (train vs validation)
   - Formato: Curvas en misma figura

2. **Confusion Matrix:**
   - Heatmap con seaborn
   - Anotaciones de valores
   - Normalizada y sin normalizar

3. **ROC Curve:**
   - Curva ROC con AUC
   - Línea diagonal de referencia
   - Threshold óptimo marcado

4. **Predicciones de Muestra:**
   - Grid de imágenes reales
   - Predicciones vs etiquetas verdaderas
   - Probabilidades de confianza

5. **Distribución de Probabilidades:**
   - Histograma de predicciones
   - Separación por clase real

#### 8.2 Archivos de Salida
```
results/figures/
  ├── training_history.png       - Curvas de entrenamiento
  ├── confusion_matrix.png       - Matriz de confusión
  ├── confusion_matrix_norm.png  - Matriz normalizada
  ├── roc_curve.png              - Curva ROC
  ├── sample_predictions.png     - Muestras de predicciones
  └── probability_distribution.png - Distribución de probabilidades
```

---

### **FASE 9: DOCUMENTACIÓN FINAL (Pendiente)**

#### 9.1 Actualización de README
- Resultados finales del entrenamiento
- Métricas de performance
- Instrucciones de uso del modelo
- Ejemplos de predicción

#### 9.2 Documento de Resultados
- Análisis de métricas
- Comparación con objetivos
- Limitaciones del modelo
- Recomendaciones para mejoras

#### 9.3 Commit Final
```bash
git add models/best_model.keras
git add results/
git add README.md
git add REGISTRO_PROCESO.md
git commit -m "feat: Modelo CNN entrenado con métricas completas"
git push origin main
```

---

## 📊 ESTADÍSTICAS DEL PROYECTO

### Dataset
- **Imágenes originales descargadas:** 85
- **Imágenes después de augmentación:** 5,518
- **Factor de aumento:** 64.9x
- **Tamaño total procesado:** ~2.5 GB
- **Bandas espectrales por imagen:** 5 (ASTER térmico)
- **Resolución final:** 224x224 píxeles

### Modelo
- **Arquitectura:** CNN personalizada ResNet-inspired
- **Capas totales:** 52
- **Parámetros entrenables:** 5,025,409
- **Input shape:** (224, 224, 5)
- **Output:** Clasificación binaria (sigmoid)

### Distribución de Datos
```
Training:    3,862 imágenes (70%)
Validation:    828 imágenes (15%)
Test:          828 imágenes (15%)
Total:       5,518 imágenes (100%)

Balance de clases:
  Positivo (geotérmico): 77.5%
  Negativo (control):    22.5%
```

---

## 🛠️ TECNOLOGÍAS UTILIZADAS

### Framework de Deep Learning
- **TensorFlow:** 2.20.0
- **Keras:** 3.12.0

### Procesamiento de Datos
- **NumPy:** 2.2.6
- **pandas:** última versión
- **scikit-learn:** 1.7.2
- **scikit-image:** 0.25.2
- **opencv-python:** 4.12.0
- **scipy:** 1.15.3

### Datos Geoespaciales
- **rasterio:** Lectura de GeoTIFF
- **earthengine-api:** Google Earth Engine
- **Dataset:** NASA ASTER GED AG100_003

### Visualización
- **matplotlib:** Gráficos estáticos
- **seaborn:** Visualizaciones estadísticas
- **TensorBoard:** Monitoreo de entrenamiento

### Control de Versiones
- **Git:** Control de versiones
- **GitHub:** Repositorio remoto

---

## 📈 RESULTADOS ESPERADOS

### Objetivos de Performance
- **Accuracy mínima esperada:** 85%
- **Precision objetivo:** >80% para ambas clases
- **Recall objetivo:** >80% para ambas clases
- **F1-Score objetivo:** >0.80
- **AUC objetivo:** >0.90

### Aplicación Práctica
El modelo entrenado podrá:
1. Identificar zonas con potencial geotérmico en Colombia
2. Diferenciar entre zonas volcánicas activas y zonas de control
3. Procesar imágenes satelitales ASTER de 5 bandas térmicas
4. Proporcionar probabilidades de confianza en las predicciones
5. Servir como herramienta de apoyo para exploración geotérmica

---

## 🔄 PRÓXIMOS PASOS

1. ⏳ **Completar entrenamiento del modelo** (~2-3 horas)
2. ⏳ **Evaluar performance en test set**
3. ⏳ **Generar visualizaciones de alta calidad**
4. ⏳ **Documentar resultados finales**
5. ⏳ **Preparar presentación para tesis**

---

## 👥 EQUIPO

**Estudiantes:**
- Cristian Camilo Vega Sánchez (Lead Developer)
- Daniel Santiago Arévalo Rubiano (Co-author)

**Asesor:**
- Prof. Yeison Eduardo Conejo Sandoval

**Institución:**
- Universidad de San Buenavenaria - Bogotá
- Facultad de Ingeniería
- Programa de Ingeniería de Sistemas

---

## 📝 NOTAS TÉCNICAS

### Consideraciones Importantes

1. **Balance de Clases:**
   - Se aplicaron pesos de clase para compensar desbalance
   - Data augmentation más agresiva en clase minoritaria

2. **Validación Cruzada:**
   - División estratificada mantiene proporción de clases
   - Random state fijo (42) garantiza reproducibilidad

3. **Optimizaciones de Hardware:**
   - CPU con instrucciones SIMD habilitadas
   - Mixed precision para acelerar entrenamiento
   - Batch size optimizado para memoria disponible

4. **Prevención de Overfitting:**
   - Dropout layers en arquitectura
   - Early stopping con patience=15
   - Data augmentation en tiempo real
   - Regularización L2 en capas densas

5. **Monitoreo:**
   - TensorBoard para seguimiento en tiempo real
   - CSVLogger para análisis posterior
   - Checkpoints automáticos del mejor modelo

---

**Última actualización:** 3 de noviembre de 2025 - 18:35  
**Estado del proyecto:** Fase 6 - Entrenamiento en progreso  
**Próxima revisión:** Al completar entrenamiento
