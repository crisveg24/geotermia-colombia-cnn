# Modelo Predictivo de Potencial Geotérmico en Colombia con CNN

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Universidad](https://img.shields.io/badge/Universidad-San%20Buenaventura%20Bogot%C3%A1-red.svg)](https://www.usbbog.edu.co/)

<p align="center">
 <img src="https://img.shields.io/badge/Deep%20Learning-CNN-blueviolet" />
 <img src="https://img.shields.io/badge/Computer%20Vision-Geospatial-success" />
 <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

## Descripción

**Proyecto de Grado - Universidad de San Buenaventura Bogotá**

Implementación de un **modelo de Deep Learning basado en Redes Neuronales Convolucionales (CNN)** para la identificación automatizada de zonas con alto potencial geotérmico en Colombia mediante el análisis de imágenes satelitales térmicas del sensor **NASA ASTER** (Advanced Spaceborne Thermal Emission and Reflection Radiometer).

> **Herramienta de screening (Fase 1 de exploración geotérmica):** Este modelo actúa como un filtro automatizado que analiza datos ASTER y produce una probabilidad (0–100%) de potencial geotérmico para cualquier punto de Colombia. Su objetivo es priorizar zonas para inversión en exploración detallada, no confirmar la existencia de recursos explotables. Ver [docs/CONTEXTO_GEOTERMICO.md](docs/CONTEXTO_GEOTERMICO.md) para fundamentos teóricos.

> **Nota sobre el dataset (v3):** Para robustecer el entrenamiento, el dataset incluye zonas de la **Región Andina** (Colombia, Ecuador, Perú y Chile), ya que comparten el mismo contexto geológico del Cinturón de Fuego del Pacífico. La aplicación web y las conclusiones se limitan exclusivamente a Colombia. Ver [docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md](docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md).

### Características Principales

- **Dataset Región Andina v3**: 2,019 imágenes base de 4 países (Colombia, Ecuador, Perú, Chile)
- **22,209 imágenes** tras augmentación (×10 variaciones por imagen)
- **Anti-leakage geográfico**: GroupShuffleSplit agrupa por zona geográfica base
- **Descarga paralela**: 3 hilos concurrentes vía Google Earth Engine API
- **Arquitectura CNN moderna** con bloques residuales (ResNet-inspired)
- **Transfer Learning** con EfficientNet y ResNet50V2
- **Mixed Precision Training** para optimizar rendimiento
- **Data Augmentation** avanzado con SpatialDropout2D
- **Métricas con Bootstrap CI 95%** (Accuracy, Precision, Recall, F1-Score, ROC AUC, MCC)
- **Pipeline completo** desde descarga de datos hasta predicción
- **Interfaz Web** con Streamlit para visualización interactiva
- **Optimizador AdamW** con regularización de pesos mejorada
- **Label Smoothing** para reducir overfitting
- **Cosine Learning Rate Decay** para mejor convergencia
- **Soporte disco externo**: Variable `GEOTERMIA_DATA_ROOT` para datasets grandes

---

## Equipo de Desarrollo

| Rol | Nombre | Email | GitHub |
|-----|--------|-------|--------|
| **Desarrollador** | Cristian Camilo Vega Sánchez | ccvegas@academia.usbbog.edu.co | [@crisveg24](https://github.com/crisveg24) |
| **Co-autor** | Daniel Santiago Arévalo Rubiano | dsarevalor@academia.usbbog.edu.co | - |
| **Co-autora** | Yuliet Katerin Espitia Ayala | ykespitiaa@academia.usbbog.edu.co | - |
| **Co-autora** | Laura Sophie Rivera Martín | lsriveram@academia.usbbog.edu.co | - |
| **Asesor Académico** | Prof. Yeison Eduardo Conejo Sandoval | yconejo@usbbog.edu.co | - |

**Institución**: Universidad de San Buenaventura - Sede Bogotá 
**Programa**: Ingeniería de Sistemas (Pregrado) 
**Año**: 2025-2026

---

## Interfaz Web Interactiva

El proyecto incluye una **aplicación web** desarrollada con Streamlit para:

- **Predicción por coordenadas**: Ingresa latitud/longitud y obtén predicción de potencial geotérmico
- **Mapa interactivo**: Visualiza zonas geotérmicas de Colombia
- **Métricas del modelo**: Gráficos interactivos de rendimiento
- **Arquitectura**: Diagrama visual de la red neuronal

### Ejecutar la interfaz

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

---

## Zonas de Estudio

El modelo se entrena con zonas de la **Región Andina** (4 países) y se aplica a **Colombia**.

### Dataset v3 — Región Andina

| País | Zonas positivas | Zonas negativas | Total |
|------|:-:|:-:|:-:|
| **Colombia** | ~375 (volcanes, termas, campos SGC) | ~89 (Llanos, Amazonía, Costa) | ~464 |
| **Ecuador** | ~170 (Cotopaxi, Tungurahua, Chachimbiro…) | ~180 (costa, Amazonía) | ~350 |
| **Perú** | ~160 (Misti, Ubinas, Tacna…) | ~180 (costa, sierra estable) | ~340 |
| **Chile** | ~280 (El Tatio, Cerro Pabellón, Villarrica…) | ~575 (valle central, Patagonia) | ~855 |
| **Total** | **997** | **1,022** | **2,019** |

Cada zona se expande en una grilla de 9 tiles (center + 8 direcciones) para maximizar cobertura espacial. Ver catálogo completo en [docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md](docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md).

### Zonas Clave de Colombia

1. **Nevado del Ruiz** (Tolima) — Volcán activo, campo geotérmico de alta entalpía
2. **Volcán Puracé** (Cauca) — Campo geotérmico confirmado por SGC
3. **Paipa-Iza** (Boyacá) — Campo de baja entalpía, aguas termales
4. **Volcán Galeras** (Nariño) — Estratovolcán con fumarolas permanentes
5. **Tufiño-Chiles** (Nariño) — Campo binacional Colombia-Ecuador

### Dataset Satelital

**ASTER Global Emissivity Dataset (AG100) V003**
- **Proveedor**: NASA/METI/AIST/Japan Spacesystems
- **Resolución espacial**: 100 metros (escala 90 m/px en GEE)
- **Bandas**: 7 (emisividad bandas 10-14 + temperatura superficial + NDVI)
- **Buffer por zona**: 5 km de radio
- **Cobertura**: Región Andina (Colombia, Ecuador, Perú, Chile)
- **Fuente**: Google Earth Engine API (descarga paralela, 3 hilos)

---

## Arquitectura del Proyecto

```
geotermia-colombia-cnn/
│
├── app.py # Interfaz web Streamlit
├── config.py # Configuración centralizada de rutas
│
├── data/ # Datos (o GEOTERMIA_DATA_ROOT externo)
│ ├── raw/ # Imágenes satelitales (.tif) + labels.csv
│ ├── augmented/ # Dataset augmentado (×10, se genera)
│ └── processed/ # Datos procesados (.npy, se genera)
│
├── docs/ # Documentación técnica
│ ├── RESUMEN_PROYECTO.md # Vista general, estado y monitoreo
│ ├── MODELO_PREDICTIVO.md # Documentación técnica del modelo CNN
│ ├── CAMPOS_GEOTERMICOS_REGION_ANDINA.md # Catálogo de campos geotérmicos (4 países)
│ ├── CONTEXTO_GEOTERMICO.md # Fundamentos teóricos geotérmicos
│ ├── ANALISIS_ENTRENAMIENTO.md # Análisis de métricas por época
│ ├── CHANGELOG_V2.md # Registro de cambios v1→v2
│ ├── PREDICCIONES_PRUEBA.md # Predicciones y análisis
│ └── GUIA_PASO_A_PASO.md # Guía completa paso a paso
│
├── models/ # Modelos de Deep Learning
│ ├── __init__.py
│ ├── cnn_geotermia.py # Arquitectura CNN principal
│ ├── README.md
│ └── saved_models/ # Modelos entrenados (.keras)
│
├── scripts/ # Scripts de ejecución
│ ├── download_dataset.py # Descarga de imágenes ASTER
│ ├── augment_full_dataset.py # Augmentación del dataset
│ ├── prepare_dataset.py # Preparación de datos (.npy)
│ ├── train_model.py # Entrenamiento CNN
│ ├── evaluate_model.py # Evaluación de métricas
│ ├── visualize_results.py # Visualizaciones
│ ├── predict.py # Predicciones
│ ├── visualize_architecture.py # Visualización de arquitectura
│ ├── miniprueba/ # Scripts de validación rápida
│ └── README.md
│
├── notebooks/ # Jupyter Notebooks
│ └── descargarimagenes.ipynb # Exploración de datos
│
├── results/ # Resultados para tesis
│ ├── figures/ # Gráficos (PNG 300 DPI)
│ ├── metrics/ # Métricas (JSON, CSV)
│ └── reporte_mini_dataset.pdf # Reporte PDF generado
│
├── logs/ # Logs de entrenamiento
│
├── requirements.txt # Dependencias Python
├── README.md # Este archivo
├── LICENSE # Licencia MIT
└── setup.py # Script de configuración
```

---

## Instalación y Configuración

### 1. Requisitos Previos

- **Python 3.10 o superior**
- **CUDA 12+** (opcional, para GPU — requerido por TensorFlow 2.20+)
- **Cuenta de Google Earth Engine** ([registrarse aquí](https://earthengine.google.com/signup/))
- **Git**

### 2. Clonar el Repositorio

```bash
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn
```

### 3. Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Autenticar Google Earth Engine

```bash
python -c "import ee; ee.Authenticate()"
```

Sigue las instrucciones en el navegador para autorizar el acceso.

### 6. Verificar Instalación

```bash
python setup.py
```

---

## Guía de Uso

### Pipeline Completo

> **Disco externo (opcional):** Si usas un disco externo para datos, configura antes:
> ```bash
> # Windows PowerShell
> $env:GEOTERMIA_DATA_ROOT = "E:\geotermia_datos"
> ```

#### **Paso 1: Descargar Imágenes ASTER**

```bash
python scripts/download_dataset.py
```

**¿Qué hace?**
- Descarga imágenes ASTER GED desde Google Earth Engine
- 2,019 imágenes de la Región Andina (CO, EC, PE, CL)
- Descarga paralela con 3 hilos concurrentes
- Expande cada zona en grilla de 9 tiles
- Genera `labels.csv` con metadatos

**Salidas:**
- `data/raw/*.tif` (imágenes 7 bandas, ~58 KB c/u)
- `data/raw/labels.csv`

---

#### **Paso 2: Augmentar Dataset**

```bash
python scripts/augment_full_dataset.py
```

**¿Qué hace?**
- Genera 10 variaciones por imagen (flips, rotación, brillo, noise, etc.)
- Preserva las 7 bandas originales
- 2,019 → 22,209 imágenes

**Salidas:**
- `data/augmented/*.tif` (~364 KB c/u, float32 sin compresión)
- `data/augmented/labels.csv`

---

#### **Paso 3: Preparar Dataset**

```bash
python scripts/prepare_dataset.py
```

**¿Qué hace?**
- Carga imágenes .tif desde `data/augmented/`
- Normaliza y redimensiona a 224×224 (interpolación bicúbica)
- Crea splits train/val/test (70/15/15) **con anti-leakage geográfico**
- Usa GroupShuffleSplit agrupando por zona geográfica base
- Genera archivos .npy particionados para carga incremental
- Calcula pesos de clase para balanceo

**Salidas:**
- `data/processed/X_train_part_*.npy` (31 archivos)
- `data/processed/y_train.npy`
- `data/processed/X_val_part_*.npy` (7 archivos)
- `data/processed/X_test_part_*.npy` (7 archivos)
- `data/processed/class_weights.json`

---

#### **Paso 4: Entrenar Modelo CNN**

```bash
python scripts/train_model.py
```

**¿Qué hace?**
- Construye arquitectura CNN con bloques residuales
- Aplica data augmentation (flips, rotations, zoom)
- Entrena con Mixed Precision
- Guarda mejor modelo automáticamente
- Registra logs en TensorBoard

**Salidas:**
- `models/saved_models/geotermia_cnn_custom_best.keras` (mejor modelo)
- `models/saved_models/geotermia_cnn_custom_final.keras` (último)
- `logs/history_custom.json`
- `logs/tensorboard/` (visualizaciones)

**Visualizar entrenamiento:**
```bash
tensorboard --logdir=logs
```

---

#### **Paso 5: Evaluar Modelo**

```bash
python scripts/evaluate_model.py
```

**¿Qué hace?**
- Carga modelo entrenado
- Realiza predicciones en conjunto de test
- Calcula métricas con **Bootstrap CI 95%** (2,000 iteraciones)

**Métricas calculadas:**
- Accuracy (Exactitud) + IC 95%
- Precision (Precisión) + IC 95%
- Recall (Sensibilidad) + IC 95%
- F1-Score + IC 95%
- ROC AUC + IC 95%
- MCC (Matthews Correlation Coefficient) + IC 95%
- Confusion Matrix
- Classification Report

**Salidas:**
- `results/metrics/evaluation_metrics.json`
- `results/metrics/metrics_table.csv` ← **Para la tesis**

---

#### **Paso 6: Generar Visualizaciones**

```bash
python scripts/visualize_results.py
```

**¿Qué hace?**
- Genera gráficos profesionales de alta resolución (300 DPI)

**Visualizaciones generadas:**
- **Training History** (Loss y Accuracy)
- **Confusion Matrix** (Matriz de confusión)
- **ROC Curve** (Curva ROC con AUC)
- **Metrics Comparison** (Comparación de métricas)

**Salidas:**
- `results/figures/*.png` ← **Listas para incluir en tesis**

---

#### **Paso 7: Hacer Predicciones**

**Predicción en una imagen:**
```bash
python scripts/predict.py --image data/raw/Nevado_del_Ruiz.tif
```

**Predicción en múltiples imágenes:**
```bash
python scripts/predict.py --folder data/raw --output results/predictions.json
```

**Con modelo específico:**
```bash
python scripts/predict.py --image test.tif --model models/saved_models/mi_modelo.keras
```

---

## Arquitectura del Modelo CNN

### Modelo Custom (Recomendado)

```python
GeotermiaCNN(
 input_shape=(224, 224, 7), # 7 bandas (5 TIR + temp + NDVI)
 num_classes=2, # Clasificación binaria
 dropout_rate=0.5, # Regularización
 l2_reg=0.0001 # Regularización L2
)
```

**Arquitectura:**
```
Input (224×224×7)
 ↓
Rescaling (normalización)
 ↓
Conv Block (32 filters, 7×7) + SpatialDropout2D + MaxPool
 ↓
Residual Block (64 filters) + SpatialDropout2D + MaxPool
 ↓
Residual Block (128 filters) + SpatialDropout2D + MaxPool
 ↓
Residual Block (256 filters) + SpatialDropout2D + MaxPool
 ↓
Residual Block (512 filters) + SpatialDropout2D
 ↓
Global Average Pooling
 ↓
Dense (256) + BatchNorm + Dropout
 ↓
Output (1 neuron, sigmoid)
```

### Optimizaciones Implementadas

| Técnica | Descripción | Beneficio |
|---------|-------------|----------|
| **SpatialDropout2D** | Dropout espacial para CNNs | Mejor regularización en imágenes |
| **AdamW** | Adam con weight decay correcto | Mejor generalización |
| **Label Smoothing** | Suavizado de etiquetas (0.1) | Reduce overfitting |
| **Cosine LR Decay** | Learning rate decae como coseno | Mejor convergencia |
| **Bootstrap CI 95%** | Intervalos de confianza para métricas | Validez estadística |
| **GroupShuffleSplit** | Anti-leakage geográfico | Sin contaminación train/test |

### Modelo con Transfer Learning (Alternativa)

```python
# Usar EfficientNetB0 pre-entrenado
model = create_geotermia_model(
 input_shape=(224, 224, 7),
 model_type='transfer_learning',
 base_model_name='efficientnet'
)
```

---

## Resultados

### Resultados v2 (Colombia, 200 imágenes base)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 91.45% |
| **Precision** | 92.31% |
| **Recall** | 93.75% |
| **F1-Score** | 93.02% |
| **ROC AUC** | 0.983 |

### Resultados v3 (Región Andina, 2,019 imágenes base)

> **Pendiente**: El modelo v3 aún no ha sido entrenado. Los resultados se actualizarán tras completar el entrenamiento con el dataset expandido de 22,209 imágenes.

### Visualizaciones para Tesis

Todos los gráficos se generan en alta resolución (300 DPI) listos para incluir en documentos académicos:

1. **Training History**: Evolución de Loss y Accuracy por época
2. **Confusion Matrix**: Matriz de confusión con heatmap
3. **ROC Curve**: Curva ROC con AUC score e IC 95%
4. **Metrics Comparison**: Comparación visual con intervalos de confianza

---

## Tecnologías y Librerías

### Deep Learning
- **TensorFlow 2.20+**: Framework de Deep Learning
- **Keras 3.x**: API de alto nivel (incluido en TensorFlow)
- **AdamW Optimizer**: Optimizador con weight decay correcto
- **Mixed Precision**: Entrenamiento optimizado
- **Label Smoothing**: Regularización para reducir overfitting

### Procesamiento Geoespacial
- **Google Earth Engine**: Plataforma de análisis geoespacial
- **geemap**: Interface Python para Earth Engine
- **rasterio**: Lectura/escritura de datos raster
- **geopandas**: Datos geoespaciales vectoriales

### Análisis y Visualización
- **NumPy**: Computación numérica
- **pandas**: Análisis de datos
- **matplotlib**: Visualización de datos
- **seaborn**: Visualizaciones estadísticas
- **Plotly**: Gráficos interactivos
- **scikit-learn**: Métricas de evaluación

### Interfaz Web
- **Streamlit**: Aplicación web interactiva
- **Folium**: Mapas interactivos
- **streamlit-folium**: Integración de mapas

### Desarrollo
- **Jupyter**: Notebooks interactivos
- **TensorBoard**: Visualización de entrenamiento
- **FPDF2**: Generación de reportes PDF

---

## Metodología

El proyecto sigue el proceso **CRISP-DM** (Cross-Industry Standard Process for Data Mining), adaptado para Deep Learning:

1. **Comprensión del negocio**: Necesidad de exploración geotérmica eficiente en Colombia
2. **Comprensión de los datos**: Análisis de imágenes ASTER GED (7 bandas: emisividad TIR + temperatura + NDVI)
3. **Preparación de datos**: Descarga paralela desde GEE, augmentación (×10), normalización, split con anti-leakage geográfico (GroupShuffleSplit)
4. **Modelado**: Arquitectura CNN con bloques residuales + Transfer Learning (EfficientNet/ResNet50V2)
5. **Evaluación**: Métricas con Bootstrap CI 95% (Accuracy, Precision, Recall, F1, ROC-AUC, MCC)
6. **Despliegue**: Interfaz web con Streamlit para predicción interactiva sobre Colombia

---

## Contribuciones Científicas

### Aporte Principal

Este proyecto contribuye a la **exploración geotérmica en Colombia** mediante:

1. **Automatización**: Sistema automatizado de identificación de zonas geotérmicas
2. **Eficiencia**: Reducción de costos de exploración preliminar
3. **Escalabilidad**: Análisis de grandes extensiones territoriales
4. **Precisión**: Modelo predictivo con métricas validadas

### Aplicaciones Potenciales

- **Transición energética**: Identificar recursos geotérmicos renovables
- **Diversificación de matriz energética**: Alternativa a fuentes convencionales
- **Planificación territorial**: Guiar estudios de exploración detallada
- **Investigación**: Base para estudios geotérmicos adicionales

---

## Documentación Adicional

- **[docs/RESUMEN_PROYECTO.md](docs/RESUMEN_PROYECTO.md)**: Vista general del proyecto y guía de monitoreo
- **[docs/MODELO_PREDICTIVO.md](docs/MODELO_PREDICTIVO.md)**: Documentación técnica completa del modelo CNN
- **[docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md](docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md)**: Catálogo de campos geotérmicos (4 países)
- **[docs/CONTEXTO_GEOTERMICO.md](docs/CONTEXTO_GEOTERMICO.md)**: Fundamentos teóricos geotérmicos
- **[docs/GUIA_PASO_A_PASO.md](docs/GUIA_PASO_A_PASO.md)**: Guía completa paso a paso (incluye entrenamiento externo con GPU)
- **[docs/CHANGELOG_V2.md](docs/CHANGELOG_V2.md)**: Registro de cambios v1→v2
- **[docs/ANALISIS_ENTRENAMIENTO.md](docs/ANALISIS_ENTRENAMIENTO.md)**: Análisis detallado por época
- **[docs/PREDICCIONES_PRUEBA.md](docs/PREDICCIONES_PRUEBA.md)**: Predicciones de prueba y análisis
- **[models/README.md](models/README.md)**: Documentación de modelos
- **[scripts/README.md](scripts/README.md)**: Guía de scripts
- **[results/README.md](results/README.md)**: Interpretación de resultados

---

## Cómo Contribuir

Aunque este es un proyecto de grado, se aceptan sugerencias y mejoras:

1. **Fork** el repositorio
2. Crea una **branch** para tu feature (`git checkout -b feature/MejoraNueva`)
3. **Commit** tus cambios (`git commit -m 'Agrega nueva funcionalidad'`)
4. **Push** a la branch (`git push origin feature/MejoraNueva`)
5. Abre un **Pull Request**

---

## Contacto

### Desarrollador 
**Cristian Camilo Vega Sánchez**
- Email: [ccvegas@academia.usbbog.edu.co](mailto:ccvegas@academia.usbbog.edu.co)
- GitHub: [@crisveg24](https://github.com/crisveg24)

### Co-autores
**Daniel Santiago Arévalo Rubiano**
- Email: [dsarevalor@academia.usbbog.edu.co](mailto:dsarevalor@academia.usbbog.edu.co)

**Yuliet Katerin Espitia Ayala**
- Email: [ykespitiaa@academia.usbbog.edu.co](mailto:ykespitiaa@academia.usbbog.edu.co)

**Laura Sophie Rivera Martín**
- Email: [lsriveram@academia.usbbog.edu.co](mailto:lsriveram@academia.usbbog.edu.co)

### Asesor Académico
**Prof. Yeison Eduardo Conejo Sandoval**
- Email: [yconejo@usbbog.edu.co](mailto:yconejo@usbbog.edu.co)

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025-2026 Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín

Se concede permiso para usar, copiar, modificar y distribuir este software...
```

---

## Agradecimientos

- **Universidad de San Buenaventura Bogotá** - Institución educativa
- **Google Earth Engine** - Plataforma de datos satelitales
- **NASA/METI** - Datos ASTER
- **Servicio Geológico Colombiano** - Referencias geotérmicas
- **Comunidad Open Source** - Librerías y herramientas

---

## Referencias

### Referencias Académicas

1. Alfaro, C. (2015). *Improvement of perception of the geothermal energy as a potential source of electrical energy in Colombia*. Proceedings World Geothermal Congress 2015, Melbourne, Australia.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.

3. Coolbaugh, M. F., Kratt, C., Fallacaro, A., Calvin, W. M., & Taranik, J. V. (2007). Detection of geothermal anomalies using Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) thermal infrared images at Bradys Hot Springs, Nevada, USA. *Remote Sensing of Environment*, 106(3), 350-359.

4. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

5. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. https://doi.org/10.1186/s40537-019-0197-0

6. Lahsen, A. (1982). Upper Cenozoic volcanism and tectonism in the Andes of northern Chile. *Earth-Science Reviews*, 18(3), 285-302.

7. Muñoz-Sáez, C., Manga, M., & Hurwitz, S. (2018). Hydrothermal discharge from the El Tatio basin, Atacama, Chile. *Journal of Volcanology and Geothermal Research*, 361, 25-35.

8. Bona, P., & Coviello, M. (2016). *Valoración y gobernanza de los proyectos geotérmicos en América del Sur*. CEPAL.

9. INGEMMET. (2014). *Inventario de fuentes termales del Perú*. Instituto Geológico, Minero y Metalúrgico del Perú.

10. Siebert, L., Simkin, T., & Kimberly, P. (2010). *Volcanoes of the World* (3rd ed.). Smithsonian Institution / University of California Press.

11. Alfaro, C., Ponce, P., Monsalve, M. L., & Ortiz, I. (2017). Geothermal potential of the Paipa volcano-hydrothermal system, Colombia. *Proceedings World Geothermal Congress 2015*.

12. Servicio Geológico Colombiano (SGC). (2019). *Mapa de amenaza volcánica del Volcán Nevado del Ruiz*.

13. Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18-27.

14. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.

### Dataset

- **ASTER GED AG100**: NASA/METI/AIST/Japan Spacesystems, University of Tokyo, and U.S./Japan ASTER Science Team. (2019). *ASTER Global Emissivity Dataset 100-meter V003*. NASA EOSDIS Land Processes DAAC.

---

## Citar Este Proyecto

### BibTeX

```bibtex
@misc{vega2026geotermia,
 author = {Vega Sánchez, Cristian Camilo and Arévalo Rubiano, Daniel Santiago and Espitia Ayala, Yuliet Katerin and Rivera Martín, Laura Sophie},
 title = {Modelo Predictivo Basado en Deep Learning y Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia},
 year = {2026},
 publisher = {Universidad de San Buenaventura Bogotá},
 url = {https://github.com/crisveg24/geotermia-colombia-cnn},
 note = {Proyecto de Grado - Ingeniería de Sistemas}
}
```

### APA 7th Edition

Vega Sánchez, C. C., Arévalo Rubiano, D. S., Espitia Ayala, Y. K., & Rivera Martín, L. S. (2026). *Modelo Predictivo Basado en Deep Learning y Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia* [Proyecto de Grado, Universidad de San Buenaventura Bogotá]. GitHub. https://github.com/crisveg24/geotermia-colombia-cnn

---

<p align="center">
 <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red" />
 <img src="https://img.shields.io/badge/For-Geothermal%20Research-green" />
 <img src="https://img.shields.io/badge/Colombia-2026-yellow" />
</p>

<p align="center">
 <strong>Universidad de San Buenaventura - Bogotá</strong><br>
 Facultad de Ingeniería<br>
 Programa de Ingeniería de Sistemas<br>
 2025-2026
</p>

---

Si este proyecto es de utilidad para su investigación, puede citarlo usando el formato BibTeX o APA indicado arriba.
