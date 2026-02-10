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

### Características Principales

- **Arquitectura CNN moderna** con bloques residuales (ResNet-inspired)
- **Transfer Learning** con EfficientNet y ResNet50V2
- **Mixed Precision Training** para optimizar rendimiento
- **Data Augmentation** avanzado con SpatialDropout2D
- **Métricas completas** (Accuracy, Precision, Recall, F1-Score, ROC AUC, PR-AUC)
- **Visualizaciones profesionales** para análisis de resultados
- **Pipeline completo** desde descarga de datos hasta predicción
- **Interfaz Web** con Streamlit para visualización interactiva
- **Optimizador AdamW** con regularización de pesos mejorada
- **Label Smoothing** para reducir overfitting
- **Cosine Learning Rate Decay** para mejor convergencia

---

## Equipo de Desarrollo

| Rol | Nombre | Email | GitHub |
|-----|--------|-------|--------|
| **Desarrollador** | Cristian Camilo Vega Sánchez | ccvegas@academia.usbbog.edu.co | [@crisveg24](https://github.com/crisveg24) |
| **Co-autor** | Daniel Santiago Arévalo Rubiano | dsarevalor@academia.usbbog.edu.co | - |
| **Co-autora** | Yuliet Katerin Espitia Ayala | ykespitiaa@academia.usbbog.edu.co | - |
| **Co-autora** | Laura Sophie Rivera Martin | lsriveram@academia.usbbog.edu.co | - |
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

El proyecto analiza zonas geotérmicas de interés en Colombia:

### Zonas de Alta Actividad Geotérmica

1. ** Nevado del Ruiz** (Tolima)
 - Coordenadas: -75.3222, 4.8951
 - Volcán activo con alta actividad geotérmica

2. ** Volcán Purácé** (Cauca)
 - Coordenadas: -76.4036, 2.3206
 - Sistema hidrotermal activo

3. ** Paipa-Iza** (Boyacá)
 - Coordenadas: -73.1124, 5.7781
 - Campo geotérmico con aguas termales

4. ** Volcán Galeras** (Nariño)
 - Volcán activo con manifestaciones geotérmicas

### Dataset Satelital

**ASTER Global Emissivity Dataset (AG100) V003**
- **Proveedor**: NASA/METI/AIST/Japan Spacesystems
- **Resolución espacial**: 100 metros
- **Bandas térmicas**: 10-14 (emisividad térmica infrarroja)
- **Cobertura**: Global
- **Fuente**: Google Earth Engine

---

## Arquitectura del Proyecto

```
geotermia-colombia-cnn/
│
├── app.py # Interfaz web Streamlit
├── config.py # Configuración centralizada de rutas
│
├── data/ # Datos del proyecto
│ ├── raw/ # Imágenes satelitales (.tif) + labels.csv
│ ├── augmented/ # Dataset augmentado (se genera)
│ └── processed/ # Datos procesados (.npy, se genera)
│
├── docs/ # Documentación técnica
│ ├── RESUMEN_PROYECTO.md # Vista general, estado y monitoreo
│ ├── MODELO_PREDICTIVO.md # Documentación técnica del modelo CNN
│ ├── REGISTRO_PROCESO.md # Bitácora cronológica del proyecto
│ ├── ANALISIS_ENTRENAMIENTO.md # Análisis de métricas por época
│ ├── MEJORAS_MODELO.md # Roadmap de optimizaciones
│ └── ENTRENAMIENTO_EXTERNO.md # Guía para entrenar con GPU
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
- **CUDA 11.8+** (opcional, para GPU)
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

#### **Paso 1: Preparar Dataset**

```bash
python scripts/prepare_dataset.py
```

**¿Qué hace?**
- Carga imágenes .tif desde `data/raw/`
- Normaliza y redimensiona a 224×224
- Crea splits train/validation/test (70/15/15)
- Genera archivos .npy para carga rápida
- Calcula pesos de clase para balanceo

**Salidas:**
- `data/processed/X_train.npy`
- `data/processed/y_train.npy`
- `data/processed/X_val.npy`, `y_val.npy`
- `data/processed/X_test.npy`, `y_test.npy`

---

#### **Paso 2: Entrenar Modelo CNN**

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

#### **Paso 3: Evaluar Modelo**

```bash
python scripts/evaluate_model.py
```

**¿Qué hace?**
- Carga modelo entrenado
- Realiza predicciones en conjunto de test
- Calcula métricas completas

**Métricas calculadas:**
- Accuracy (Exactitud)
- Precision (Precisión)
- Recall (Sensibilidad)
- F1-Score
- ROC AUC
- R² Score
- Confusion Matrix
- Classification Report

**Salidas:**
- `results/metrics/evaluation_metrics.json`
- `results/metrics/metrics_table.csv` ← **Para la tesis**

---

#### **Paso 4: Generar Visualizaciones**

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

#### **Paso 5: Hacer Predicciones**

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
 input_shape=(224, 224, 5), # 5 bandas térmicas ASTER
 num_classes=2, # Clasificación binaria
 dropout_rate=0.5, # Regularización
 l2_reg=0.0001 # Regularización L2
)
```

**Arquitectura:**
```
Input (224×224×5)
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
| **PR-AUC Métric** | AUC de Precision-Recall | Mejor para clases desbalanceadas |
| **F1-Score directo** | Métrica F1 durante entrenamiento | Monitoreo completo |

### Modelo con Transfer Learning (Alternativa)

```python
# Usar EfficientNetB0 pre-entrenado
model = create_geotermia_model(
 input_shape=(224, 224, 5),
 model_type='transfer_learning',
 base_model_name='efficientnet'
)
```

---

## Resultados Esperados

### Métricas de Rendimiento

| Métrica | Valor Esperado |
|---------|----------------|
| **Accuracy** | > 85% |
| **Precision** | > 80% |
| **Recall** | > 80% |
| **F1-Score** | > 80% |
| **ROC AUC** | > 0.90 |

### Visualizaciones para Tesis

Todos los gráficos se generan en alta resolución (300 DPI) listos para incluir en documentos académicos:

1. **Training History**: Evolución de Loss y Accuracy
2. **Confusion Matrix**: Matriz de confusión con heatmap
3. **ROC Curve**: Curva ROC con AUC score
4. **Metrics Comparison**: Comparación visual de todas las métricas

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

### Metodología Híbrida

El proyecto sigue una **metodología mixta** (cuantitativa + cualitativa) combinando:

1. **Scrum**: Gestión ágil del proyecto
2. **CRISP-DM**: Proceso estándar de minería de datos
 - Comprensión de datos
 - Preparación de datos
 - Modelado (CNN)
 - Evaluación
 - Despliegue
3. **KDD**: Knowledge Discovery in Databases
4. **Six Sigma (DMAIC)**: Control de calidad

### Enfoque Cuantitativo

- Análisis de grandes volúmenes de datos satelitales
- Métricas estadísticas rigurosas
- Evaluación objetiva del modelo

### Enfoque Cualitativo

- Interpretación de patrones geológicos
- Análisis de correlaciones geotérmicas
- Validación con conocimiento experto

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
- **[docs/REGISTRO_PROCESO.md](docs/REGISTRO_PROCESO.md)**: Bitácora cronológica de todas las fases
- **[docs/ENTRENAMIENTO_EXTERNO.md](docs/ENTRENAMIENTO_EXTERNO.md)**: Guía paso a paso para entrenar en GPU
- **[docs/MEJORAS_MODELO.md](docs/MEJORAS_MODELO.md)**: Roadmap de optimizaciones aplicadas y futuras
- **[docs/ANALISIS_ENTRENAMIENTO.md](docs/ANALISIS_ENTRENAMIENTO.md)**: Análisis detallado por época
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

**Laura Sophie Rivera Martin**
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
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martin

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

1. Alfaro, C. (2015). *Improvement of perception of the geothermal energy as a potential source of electrical energy in Colombia*. World Geothermal Congress.

2. González, M., Gómez, J., & Pérez, A. (2020). *Desafíos de la energía geotérmica en Colombia: Hacia la diversificación energética*. Revista de Energías Renovables, 11(3), 134-145.

3. Muñoz, Y., & Pérez, A. (2021). *Aplicación de redes neuronales para la identificación de zonas geotérmicas en Colombia*. Journal of Geothermal Energy, 23(5), 567-578.

4. Rodríguez, S., Gómez, F., & López, C. (2022). *Uso de redes neuronales convolucionales en la identificación de zonas geotérmicas en Colombia*. Geothermal Science Review, 9(2), 45-58.

5. Serrano, M. (2018). *La geotermia como alternativa para la matriz energética colombiana*. Revista de Energías Alternativas, 14(2), 112-120.

### Dataset

- **ASTER GED AG100**: NASA/METI/AIST/Japan Spacesystems, University of Tokyo, and U.S./Japan ASTER Science Team. (2019). *ASTER Global Emissivity Dataset 100-meter V003*. NASA EOSDIS Land Processes DAAC.

---

## Citar Este Proyecto

### BibTeX

```bibtex
@misc{vega2026geotermia,
 author = {Vega Sánchez, Cristian Camilo and Arévalo Rubiano, Daniel Santiago and Espitia Ayala, Yuliet Katerin and Rivera Martin, Laura Sophie},
 title = {Modelo Predictivo Basado en Deep Learning y Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia},
 year = {2026},
 publisher = {Universidad de San Buenaventura Bogotá},
 url = {https://github.com/crisveg24/geotermia-colombia-cnn},
 note = {Proyecto de Grado - Ingeniería de Sistemas}
}
```

### APA 7th Edition

Vega Sánchez, C. C., Arévalo Rubiano, D. S., Espitia Ayala, Y. K., & Rivera Martin, L. S. (2026). *Modelo Predictivo Basado en Deep Learning y Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia* [Proyecto de Grado, Universidad de San Buenaventura Bogotá]. GitHub. https://github.com/crisveg24/geotermia-colombia-cnn

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

** Si este proyecto te parece útil, considera darle una estrella en GitHub!**
