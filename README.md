# Modelo Predictivo de Potencial Geotérmico — CNN + ASTER (v3)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Universidad](https://img.shields.io/badge/Universidad-San%20Buenaventura%20Bogot%C3%A1-red.svg)](https://www.usbbog.edu.co/)

<p align="center">
  <img src="https://img.shields.io/badge/EfficientNetB0-Transfer%20Learning-blueviolet" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Geospatial-success" />
  <img src="https://img.shields.io/badge/v3-Final-brightgreen" />
</p>

---

## Descripción

**Proyecto de Grado — Universidad de San Buenaventura, Bogotá**

Modelo de **Deep Learning (CNN)** para la identificación automatizada de zonas con alto potencial geotérmico en la **Región Andina** (Colombia, Ecuador, Perú y Chile) mediante imágenes satelitales térmicas del sensor **NASA ASTER**. La versión final (v3) utiliza **EfficientNetB0** con Transfer Learning y un Channel Adapter personalizado para procesar las 7 bandas ASTER.

> **Herramienta de screening geotérmico:** Actúa como filtro automatizado que produce una probabilidad (0–100 %) de potencial geotérmico para cualquier punto de Colombia. Su objetivo es priorizar zonas para exploración detallada, no confirmar recursos explotables.

> **Dataset multinacional:** El modelo se entrena con datos de 4 países andinos que comparten el contexto geológico del Cinturón de Fuego del Pacífico. La aplicación web y las conclusiones se aplican exclusivamente a Colombia.

---

## Resultados v3 (versión final)

| Métrica | Valor | IC 95 % |
|---------|:-----:|:-------:|
| **Accuracy** | 92.28 % | ± ~1.0 % |
| **Precision** | 91.27 % | ± ~1.2 % |
| **Recall** | 93.17 % | ± ~1.1 % |
| **F1-Score** | 92.21 % | ± ~0.9 % |
| **ROC AUC** | 0.9737 | ± ~0.004 |
| **MCC** | 0.8458 | ± ~0.02 |

- **Test set:** 3,619 imágenes (nunca vistas durante entrenamiento)
- **Bootstrap CI:** 2,000 iteraciones
- **0 % data leakage** — GroupShuffleSplit por zona geográfica base

### Evolución del proyecto

| Versión | Dataset | Imágenes (aug.) | Arquitectura | Accuracy |
|:-------:|:-------:|:----------------:|:-------------|:--------:|
| v1 | Colombia (85 base) | 935 | ResNet-inspired custom | 68.43 % |
| v2 | Colombia (200 base) | 2,200 | ResNet-inspired custom | 91.45 % |
| **v3** | **Región Andina (2,019 base)** | **22,209** | **EfficientNetB0 + Channel Adapter** | **92.28 %** |

---

## Arquitectura del Modelo

```
Input (224×224×7)          ← 7 bandas ASTER
      ↓
Channel Adapter            ← Conv2D 1×1 (7→16→3) + BatchNorm + ReLU
      ↓
EfficientNetB0 (ImageNet)  ← Backbone pre-entrenado, fine-tuned últimas 39 capas
      ↓
Global Average Pooling
      ↓
Dense (256) + Dropout 0.3
      ↓
Output (1, sigmoid)        ← Probabilidad geotérmica
```

**Parámetros totales:** 4,396,112 | **Entrenables (fase 2):** 3,871,857

### Estrategia de entrenamiento (2 fases)

| | Fase 1 (warm-up) | Fase 2 (fine-tuning) |
|-|:-:|:-:|
| **Épocas** | 30 | 50 |
| **Backbone** | Congelado | Últimas 39 capas desbloqueadas |
| **Learning Rate** | 1×10⁻³ | 1×10⁻⁴ (Cosine Decay) |
| **Regularización** | MixUp α=0.2, Label Smoothing 0.1 | Igual + AdamW weight decay |
| **Precisión** | mixed_float16 | mixed_float16 |

Ver documentación completa en [docs/MODELO_TECNICO.md](docs/MODELO_TECNICO.md).

---

## Dataset

**ASTER Global Emissivity Dataset (AG100) V003** — NASA/METI

| País | Positivas | Negativas | Total |
|------|:-:|:-:|:-:|
| Colombia | ~375 | ~89 | ~464 |
| Ecuador | ~170 | ~180 | ~350 |
| Perú | ~160 | ~180 | ~340 |
| Chile | ~280 | ~575 | ~855 |
| **Total** | **997** | **1,022** | **2,019** |

- **7 bandas:** emisividad TIR (10–14), temperatura superficial, NDVI
- **Resolución:** 100 m (90 m/px en GEE), buffer 5 km por zona
- **Augmentación:** ×10 variaciones → **22,209 imágenes** totales
- **Splits:** Train 15,037 / Val 3,553 / Test 3,619 — con **anti-leakage geográfico**

---

## Equipo

| Rol | Nombre | Email |
|-----|--------|-------|
| **Desarrollador** | Cristian Camilo Vega Sánchez | ccvegas@academia.usbbog.edu.co |
| **Co-autor** | Daniel Santiago Arévalo Rubiano | dsarevalor@academia.usbbog.edu.co |
| **Co-autora** | Yuliet Katerin Espitia Ayala | ykespitiaa@academia.usbbog.edu.co |
| **Co-autora** | Laura Sophie Rivera Martín | lsriveram@academia.usbbog.edu.co |
| **Asesor** | Prof. Yeison Eduardo Conejo Sandoval | yconejo@usbbog.edu.co |

**Institución:** Universidad de San Buenaventura — Sede Bogotá  
**Programa:** Ingeniería de Sistemas (Pregrado)  
**Periodo:** 2025–2026

---

## Inicio rápido

```bash
# 1. Clonar
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn

# 2. Entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # Linux / macOS

# 3. Dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opcional) Disco externo para datos
$env:GEOTERMIA_DATA_ROOT = "E:\geotermia_datos"

# 5. Pipeline completo
python scripts/download_dataset.py       # Descarga 2,019 imágenes ASTER
python scripts/augment_full_dataset.py   # Augmentación → 22,209 imágenes
python scripts/prepare_dataset.py        # Normalización + splits .npy
python scripts/train_model_v7.py         # Entrenamiento 2 fases (GPU recomendada)
python scripts/evaluate_model.py         # Evaluación + Bootstrap CI 95%

# 6. Interfaz web
streamlit run app.py                     # → http://localhost:8501
```

Para la guía detallada paso a paso, ver [docs/GUIA_REPRODUCCION.md](docs/GUIA_REPRODUCCION.md).

---

## Interfaz Web (Streamlit)

- **Predicción por coordenadas:** Ingresa latitud/longitud o haz clic en el mapa interactivo
- **Métricas del modelo:** Gráficos interactivos de rendimiento
- **Arquitectura:** Diagrama visual de la red neuronal
- **Acerca de:** Equipo e información del proyecto

```bash
streamlit run app.py
```

---

## Estructura del repositorio

```
geotermia-colombia-cnn/
├── app.py                          # Interfaz web Streamlit
├── config.py                       # Configuración centralizada
├── setup.py                        # Verificación del entorno
├── requirements.txt                # Dependencias Python
├── LICENSE                         # MIT License
│
├── models/
│   ├── cnn_geotermia.py            # Arquitectura CNN (v2 + v3)
│   └── saved_models/               # Modelos entrenados (.keras)
│       └── geotermia_v7_phase2_best.keras
│
├── scripts/
│   ├── download_dataset.py         # Descarga ASTER GED (3 hilos)
│   ├── augment_full_dataset.py     # Augmentación (×10)
│   ├── prepare_dataset.py          # Normalización global + splits
│   ├── train_model_v7.py           # Entrenamiento v3 (2 fases, GPU)
│   ├── train_model.py              # Entrenamiento v2 (CPU)
│   ├── evaluate_model.py           # Evaluación + Bootstrap CI
│   ├── resplit_data.py             # Re-partición anti-leakage
│   ├── predict.py                  # Predicción CLI
│   ├── _check_leakage.py           # Verificación fuga de datos
│   └── _check_splits.py            # Verificación de splits
│
├── data/
│   ├── raw/                        # 2,019 imágenes .tif originales
│   ├── augmented/                  # 22,209 imágenes augmentadas
│   └── processed/                  # .npy particionados + band_stats
│
├── docs/
│   ├── TESIS_CONTENIDO_APA.md      # Contenido completo de la tesis (APA 7ª ed.)
│   ├── MODELO_TECNICO.md           # Documentación técnica integral del modelo
│   └── GUIA_REPRODUCCION.md        # Guía paso a paso para reproducir el proyecto
│
├── results/                        # Métricas y figuras (300 DPI)
├── logs/                           # Logs de entrenamiento
└── notebooks/                      # Notebooks de exploración
```

---

## Tecnologías principales

| Categoría | Tecnologías |
|-----------|-------------|
| **Deep Learning** | TensorFlow 2.20, Keras 3.x, EfficientNetB0, AdamW, Mixed Precision |
| **Geoespacial** | Google Earth Engine, rasterio, geemap |
| **Análisis** | NumPy, pandas, scikit-learn, matplotlib, seaborn, Plotly |
| **Interfaz** | Streamlit, Folium |

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [docs/TESIS_CONTENIDO_APA.md](docs/TESIS_CONTENIDO_APA.md) | Contenido completo de la tesis en formato APA 7ª edición |
| [docs/MODELO_TECNICO.md](docs/MODELO_TECNICO.md) | Documentación técnica integral: arquitectura, entrenamiento, métricas, dataset, contexto geotérmico, catálogo de campos, historial de bugs (34 correcciones v1→v3) |
| [docs/GUIA_REPRODUCCION.md](docs/GUIA_REPRODUCCION.md) | Guía paso a paso para reproducir el pipeline completo desde cero |

---

## Licencia

MIT License — Ver [LICENSE](LICENSE).

```
Copyright (c) 2025-2026 Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín
```

---

## Agradecimientos

- **Universidad de San Buenaventura Bogotá** — Institución educativa
- **Google Earth Engine** — Plataforma de datos satelitales
- **NASA/METI** — Datos ASTER
- **Servicio Geológico Colombiano** — Referencias geotérmicas

---

*Universidad de San Buenaventura — Bogotá | Proyecto de Grado 2025–2026*

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
