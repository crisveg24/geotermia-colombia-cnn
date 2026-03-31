# Guía de Reproducción — CNN Geotermia Región Andina (v3)

> **Propósito:** Permitir que cualquier persona reproduzca el pipeline completo
> del proyecto, desde cero hasta un modelo entrenado y la interfaz Streamlit funcionando.
>
> **Última actualización:** 7 de marzo de 2026
> **Universidad de San Buenaventura — Bogotá**

---

## 1. Requisitos previos

| Componente | Versión mínima | Notas |
|------------|:--------------:|-------|
| Python | 3.10.x – 3.12.x | 3.10+ requerido por TensorFlow |
| pip | 23+ | `python -m pip install --upgrade pip` |
| Git | 2.x | Para clonar el repositorio |
| Cuenta Google | — | Para Google Earth Engine (solo si vas a descargar imágenes) |
| Espacio en disco | ~40 GB | Dataset completo (~30 GB procesado + raw + augmented) |
| GPU (recomendada) | NVIDIA + CUDA 12 | Entrenamiento ~45× más rápido. Sin GPU funciona en CPU |

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn
```

---

## 3. Crear y activar entorno virtual

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verificar:**
```bash
python --version   # 3.10.x – 3.12.x
pip --version      # Debe apuntar al .venv
```

---

## 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verificar TensorFlow:**
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Debe imprimir 2.20.0 o superior
```

**Verificar GPU (opcional):**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Con GPU NVIDIA + CUDA → mostrará la lista de GPUs
# Sin GPU → mostrará [] (usará CPU)
```

---

## 5. Configurar Google Earth Engine

Solo necesario si vas a **descargar** imágenes nuevas. Si ya tienes las imágenes, salta al paso 6.

```bash
python -c "import ee; ee.Authenticate()"
# Se abre el navegador. Inicia sesión con Google y copia el token.

python -c "import ee; ee.Initialize(project='alpine-air-469115-f0'); print('OK')"
```

---

## 6. Configurar origen de datos

### Opción A — Datos locales (por defecto)

Los scripts buscan las imágenes en `data/raw/`, `data/augmented/`, `data/processed/`.

### Opción B — Disco duro externo

Útil cuando no tienes espacio local o quieres transportar el dataset.

**Estructura esperada:**
```
E:\geotermia_datos\       ← (o la letra que tenga tu disco)
├── raw\
│   ├── positive\         ← .tif geotérmicos
│   ├── negative\         ← .tif control
│   └── labels.csv
├── augmented\
│   ├── positive\         ← .tif augmentados
│   ├── negative\
│   └── labels.csv
└── processed\
    ├── X_train_part00.npy … X_train_part30.npy
    ├── y_train.npy
    ├── X_val_part00.npy … X_val_part07.npy
    ├── y_val.npy
    ├── X_test_part00.npy … X_test_part07.npy
    ├── y_test.npy
    ├── band_stats_v3.json
    └── split_info.json
```

**Configurar la ruta:**
```powershell
# Windows PowerShell (sesión actual)
$env:GEOTERMIA_DATA_ROOT = "E:\geotermia_datos"

# Windows PowerShell (permanente)
[Environment]::SetEnvironmentVariable("GEOTERMIA_DATA_ROOT", "E:\geotermia_datos", "User")
```

```bash
# Linux / macOS
export GEOTERMIA_DATA_ROOT="/media/usuario/disco/geotermia_datos"
```

**Verificar:** `python config.py` imprime un resumen de la configuración detectada.

---

## 7. Pipeline completo (paso a paso)

### Paso 1: Descargar imágenes ASTER

```bash
python scripts/download_dataset.py
```

- Descarga **2.019 imágenes** ASTER GED de la Región Andina (CO, EC, PE, CL)
- Descarga paralela con 3 hilos, expansión por grilla 3×3
- Tiempo: ~30–60 minutos
- **Salida:** `data/raw/` → 2.019 archivos .tif (~115 MB)

### Paso 2: Augmentar dataset

```bash
python scripts/augment_full_dataset.py
```

- Aplica 10 variaciones por imagen (rotaciones, flips, brillo, contraste, ruido, gamma)
- Tiempo: ~30–50 minutos
- **Salida:** `data/augmented/` → 22.209 imágenes .tif (~7,9 GB)

### Paso 3: Preparar dataset

```bash
python scripts/prepare_dataset.py
```

- Carga .tif, filtra NoData (-9999), redimensiona a 224×224
- Normaliza z-score **global por banda** (algoritmo de Welford)
- Division anti-fuga: GroupShuffleSplit por zona geográfica base
- **Salida:** `data/processed/` → archivos .npy particionados (~29,8 GB)
  - Train: 15.037 imágenes (31 partes)
  - Val: 3.553 imágenes (8 partes)
  - Test: 3.619 imágenes (8 partes)
  - `band_stats_v3.json` — estadísticas globales por banda

### Paso 4: Entrenar modelo

```bash
# Modelo v3 (EfficientNetB0 + Channel Adapter, 2 fases)
python scripts/train_model_v7.py
```

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | EfficientNetB0 + Channel Adapter (7→16→3) |
| Fase 1 | 30 épocas, backbone congelado, LR=1×10⁻³ |
| Fase 2 | 50 épocas, fine-tuning últimas 39 capas, LR=1×10⁻⁴ |
| Batch size | 32 |
| Mixed Precision | float16 (automático en GPU) |
| Regularización | MixUp α=0,2, Label Smoothing 0,1, AdamW, Dropout |

**Tiempos estimados:**

| Hardware | Por época | Total (~80 épocas) |
|----------|:---------:|:-------------------:|
| CPU (i5/i7) | ~5–8 min | ~7–10 horas |
| GPU GTX 1060+ | 30–60 s | ~40–80 min |
| GPU RTX 4070 | ~20 s | ~27 min |

**Salida:**
```
models/saved_models/
├── geotermia_v7_phase2_best.keras  ← mejor modelo (usar este)
└── geotermia_v7_final.keras        ← modelo al final
logs/
├── geotermia_v7_phase1.csv
├── geotermia_v7_phase2.csv
└── history_v7.json
```

### Paso 5: Evaluar modelo

```bash
python scripts/evaluate_model.py
```

- Evalúa sobre el conjunto de test (3.619 imágenes)
- Calcula métricas con **Bootstrap CI 95 %** (2.000 iteraciones)
- **Salida:** `results/metrics/evaluation_metrics.json`, `results/figures/`

### Paso 6: Ejecutar interfaz web

```bash
streamlit run app.py
```

Abrir **http://localhost:8501** en el navegador.

Secciones:
1. **Inicio** — Descripción del proyecto
2. **Predicción** — Ingresa coordenadas o haz clic en el mapa para predecir
3. **Métricas** — Visualización de rendimiento del modelo
4. **Arquitectura** — Diagrama de la red neuronal
5. **Acerca de** — Equipo e información

---

## 8. Verificar integridad de datos

### Verificar que no hay fuga de datos
```bash
python scripts/_check_leakage.py
```
Debe reportar **0 % leakage** entre todos los subconjuntos.

### Verificar distribución de splits
```bash
python scripts/_check_splits.py
```
Verifica balance de clases y distribución por zonas en cada split.

---

## 9. Solución de problemas

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'ee'` | `pip install earthengine-api` |
| `ee.Initialize()` falla | Ejecutar `ee.Authenticate()` primero |
| Error de memoria (OOM) en prepare_dataset | Reducir tamaño de partición en `prepare_dataset.py` |
| Error de memoria en entrenamiento | Reducir `batch_size` a 16 |
| `GEOTERMIA_DATA_ROOT` no detectado | Verificar con `echo $env:GEOTERMIA_DATA_ROOT` (PowerShell) |
| Modelo no carga (quantization_config) | Versión de Keras local difiere de la de entrenamiento. Ver monkey-patch en `app.py` |
| GPU no detectada por TensorFlow | Verificar instalación de CUDA/cuDNN. En Windows nativo, TF 2.11+ no soporta GPU |

---

## 10. Estructura del repositorio

```
geotermia-colombia-cnn/
├── app.py                    # Interfaz web Streamlit
├── config.py                 # Configuración centralizada
├── setup.py                  # Verificación del entorno
├── requirements.txt          # Dependencias Python
├── LICENSE                   # MIT License
│
├── models/
│   ├── cnn_geotermia.py      # Arquitectura CNN v2 + v3
│   └── saved_models/         # Modelos entrenados (.keras)
│
├── scripts/
│   ├── download_dataset.py   # Descarga ASTER GED (3 hilos)
│   ├── augment_full_dataset.py # Augmentación (×10)
│   ├── prepare_dataset.py    # Preparación + normalize global
│   ├── train_model_v7.py     # Entrenamiento v3 (2 fases, GPU)
│   ├── train_model.py        # Entrenamiento v2 (CPU)
│   ├── evaluate_model.py     # Evaluación + Bootstrap CI
│   ├── resplit_data.py       # Re-partición anti-leakage
│   ├── predict.py            # Predicción CLI
│   ├── _check_leakage.py     # Verificación de fuga de datos
│   └── _check_splits.py      # Verificación de splits
│
├── data/
│   ├── raw/                  # 2.019 imágenes ASTER .tif
│   ├── augmented/            # 22.209 imágenes augmentadas
│   └── processed/            # .npy particionados + band_stats
│
├── docs/
│   ├── TESIS_CONTENIDO_APA.md # Contenido de la tesis (APA 7ª ed.)
│   ├── MODELO_TECNICO.md      # Documentación técnica completa
│   └── GUIA_REPRODUCCION.md   # Esta guía
│
├── results/                  # Métricas y figuras
├── logs/                     # Logs de entrenamiento
└── notebooks/                # Notebooks de exploración
```

---

*Universidad de San Buenaventura — Bogotá | Proyecto de Grado 2025–2026*
