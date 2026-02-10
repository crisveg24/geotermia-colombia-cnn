# GUÍA PASO A PASO — Geotermia Colombia CNN

> **Propósito**: Este documento permite que **cualquier IA o persona** reproduzca
> el pipeline completo del proyecto: desde cero hasta un modelo entrenado y la
> interfaz Streamlit funcionando.
>
> **Última actualización**: 9 de febrero de 2026 
> **Autores**: Cristian Vega, Daniel Arévalo, Yuliet Espitia, Laura Rivera 
> **Universidad de San Buenaventura — Bogotá**

---

## ÍNDICE

1. [Requisitos previos](#1--requisitos-previos)
2. [Clonar el repositorio](#2--clonar-el-repositorio)
3. [Crear y activar entorno virtual](#3--crear-y-activar-entorno-virtual)
4. [Instalar dependencias](#4--instalar-dependencias)
5. [Configurar Google Earth Engine](#5--configurar-google-earth-engine)
6. [Configurar origen de datos (local vs disco externo)](#6--configurar-origen-de-datos)
7. [Descargar imágenes ASTER](#7--descargar-imágenes-aster)
8. [Augmentación de datos](#8--augmentación-de-datos)
9. [Preparar dataset (.npy)](#9--preparar-dataset)
10. [Entrenar el modelo CNN](#10--entrenar-el-modelo-cnn)
11. [Evaluar el modelo](#11--evaluar-el-modelo)
12. [Ejecutar la interfaz Streamlit](#12--ejecutar-la-interfaz-streamlit)
13. [Solución de problemas](#13--solución-de-problemas)

---

## 1 · Requisitos previos

| Componente | Versión mínima | Notas |
|-----------------|----------------|-------|
| Python | 3.10.x | 3.10 o 3.11. NO 3.12+ (compatibilidad TF) |
| pip | 23+ | `python -m pip install --upgrade pip` |
| Git | 2.x | Para clonar el repo |
| Cuenta Google | — | Para Google Earth Engine |
| Espacio disco | ~5 GB | Dataset completo + modelo |
| GPU (opcional) | NVIDIA + CUDA 12 | Acelera el entrenamiento. Sin GPU funciona en CPU |

---

## 2 · Clonar el repositorio

```bash
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn
```

---

## 3 · Crear y activar entorno virtual

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verificar**:
```bash
python --version # Debe decir 3.10.x o 3.11.x
pip --version # Debe apuntar al .venv
```

---

## 4 · Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verificar TensorFlow**:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Debe imprimir 2.20.0 o superior
```

**Verificar GPU** (opcional):
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Si tienes GPU NVIDIA con CUDA → mostrará la lista de GPUs
# Si no → mostrará [] (usará CPU, más lento pero funcional)
```

---

## 5 · Configurar Google Earth Engine

Este paso solo es necesario si vas a **descargar** imágenes nuevas.
Si ya tienes las imágenes en un disco externo, salta al [paso 6](#6--configurar-origen-de-datos).

```bash
# 1. Autenticarse
python -c "import ee; ee.Authenticate()"
# Se abrirá un navegador. Inicia sesión con tu cuenta Google y copia el token.

# 2. Verificar
python -c "import ee; ee.Initialize(project='alpine-air-469115-f0'); print('OK')"
```

> **Nota**: El proyecto GCP es `alpine-air-469115-f0`. Si usas otro proyecto,
> edita la línea correspondiente en `scripts/download_dataset.py`.

---

## 6 · Configurar origen de datos

### Opción A — Datos locales (por defecto)

No necesitas hacer nada especial. Los scripts buscarán las imágenes en:
```
geotermia-colombia-cnn/
 data/
 raw/positive/ ← .tif geotérmicos
 raw/negative/ ← .tif control
 augmented/ ← .tif aumentados
 processed/ ← .npy listos para entrenar
```

### Opción B — Datos en disco duro externo (USB, SSD, etc.)

Esto es útil cuando:
- No tienes espacio en el equipo local
- Vas a entrenar en los computadores de la universidad
- Quieres transportar el dataset sin re-descargar (~5 GB)

**Estructura esperada en el disco externo:**
```
D:\geotermia_datos\ ← (o E:\, F:\, la letra que tenga tu disco)
 raw\
 positive\ ← archivos .tif descargados (zonas geotérmicas)
 negative\ ← archivos .tif descargados (zonas control)
 labels.csv
 augmented\
 positive\ ← .tif con augmentación
 negative\
 labels.csv
 processed\
 X_train.npy
 y_train.npy
 X_val.npy
 y_val.npy
 X_test.npy
 y_test.npy
 split_info.json
```

**Configurar la ruta** (elige UNA de estas opciones):

#### Opción B.1 — Variable de entorno (recomendada)

```powershell
# Windows PowerShell (sesión actual)
$env:GEOTERMIA_DATA_ROOT = "D:\geotermia_datos"

# Windows PowerShell (permanente para el usuario)
[Environment]::SetEnvironmentVariable("GEOTERMIA_DATA_ROOT", "D:\geotermia_datos", "User")
```

```bash
# Linux / macOS
export GEOTERMIA_DATA_ROOT="/media/usuario/disco_externo/geotermia_datos"
```

#### Opción B.2 — Archivo .env en la raíz del proyecto

Crea un archivo `.env` en la raíz del proyecto:
```
GEOTERMIA_DATA_ROOT=D:\geotermia_datos
```

> Los scripts leen esta variable automáticamente desde `config.py`.

**Verificar configuración**:
```bash
python config.py
```

Esto imprimirá un resumen mostrando:
- Si detectó el disco externo
- Cuántas imágenes hay en cada carpeta
- Si faltan directorios

Ejemplo de salida:
```
============================================================
CONFIGURACIÓN DEL PROYECTO GEOTERMIA CNN
============================================================
 Fuente de datos : env $GEOTERMIA_DATA_ROOT
 Disco externo : SÍ
 Data root : D:\geotermia_datos
 
 raw/positive → 45 .tif, 0 .npy
 raw/negative → 43 .tif, 0 .npy
 augmented → 2728 .tif, 0 .npy
 processed → 0 .tif, 6 .npy
============================================================
```

### Preparar un disco externo con imágenes existentes

Si ya descargaste las imágenes en otro equipo:

```powershell
# 1. Crear estructura en el disco externo
mkdir D:\geotermia_datos\raw\positive
mkdir D:\geotermia_datos\raw\negative
mkdir D:\geotermia_datos\augmented\positive
mkdir D:\geotermia_datos\augmented\negative
mkdir D:\geotermia_datos\processed

# 2. Copiar imágenes desde el proyecto local
xcopy /E data\raw\positive D:\geotermia_datos\raw\positive\
xcopy /E data\raw\negative D:\geotermia_datos\raw\negative\
copy data\raw\labels.csv D:\geotermia_datos\raw\

# 3. Si ya tienes augmentados y procesados, copiarlos también
xcopy /E data\augmented D:\geotermia_datos\augmented\
xcopy /E data\processed D:\geotermia_datos\processed\
```

---

## 7 · Descargar imágenes ASTER

> **¿Ya tienes las imágenes?** Si las tienes en un disco externo o en `data/raw/`,
> salta al [paso 8](#8--augmentación-de-datos).

```bash
python scripts/download_dataset.py
```

- Descarga ~88 imágenes ASTER GED (5 bandas térmicas de emisividad)
- 45 zonas geotérmicas (volcanes: Ruiz, Puracé, Galeras, Paipa-Iza, etc.)
- 43 zonas de control (Llanos, Amazonía, Costa Caribe, etc.)
- Resolución: 90 m/pixel, radio 5 km por zona
- Tiempo estimado: 15-30 minutos
- Requiere conexión a internet y auth de Earth Engine

**Salida**:
```
data/raw/positive/ ← ~45 archivos .tif
data/raw/negative/ ← ~43 archivos .tif
data/raw/labels.csv ← archivo de etiquetas
```

> Si configuraste `GEOTERMIA_DATA_ROOT`, las imágenes se guardarán directamente 
> en el disco externo.

---

## 8 · Augmentación de datos

```bash
python scripts/augment_full_dataset.py
```

- Aplica 30 técnicas de augmentación a cada imagen
- Genera ~31× más imágenes (original + 30 variaciones)
- Técnicas: rotación, flip, brillo, contraste, ruido, blur, crop, combinaciones
- Tiempo estimado: 10-20 minutos

**Salida** (si ~88 originales):
```
data/augmented/positive/ ← ~1,395 archivos .tif
data/augmented/negative/ ← ~1,333 archivos .tif
data/augmented/labels.csv
```

---

## 9 · Preparar dataset

```bash
python scripts/prepare_dataset.py
```

- Carga todos los .tif de `data/augmented/`
- Redimensiona a 224×224 píxeles
- Normaliza por banda (z-score: media=0, std=1)
- Divide en train/val/test (70/15/15, estratificado)
- Calcula pesos de clase para balanceo
- Tiempo estimado: 5-15 minutos

**Salida**:
```
data/processed/
 X_train.npy, y_train.npy ← ~70% de las imágenes
 X_val.npy, y_val.npy ← ~15%
 X_test.npy, y_test.npy ← ~15%
 split_info.json ← metadatos del split
```

> Los archivos .npy pueden ser **muy grandes** (~2-4 GB). Si usas disco externo,
> estos también se guardarán ahí.

---

## 10 · Entrenar el modelo CNN

```bash
python scripts/train_model.py
```

### Configuración del entrenamiento

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| Input shape | 224×224×5 | 5 bandas ASTER (emisividad térmica) |
| Batch size | 32 | Reducir a 16 si hay poca RAM/VRAM |
| Épocas | 100 máx | EarlyStopping con patience=15 |
| Optimizer | AdamW | lr=0.001, weight_decay=0.0001 |
| Label Smoothing | 0.1 | Regularización |
| Mixed Precision | Sí | float16 para GPU (auto-desactiva en CPU) |
| Data Augmentation | Sí | RandomFlip, Rotation, Zoom, Translation, Contrast |

### Tiempo estimado

| Hardware | Tiempo por época | Total (~30-50 épocas) |
|----------|-----------------|----------------------|
| CPU (i5/i7) | 3-5 min | 2-4 horas |
| GPU GTX 1060+ | 15-30 seg | 15-30 min |
| GPU RTX 3060+ | 5-15 seg | 5-15 min |

### Callbacks automáticos

- **ModelCheckpoint**: Guarda el mejor modelo según `val_loss`
- **EarlyStopping**: Para si val_loss no mejora en 15 épocas
- **ReduceLROnPlateau**: Reduce learning rate si val_loss se estanca 5 épocas
- **TensorBoard**: Logs para visualización
- **CSVLogger**: Historial en CSV

**Salida**:
```
models/saved_models/
 geotermia_cnn_custom_best.keras ← mejor modelo (usar este)
 geotermia_cnn_custom_final.keras ← modelo al final del entrenamiento
logs/
 history_custom.json ← historial de métricas
 training_YYYYMMDD_HHMMSS/ ← logs de TensorBoard
```

### Monitorear con TensorBoard (opcional)

```bash
tensorboard --logdir=logs
# Abrir http://localhost:6006 en el navegador
```

---

## 11 · Evaluar el modelo

```bash
python scripts/evaluate_model.py
```

- Evalúa el mejor modelo sobre el conjunto de test
- Genera matriz de confusión, curvas ROC/PR, métricas detalladas
- **Salida**: `results/` con gráficas y métricas

---

## 12 · Ejecutar la interfaz Streamlit

```bash
streamlit run app.py --server.headless true
```

- Abrir **http://localhost:8501** en el navegador
- 5 secciones: Inicio, Predicción (mapa interactivo), Métricas, Arquitectura, Acerca de
- Permite ingresar coordenadas y obtener predicción de potencial geotérmico

> Si el modelo no está entrenado, la app mostrará "Modelo no disponible" pero
> seguirá funcionando con las demás secciones.

---

## 13 · Solución de problemas

### Error: `No module named streamlit`
```bash
pip install streamlit streamlit-folium fpdf2
```

### Error: `Descriptors cannot be created directly` (protobuf)
```bash
pip install --upgrade tensorflow
# TF 2.20+ resuelve el conflicto con protobuf moderno
```

### Error: `CUDA_ERROR_NO_DEVICE`
TensorFlow no detecta la GPU. Verifica:
```bash
nvidia-smi # ¿Está la GPU activa?
nvcc --version # ¿CUDA instalado?
```
Si no tienes GPU, el modelo entrenará en CPU (más lento pero funcional).

### Error: `FileNotFoundError: X_train.npy`
No has preparado el dataset. Ejecuta primero:
```bash
python scripts/prepare_dataset.py
```

### Error: `No .tif files found`
No hay imágenes descargadas. Opciones:
1. Descarga con `python scripts/download_dataset.py`
2. O configura el disco externo: `$env:GEOTERMIA_DATA_ROOT = "D:\geotermia_datos"`

### Las imágenes están en el disco externo pero no las encuentra
Verifica:
```bash
python config.py
```
Si dice "local (data/)", la variable de entorno no está configurada.
Asegúrate de que la letra de unidad sea correcta (D:\, E:\, F:\...).

### Error: Earth Engine `Not signed up`
```bash
python -c "import ee; ee.Authenticate()"
```
Luego repite el paso 5.

### Entrenamiento muy lento en CPU
- Reducir `BATCH_SIZE` a 16 en `config.py`
- Reducir `EPOCHS` a 50
- Considerar entrenar en Google Colab con GPU gratuita

---

## FLUJO RÁPIDO (RESUMEN)

```bash
# 1. Setup
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install streamlit streamlit-folium fpdf2

# 2. (Opcional) Configurar disco externo
$env:GEOTERMIA_DATA_ROOT = "D:\geotermia_datos"

# 3. Verificar config
python config.py

# 4. Pipeline de datos (saltar si ya tienes los .npy)
python scripts/download_dataset.py # ~30 min, requiere internet
python scripts/augment_full_dataset.py # ~15 min
python scripts/prepare_dataset.py # ~10 min

# 5. Entrenar
python scripts/train_model.py # 15 min (GPU) / 3 hrs (CPU)

# 6. Interfaz
streamlit run app.py --server.headless true
# → http://localhost:8501
```

---

## ESTRUCTURA FINAL DEL PROYECTO

```
geotermia-colombia-cnn/
├── config.py ← NUEVO: configuración centralizada
├── app.py ← Interfaz web Streamlit
├── requirements.txt ← Dependencias Python
├── setup.py ← Setup del proyecto
├── README.md ← Documentación principal
├── LICENSE
│
├── scripts/
│ ├── download_dataset.py ← Paso 7: descarga de imágenes
│ ├── augment_full_dataset.py← Paso 8: augmentación
│ ├── prepare_dataset.py ← Paso 9: preparación .npy
│ ├── train_model.py ← Paso 10: entrenamiento
│ ├── evaluate_model.py ← Paso 11: evaluación
│ ├── predict.py ← Predicción individual
│ └── visualize_results.py ← Visualización de resultados
│
├── models/
│ ├── cnn_geotermia.py ← Arquitectura del modelo
│ └── saved_models/ ← Modelos entrenados (.keras)
│
├── data/ ← (o disco externo si GEOTERMIA_DATA_ROOT)
│ ├── raw/
│ │ ├── positive/ ← .tif zonas geotérmicas
│ │ ├── negative/ ← .tif zonas control
│ │ └── labels.csv
│ ├── augmented/ ← .tif aumentados
│ └── processed/ ← .npy listos para entrenar
│
├── docs/ ← Documentación del proyecto
├── logs/ ← TensorBoard + historial
├── results/ ← Métricas y gráficas
└── notebooks/ ← Jupyter notebooks exploratorios
```
