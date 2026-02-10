# 🚀 GUÍA PARA ENTRENAR EN OTRA MÁQUINA

**Fecha:** 9 de febrero de 2026  
**Branch:** `main`  
**Propósito:** Entrenar modelo CNN en máquina con mejor hardware

---

## 📋 RESUMEN

Esta guía te permitirá:
1. ✅ Clonar el proyecto en otra máquina
2. ✅ Descargar los datos desde Google Earth Engine
3. ✅ Generar el dataset augmentado (5,518 imágenes)
4. ✅ Entrenar el modelo CNN completo (100 épocas)
5. ✅ Subir el modelo entrenado de vuelta al repositorio

---

## 🔧 PRERREQUISITOS EN LA NUEVA MÁQUINA

### Hardware Recomendado
```
CPU: 8+ cores (o GPU NVIDIA con CUDA)
RAM: 16 GB mínimo
Disco: 10 GB libres
SO: Windows 10/11, Linux, o macOS
```

### Software Necesario
```
✅ Python 3.10 o 3.11
✅ Git
✅ Cuenta de Google Cloud (para Earth Engine)
```

---

## 📥 PASO 1: CLONAR EL REPOSITORIO

### En la nueva máquina:

```bash
# Clonar el repositorio
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn

# Ver el estado
git status
```

---

## 🐍 PASO 2: CONFIGURAR ENTORNO PYTHON

### Crear entorno virtual:

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias:

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar todas las dependencias
pip install -r requirements.txt
```

**Tiempo estimado:** 5-10 minutos

---

## 🌍 PASO 3: CONFIGURAR GOOGLE EARTH ENGINE

### Autenticar Earth Engine:

```bash
python -c "import ee; ee.Authenticate()"
```

Esto abrirá el navegador para autenticación OAuth. Usa la misma cuenta de Google Cloud:
- **Proyecto:** `alpine-air-469115-f0` (My First Project)

**Nota:** Si ya autenticaste antes en la otra máquina, las credenciales están en `~/.config/earthengine/`

---

## 📊 PASO 4: DESCARGAR Y PREPARAR DATOS

### Opción A: Descargar Imágenes Originales (Recomendado)

```bash
# Descargar 85 imágenes ASTER desde Google Earth Engine (~5 min)
python scripts/download_dataset.py
```

**Output:**
- `data/raw/positive/` - 45 imágenes (volcanes y termales)
- `data/raw/negative/` - 40 imágenes (zonas control)
- `data/raw/labels.csv` - Etiquetas
- Total: ~2.5 MB

### Generar Dataset Augmentado:

```bash
# Augmentar a 5,518 imágenes (~30 segundos)
python scripts/augment_full_dataset.py
```

**Output:**
- `data/augmented/positive/` - 4,278 imágenes
- `data/augmented/negative/` - 1,240 imágenes
- Total: ~1.24 GB

### Preparar para Entrenamiento:

```bash
# Procesar y normalizar imágenes (~2 minutos)
python scripts/prepare_dataset.py
```

**Output:**
- `data/processed/X_train.npy` - 3,862 imágenes training
- `data/processed/X_val.npy` - 828 imágenes validation
- `data/processed/X_test.npy` - 828 imágenes test
- `data/processed/y_*.npy` - Etiquetas correspondientes
- Total: ~2.5 GB

### Opción B: Copiar Datos desde Disco Externo (Más Rápido)

Si tienes las imágenes en un disco externo (USB, SSD), puedes configurar
el proyecto para leer directamente desde ahí sin copiar nada:

```powershell
# Windows PowerShell — indicar al proyecto dónde están los datos
$env:GEOTERMIA_DATA_ROOT = "D:\geotermia_datos"

# Verificar que detecta el disco
python config.py
```

Todos los scripts (`download_dataset.py`, `augment_full_dataset.py`, `prepare_dataset.py`,
`train_model.py`) respetarán esta variable y leerán/escribirán en el disco externo.

Alternativamente, si prefieres copiar los datos procesados al proyecto:

```bash
# Copiar carpeta completa data/processed/
# Desde USB o ubicación compartida a:
# geotermia-colombia-cnn/data/processed/
```

Esto te ahorra ~35 minutos de procesamiento.

---

## 🚀 PASO 5: ENTRENAR EL MODELO

### Iniciar Entrenamiento:

```bash
python scripts/train_model.py
```

### Configuración del Entrenamiento:

```yaml
Épocas máximas: 100
Batch size: 32
Learning rate: 0.001
Optimizer: Adam
Callbacks:
  - EarlyStopping (patience=15)
  - ModelCheckpoint (guarda mejor modelo)
  - ReduceLROnPlateau (ajusta LR)
  - TensorBoard (logs)
  - CSVLogger (CSV con métricas)
```

### Tiempo Estimado por Hardware:

```
CPU (8 cores):        ~3-4 horas
CPU (16+ cores):      ~2-3 horas
GPU (NVIDIA RTX):     ~20-40 minutos
GPU (NVIDIA Tesla):   ~10-20 minutos
```

**Nota:** El tiempo real depende del EarlyStopping. Puede detenerse antes de 100 épocas.

### Monitorear el Progreso:

**Opción 1: Salida del Terminal**
```
Epoch 1/100
121/121 [======] - 85s - loss: 0.95 - accuracy: 0.65 - val_loss: 0.89 - val_accuracy: 0.70
...
```

**Opción 2: TensorBoard (Recomendado)**
```bash
# En otra terminal
tensorboard --logdir=logs
# Abrir: http://localhost:6006
```

**Opción 3: CSV**
```bash
# Ver últimas épocas
# Windows:
Get-Content logs\geotermia_cnn_custom_*.csv -Tail 5

# Linux/macOS:
tail -n 5 logs/geotermia_cnn_custom_*.csv
```

---

## 📁 ARCHIVOS GENERADOS

Al finalizar el entrenamiento:

```
models/saved_models/
  └── geotermia_cnn_custom_best.keras  (~19 MB)

logs/
  ├── geotermia_cnn_custom_*.csv       (métricas por época)
  └── tensorboard/                     (logs completos)

models/
  ├── training_history.json            (historial completo)
  └── training_history.csv             (backup)
```

---

## ⬆️ PASO 6: SUBIR MODELO ENTRENADO

### Verificar Tamaño del Modelo:

```bash
# Windows:
Get-ChildItem models\saved_models\*.keras | Select-Object Name, Length

# Linux/macOS:
ls -lh models/saved_models/*.keras
```

### Si el modelo es < 100 MB (recomendado):

```bash
# Agregar modelo al git
git add models/saved_models/geotermia_cnn_custom_best.keras
git add models/training_history.json
git add logs/*.csv

# Commit
git commit -m "feat: Modelo CNN entrenado completamente - Accuracy XX.XX%"

# Push a develop
git push origin develop
```

### Si el modelo es > 100 MB:

**Opción 1: Git LFS (Large File Storage)**
```bash
# Instalar Git LFS
git lfs install

# Trackear archivos grandes
git lfs track "*.keras"
git add .gitattributes

# Commit y push normal
git add models/saved_models/*.keras
git commit -m "feat: Modelo CNN entrenado con Git LFS"
git push origin develop
```

**Opción 2: Google Drive / Dropbox**
```bash
# Subir modelo a Drive
# Agregar link al README o en un archivo MODELO_LINK.txt

git add MODELO_LINK.txt
git commit -m "feat: Modelo entrenado - Link en Drive"
git push origin develop
```

**Opción 3: Solo métricas y logs**
```bash
# Agregar solo resultados, no el modelo
git add models/training_history.json
git add logs/*.csv
git add results/

git commit -m "feat: Resultados de entrenamiento completo"
git push origin develop

# Modelo se transfiere por otro medio
```

---

## 📊 PASO 7: EVALUAR MODELO

```bash
# Evaluar en test set (828 imágenes)
python scripts/evaluate_model.py
```

**Output:**
- `results/metrics/evaluation_metrics.json`
- `results/metrics/confusion_matrix.png`
- `results/metrics/roc_curve.png`

### Generar Visualizaciones:

```bash
python scripts/visualize_results.py
```

**Output:**
- `results/figures/training_history.png`
- `results/figures/sample_predictions.png`
- Todas las figuras en 300 DPI para tesis

### Subir Resultados:

```bash
git add results/
git commit -m "feat: Evaluación completa del modelo - Métricas finales"
git push origin develop
```

---

## 🔄 PASO 8: MERGE A MAIN (En máquina original)

Cuando el entrenamiento esté completo:

```bash
# En máquina original
git checkout main
git pull origin main

# Ver cambios en develop
git fetch origin develop
git log origin/develop

# Hacer merge
git merge develop

# O hacer Pull Request en GitHub para revisión

# Push a main
git push origin main

# Opcional: Eliminar rama develop
git branch -d develop
git push origin --delete develop
```

---

## 🚨 SOLUCIÓN DE PROBLEMAS

### Error: "No module named 'tensorflow'"

```bash
pip install tensorflow==2.20.0
```

### Error: "No such file or directory: data/processed/"

```bash
# Ejecutar preparación de datos primero
python scripts/prepare_dataset.py
```

### Error: "CUDA out of memory" (GPU)

Reducir batch size en `scripts/train_model.py`:
```python
batch_size = 16  # en lugar de 32
```

### Error: "ModuleNotFoundError: No module named 'rasterio'"

```bash
pip install rasterio
```

### Entrenamiento muy lento

**CPU:** Normal, espera 3-4 horas  
**GPU no detectada:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📝 CHECKLIST DE PASOS

```
□ Clonar repositorio y checkout develop
□ Crear y activar entorno virtual
□ Instalar requirements.txt
□ Autenticar Google Earth Engine
□ Descargar imágenes originales (o copiar procesadas)
□ Augmentar dataset
□ Preparar dataset
□ Entrenar modelo (monitorear con TensorBoard)
□ Esperar a EarlyStopping o 100 épocas
□ Evaluar en test set
□ Generar visualizaciones
□ Commit y push modelo/resultados a develop
□ Merge develop → main
□ Eliminar rama develop (opcional)
```

---

## 📊 MÉTRICAS ESPERADAS

Basado en entrenamiento parcial (30 épocas):

| Métrica | Época 30 | Proyección Final |
|---------|----------|------------------|
| **Accuracy** | 65.26% | 70-78% |
| **AUC** | 0.6252 | 0.80-0.90 |
| **Precision** | 84.61% | 85-90% |
| **Recall** | 68.27% | 75-85% |
| **F1-Score** | 75.54% | 80-87% |

---

## 🎯 OBJETIVO FINAL

Al completar todos los pasos:

✅ Modelo CNN entrenado completamente (100 épocas o early stop)  
✅ Métricas finales calculadas en test set  
✅ Visualizaciones de alta calidad (300 DPI)  
✅ Modelo disponible en repositorio o Drive  
✅ Documentación completa actualizada  
✅ Rama develop mergeada a main  

---

## 📞 CONTACTO

**Si tienes problemas:**
1. Revisa la sección "Solución de Problemas"
2. Consulta `RESUMEN_PROYECTO.md` (sección Guía de Monitoreo)
3. Revisa `ANALISIS_ENTRENAMIENTO.md`

**Desarrollador:**
- Cristian Camilo Vega Sánchez
- GitHub: @crisveg24

---

**Última actualización:** 3 de noviembre de 2025  
**Estado:** Guía completa para entrenamiento en máquina externa  
**Tiempo total estimado:** 4-5 horas (incluyendo setup)
