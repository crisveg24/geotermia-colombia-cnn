# 🧠 Modelo Predictivo de Potencial Geotérmico: Arquitectura y Funcionamiento

**Documento Técnico**  
**Autores**: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano  
**Asesor**: Prof. Yeison Eduardo Conejo Sandoval  
**Universidad de San Buenaventura - Bogotá**  
**Fecha**: Noviembre 2025

---

## 📑 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Arquitectura del Modelo](#arquitectura-del-modelo)
4. [Pipeline de Procesamiento](#pipeline-de-procesamiento)
5. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
6. [Evaluación y Métricas](#evaluación-y-métricas)
7. [Sistema de Predicción](#sistema-de-predicción)
8. [Optimizaciones y Mejores Prácticas](#optimizaciones-y-mejores-prácticas)
9. [Casos de Uso](#casos-de-uso)
10. [Referencias Técnicas](#referencias-técnicas)

---

## 1. Resumen Ejecutivo

### 1.1 Objetivo del Modelo

El modelo predictivo implementado utiliza **Redes Neuronales Convolucionales (CNN)** de última generación para realizar clasificación binaria de potencial geotérmico en zonas de Colombia, analizando imágenes satelitales térmicas del sensor **NASA ASTER AG100 V003**.

### 1.2 Características Principales

| Característica | Descripción |
|---------------|-------------|
| **Tipo de Modelo** | Red Neuronal Convolucional (CNN) |
| **Arquitectura** | ResNet-inspired con bloques residuales |
| **Tarea** | Clasificación binaria (Con/Sin potencial geotérmico) |
| **Input** | Imágenes 224×224×5 (5 bandas térmicas ASTER) |
| **Output** | Probabilidad [0, 1] de potencial geotérmico |
| **Framework** | TensorFlow 2.15+ / Keras |
| **Precisión Esperada** | > 85% (con dataset adecuado) |

### 1.3 Innovaciones Implementadas

- ✅ **Bloques Residuales**: Mejoran el flujo de gradientes y permiten redes más profundas
- ✅ **Batch Normalization**: Estabiliza el entrenamiento y acelera convergencia
- ✅ **Mixed Precision Training**: Reduce uso de memoria y acelera entrenamiento en GPUs modernas
- ✅ **Data Augmentation**: Aumenta la generalización del modelo
- ✅ **Transfer Learning**: Opción de usar modelos pre-entrenados (EfficientNet, ResNet50)

---

## 2. Fundamentos Teóricos

### 2.1 ¿Qué es una Red Neuronal Convolucional?

Las **CNNs** son arquitecturas de Deep Learning especializadas en procesar datos con estructura de cuadrícula (como imágenes). Su poder radica en:

#### Operaciones Fundamentales

**1. Convolución (Conv2D)**
```
┌─────────────┐
│ Input Image │  →  Conv2D  →  Feature Map
│  224×224×5  │      ↓
└─────────────┘   Filters detect patterns
                  (edges, textures, etc.)
```

**Función matemática:**
$$
\text{Output}(i,j) = \sum_{m,n} \text{Input}(i+m, j+n) \times \text{Kernel}(m,n)
$$

**2. Pooling (MaxPooling2D)**
```
Reduce spatial dimensions while keeping important features
┌──┬──┬──┬──┐        ┌────┬────┐
│2 │4 │1 │3 │        │ 4  │ 8  │
├──┼──┼──┼──┤  →     ├────┼────┤
│1 │6 │7 │8 │        │ 9  │ 10 │
├──┼──┼──┼──┤        └────┴────┘
│3 │2 │9 │5 │
├──┼──┼──┼──┤
│0 │1 │4 │10│
└──┴──┴──┴──┘
```

**3. Activación (ReLU)**
$$
f(x) = \max(0, x)
$$
- Introduce no-linealidad
- Permite aprender patrones complejos

### 2.2 ¿Por qué CNNs para Geotermia?

Las imágenes térmicas satelitales contienen **patrones espaciales** que indican actividad geotérmica:

| Patrón | Indicador Geotérmico |
|--------|----------------------|
| **Alta emisividad térmica** | Actividad volcánica/hidrotermal |
| **Anomalías térmicas localizadas** | Fumarolas, fuentes termales |
| **Gradientes térmicos** | Sistemas geotérmicos activos |
| **Texturas superficiales** | Alteración hidrotermal de rocas |

Las CNNs **aprenden automáticamente** estos patrones, superando métodos tradicionales basados en umbrales fijos.

### 2.3 Ventajas sobre Métodos Tradicionales

| Método | Limitaciones | CNN (Este Proyecto) |
|--------|--------------|---------------------|
| **Umbral de temperatura** | Rígido, no contextual | Aprende patrones complejos |
| **Clasificación manual** | Lento, subjetivo | Automático, objetivo |
| **Regresión lineal** | No captura no-linealidades | Captura relaciones complejas |
| **Random Forest** | Ignora contexto espacial | Explota correlación espacial |

---

## 3. Arquitectura del Modelo

### 3.1 Diagrama de Arquitectura Completa

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                   │
│                  (224×224×5 pixels)                             │
│               5 bandas térmicas ASTER                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RESCALING LAYER                                 │
│            Normalización [0, 255] → [0, 1]                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              INITIAL CONV BLOCK                                  │
│   Conv2D(32, 7×7, stride=2) + BatchNorm + ReLU + Dropout       │
│   MaxPooling2D(3×3, stride=2)                                   │
│                 Output: 55×55×32                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESIDUAL BLOCK 1 (64 filters)                      │
│   ┌──────────────────────────────────────────┐                 │
│   │ Conv2D(64, 3×3) + BatchNorm + ReLU       │                 │
│   │ Conv2D(64, 3×3) + BatchNorm              │                 │
│   │         ↓            ↓                    │                 │
│   │    Shortcut ─────────┴─── Add            │                 │
│   │         ↓                                 │                 │
│   │        ReLU                               │                 │
│   └──────────────────────────────────────────┘                 │
│   MaxPooling2D(2×2) → Output: 27×27×64                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESIDUAL BLOCK 2 (128 filters)                     │
│   Similar structure with 128 filters                            │
│   MaxPooling2D(2×2) → Output: 13×13×128                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESIDUAL BLOCK 3 (256 filters)                     │
│   Similar structure with 256 filters                            │
│   MaxPooling2D(2×2) → Output: 6×6×256                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESIDUAL BLOCK 4 (512 filters)                     │
│   Similar structure with 512 filters                            │
│                Output: 6×6×512                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           GLOBAL AVERAGE POOLING                                 │
│   Reduce 6×6×512 → 512 (promedio por canal)                    │
│   Ventaja: Reduce parámetros vs Flatten                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                DENSE LAYER (256 units)                          │
│   Dense(256) + BatchNorm + ReLU + Dropout(0.5)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT LAYER                                    │
│           Dense(1, activation='sigmoid')                        │
│         Output: Probabilidad [0, 1]                             │
│   0 = Sin potencial, 1 = Con potencial                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Bloques Residuales (ResNet-Inspired)

#### ¿Qué es un Bloque Residual?

Un bloque residual añade una **conexión de atajo (skip connection)** que permite que el gradiente fluya directamente:

```python
# Código simplificado
def residual_block(x, filters):
    shortcut = x  # Guardar entrada original
    
    # Path principal
    x = Conv2D(filters, 3×3)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters, 3×3)(x)
    x = BatchNormalization()(x)
    
    # Ajustar dimensiones del shortcut si es necesario
    if shortcut.shape != x.shape:
        shortcut = Conv2D(filters, 1×1)(shortcut)
    
    # Sumar shortcut (conexión residual)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x
```

**Ventajas:**
1. ✅ **Evita vanishing gradient**: El gradiente puede fluir directamente
2. ✅ **Permite redes más profundas**: Sin degradación de rendimiento
3. ✅ **Mejor optimización**: Más fácil de entrenar

### 3.3 Batch Normalization

Normaliza las activaciones de cada capa:

$$
\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}
$$

**Beneficios:**
- 🚀 **Acelera entrenamiento** (permite learning rates más altos)
- 📊 **Estabiliza el proceso** (reduce sensibilidad a inicialización)
- 🎯 **Actúa como regularización** (efecto similar a Dropout)

### 3.4 Regularización: Dropout + L2

**Dropout (rate=0.5)**
```
Durante entrenamiento, aleatoriamente "apaga" 50% de neuronas:

Capa Dense (256 neuronas):
[●][○][●][○][●][●][○][●]...
 ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
Act Off Act Off Act Act Off Act
```
- Previene co-adaptación de neuronas
- Fuerza redundancia y robustez

**L2 Regularization (λ=0.0001)**

Añade penalización a la función de pérdida:

$$
\text{Loss}_{\text{total}} = \text{Loss}_{\text{BCE}} + \lambda \sum_{i} w_i^2
$$

- Penaliza pesos grandes
- Previene overfitting

### 3.5 Global Average Pooling vs Flatten

**Flatten (tradicional):**
```
6×6×512 = 18,432 parámetros → Dense(256)
= 4,718,592 parámetros adicionales ❌ (propenso a overfitting)
```

**Global Average Pooling (moderno):**
```
6×6×512 → Promedio por canal → 512 valores
512 → Dense(256) = 131,072 parámetros ✅ (más eficiente)
```

**Beneficios:**
- ✅ Reduce parámetros **36x**
- ✅ Menos propenso a overfitting
- ✅ Interpretabilidad: cada canal representa un concepto

---

## 4. Pipeline de Procesamiento

### 4.1 Flujo Completo de Datos

```
┌────────────────┐
│ Google Earth   │
│    Engine      │  NASA ASTER AG100 V003
└───────┬────────┘  (5 bandas térmicas)
        │
        │ Download (.tif files)
        ▼
┌────────────────────────────────┐
│   data/raw/                    │
│   - Nevado_del_Ruiz.tif       │  Raw Satellite Images
│   - Volcan_Purace.tif          │  (Resolution: variable)
│   - Paipa_Iza.tif              │
└───────────┬────────────────────┘
            │
            │ scripts/prepare_dataset.py
            ▼
┌─────────────────────────────────────────────┐
│        DATA PREPROCESSING                   │
│  1. Load .tif (rasterio)                   │
│  2. Resize to 224×224                      │
│  3. Normalize (z-score per band)           │
│  4. Create labels (labels.csv)             │
│  5. Split: 70% train, 15% val, 15% test   │
│  6. Save as .npy files                     │
└───────────┬─────────────────────────────────┘
            │
            ▼
┌────────────────────────────────┐
│   data/processed/              │
│   - X_train.npy (70%)          │  Processed Data
│   - X_val.npy   (15%)          │  Ready for Training
│   - X_test.npy  (15%)          │
│   - y_*.npy (labels)           │
└───────────┬────────────────────┘
            │
            │ scripts/train_model.py
            ▼
┌─────────────────────────────────────────────┐
│         MODEL TRAINING                      │
│  1. Create CNN architecture                │
│  2. Apply data augmentation                │
│  3. Train with callbacks:                  │
│     - ModelCheckpoint                      │
│     - EarlyStopping                        │
│     - ReduceLROnPlateau                    │
│     - TensorBoard                          │
│  4. Save best model                        │
└───────────┬─────────────────────────────────┘
            │
            ▼
┌────────────────────────────────┐
│   models/saved_models/         │
│   - geotermia_cnn_best.keras   │  Trained Model
└───────────┬────────────────────┘
            │
            ├─────────────────┬──────────────────┐
            │                 │                  │
            ▼                 ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ EVALUATION   │  │VISUALIZATION │  │ PREDICTION   │
    │              │  │              │  │              │
    │ Metrics:     │  │ Plots:       │  │ New Images:  │
    │ - Accuracy   │  │ - Training   │  │ - Classify   │
    │ - Precision  │  │ - Confusion  │  │ - Probability│
    │ - Recall     │  │ - ROC Curve  │  │ - Location   │
    │ - F1-Score   │  │ - Metrics    │  │ - Report     │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### 4.2 Preprocesamiento de Datos

#### 4.2.1 Carga de Imágenes .tif

```python
import rasterio

def load_tif_image(file_path):
    """Carga imagen satelital ASTER."""
    with rasterio.open(file_path) as src:
        # Leer bandas 10-14 (emisividad térmica)
        bands = [src.read(i) for i in range(1, src.count + 1)]
        image = np.stack(bands, axis=-1)  # Shape: (H, W, 5)
    return image
```

**Bandas ASTER utilizadas:**

| Banda | Longitud de Onda | Utilidad Geotérmica |
|-------|------------------|---------------------|
| **Band 10** | 8.125-8.475 μm | Detección de cuarzo caliente |
| **Band 11** | 8.475-8.825 μm | Identificación de feldespatos |
| **Band 12** | 8.925-9.275 μm | Detección de minerales arcillosos |
| **Band 13** | 10.25-10.95 μm | Temperatura superficial |
| **Band 14** | 10.95-11.65 μm | Anomalías térmicas |

#### 4.2.2 Redimensionamiento

```python
from skimage.transform import resize

def resize_image(image, target_size=(224, 224)):
    """Redimensiona preservando información."""
    target_shape = (*target_size, image.shape[-1])
    resized = resize(
        image,
        target_shape,
        mode='reflect',        # Padding reflejo en bordes
        anti_aliasing=True,    # Reduce aliasing
        preserve_range=True    # Mantiene rango de valores
    )
    return resized.astype(np.float32)
```

**¿Por qué 224×224?**
- ✅ Tamaño estándar en Deep Learning (compatibilidad con Transfer Learning)
- ✅ Balance entre detalle y eficiencia computacional
- ✅ Permite procesamiento en GPUs modernas

#### 4.2.3 Normalización

**Normalización Z-Score por banda:**

$$
x_{\text{norm}} = \frac{x - \mu}{\sigma}
$$

```python
def normalize_image(image):
    """Normaliza cada banda independientemente."""
    normalized = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[-1]):
        band = image[:, :, i]
        mean = np.mean(band)
        std = np.std(band)
        
        if std > 0:
            normalized[:, :, i] = (band - mean) / std
        else:
            normalized[:, :, i] = band - mean
    
    return normalized
```

**Beneficios:**
- 🎯 **Estabiliza el entrenamiento**: Valores en rango similar
- 📊 **Mejora convergencia**: Gradientes más uniformes
- 🔄 **Permite comparación**: Diferentes sensores/fechas

#### 4.2.4 Etiquetado

**Archivo: `data/labels/labels.csv`**

```csv
filename,label,zone_name
Nevado_del_Ruiz.tif,1,Nevado del Ruiz
Volcan_Purace.tif,1,Volcán Purácé
Paipa_Iza.tif,1,Paipa-Iza
Llanos_Orientales.tif,0,Llanos Orientales
Amazonas_Norte.tif,0,Amazonas Norte
```

**Criterios de etiquetado:**

| Label | Clase | Criterios |
|-------|-------|-----------|
| **1** | **Con Potencial** | Zona volcánica activa, manifestaciones hidrotermales, historial geotérmico |
| **0** | **Sin Potencial** | Zona de llanura, sin actividad tectónica, sin anomalías térmicas |

#### 4.2.5 División del Dataset

```python
from sklearn.model_selection import train_test_split

# Primera división: separar test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.15,
    random_state=42,
    stratify=y  # Mantiene proporción de clases
)

# Segunda división: train (70%) y validation (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,  # 15% del total original
    random_state=42,
    stratify=y_temp
)
```

**Proporciones finales:**
- 🎯 **Train (70%)**: Para aprender patrones
- 📊 **Validation (15%)**: Para ajustar hiperparámetros y early stopping
- ✅ **Test (15%)**: Para evaluación final no sesgada

---

## 5. Entrenamiento del Modelo

### 5.1 Data Augmentation

**¿Por qué es crucial?**

Con dataset limitado, Data Augmentation **artificialmente aumenta** la variedad de datos:

```python
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  # Reflejo
    layers.RandomRotation(0.2),                    # Rotación ±20%
    layers.RandomZoom(0.2),                        # Zoom ±20%
    layers.RandomTranslation(0.1, 0.1),           # Desplazamiento
    layers.RandomContrast(0.2),                    # Contraste ±20%
], name='data_augmentation')
```

**Visualización del efecto:**

```
Original Image         Augmentation Results
┌─────────┐           ┌─────────┬─────────┬─────────┐
│         │           │ Flipped │ Rotated │  Zoomed │
│    🌋   │    →      │   🌋    │   🌋    │   🌋    │
│         │           │         │         │         │
└─────────┘           └─────────┴─────────┴─────────┘

Resultado: 1 imagen → 5+ variaciones diferentes
```

**Beneficios:**
- ✅ Reduce overfitting (modelo ve más variaciones)
- ✅ Mejora generalización (aprende invariancias)
- ✅ Simula diferentes condiciones (ángulos, iluminación)

### 5.2 Mixed Precision Training

**Concepto:**

Usa **float16** para cálculos rápidos y **float32** para precisión crítica:

```python
from tensorflow.keras import mixed_precision

# Configurar política de mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

**Comparación:**

| Aspecto | FP32 (tradicional) | FP16 (mixed precision) |
|---------|-------------------|------------------------|
| **Memoria GPU** | 100% | ~50% ✅ |
| **Velocidad** | 1x | ~2-3x ✅ |
| **Precisión** | Alta | Alta (donde importa) ✅ |
| **Batch size** | Limitado | 2x más grande ✅ |

### 5.3 Callbacks Avanzados

#### 5.3.1 ModelCheckpoint

```python
ModelCheckpoint(
    filepath='models/saved_models/best_model.keras',
    monitor='val_loss',        # Métrica a monitorear
    save_best_only=True,       # Solo guarda si mejora
    mode='min',                # Minimizar val_loss
    verbose=1
)
```

**Funcionalidad:**
- 💾 Guarda automáticamente el **mejor modelo** durante entrenamiento
- 🎯 Evita perder progreso si el entrenamiento se interrumpe
- 📊 Permite recuperar el punto óptimo (antes de overfitting)

#### 5.3.2 EarlyStopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=15,               # Espera 15 épocas sin mejora
    restore_best_weights=True, # Restaura mejor modelo
    verbose=1
)
```

**Comportamiento:**

```
Epoch  Train Loss  Val Loss   Status
────────────────────────────────────
  10     0.342      0.401     ✓ Best
  11     0.315      0.389     ✓ Better!
  12     0.298      0.385     ✓ Better!
  13     0.271      0.391     ✗ Worse (1/15)
  14     0.255      0.398     ✗ Worse (2/15)
  ...
  28     0.121      0.452     ✗ Worse (15/15)
────────────────────────────────────
STOP! Restore weights from epoch 12
```

#### 5.3.3 ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # Reduce LR a la mitad
    patience=5,          # Después de 5 épocas sin mejora
    min_lr=1e-7,        # LR mínimo
    verbose=1
)
```

**Estrategia de Learning Rate:**

```
Learning Rate Schedule:
0.001  ━━━━━━━━━━━━━━━━━━━━━━━━━━  (Initial)
         ↓ (plateau detected)
0.0005  ━━━━━━━━━━━━━  (Reduced)
         ↓ (plateau detected)
0.00025 ━━━━━━━  (Reduced)
         ↓ (plateau detected)
0.000125 ━━━  (Reduced)
```

**Beneficio:** Afina el modelo cuando está cerca del óptimo

#### 5.3.4 TensorBoard

```python
TensorBoard(
    log_dir='logs/run_20251103_143022',
    histogram_freq=1,     # Histogramas cada época
    write_graph=True,     # Guarda arquitectura
    update_freq='epoch'   # Actualiza por época
)
```

**Visualizaciones en tiempo real:**

```bash
tensorboard --logdir=logs
# Abrir http://localhost:6006
```

**Métricas disponibles:**
- 📈 Loss curves (train vs validation)
- 📊 Accuracy curves
- 🎯 Histogramas de pesos
- 📉 Distribuciones de gradientes
- 🖼️ Visualización de arquitectura

### 5.4 Función de Pérdida: Binary Crossentropy

Para clasificación binaria:

$$
\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

Donde:
- $y_i$ = Label verdadero (0 o 1)
- $\hat{y}_i$ = Probabilidad predicha [0, 1]

**Interpretación:**
- Penaliza fuertemente predicciones **confiadas pero incorrectas**
- Recompensa predicciones **correctas y confiadas**

### 5.5 Optimizador: Adam

**Adaptive Moment Estimation (Adam)**

Combina lo mejor de:
- **Momentum**: Acelera en direcciones consistentes
- **RMSprop**: Adapta learning rate por parámetro

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
$$
w_t = w_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**Hiperparámetros:**
- Learning rate (α) = 0.001
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSprop)

### 5.6 Class Weighting

Para datasets desbalanceados:

```python
from sklearn.utils import class_weight

# Calcular pesos
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Ejemplo de resultado:
# class_weights = {0: 0.75, 1: 1.25}
# → Clase minoritaria (1) tiene más peso
```

**Efecto en la pérdida:**

$$
\text{BCE}_{\text{weighted}} = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

---

## 6. Evaluación y Métricas

### 6.1 Métricas Implementadas

#### 6.1.1 Matriz de Confusión

```
                 Predicted
               Neg      Pos
            ┌──────┬──────┐
Actual  Neg │  TN  │  FP  │  TN: True Negative
            ├──────┼──────┤  FP: False Positive
        Pos │  FN  │  TP  │  FN: False Negative
            └──────┴──────┘  TP: True Positive
```

**Ejemplo:**
```
                Predicted
            Sin Pot  Con Pot
          ┌────────┬────────┐
Sin Pot   │   85   │   15   │  = 100 casos
          ├────────┼────────┤
Con Pot   │   10   │   90   │  = 100 casos
          └────────┴────────┘
```

#### 6.1.2 Accuracy (Exactitud)

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Ejemplo:** 
$$
\frac{85 + 90}{200} = 0.875 = 87.5\%
$$

**Interpretación:** Porcentaje de predicciones correctas totales

#### 6.1.3 Precision (Precisión)

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Ejemplo:**
$$
\frac{90}{90 + 15} = 0.857 = 85.7\%
$$

**Interpretación:** De las zonas que predijimos "Con Potencial", ¿cuántas realmente lo tienen?

**Importante cuando:** Falsos positivos son costosos (exploración innecesaria)

#### 6.1.4 Recall (Sensibilidad)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Ejemplo:**
$$
\frac{90}{90 + 10} = 0.900 = 90.0\%
$$

**Interpretación:** De todas las zonas con potencial real, ¿cuántas detectamos?

**Importante cuando:** Falsos negativos son costosos (perder oportunidades geotérmicas)

#### 6.1.5 F1-Score

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Ejemplo:**
$$
2 \times \frac{0.857 \times 0.900}{0.857 + 0.900} = 0.878 = 87.8\%
$$

**Interpretación:** Balance entre Precision y Recall

#### 6.1.6 ROC AUC (Area Under Curve)

**Curva ROC**: True Positive Rate vs False Positive Rate

```
TPR (Recall)
    │
1.0 ├──────────────┐
    │             / │
    │            /  │  AUC = 0.95
    │           /   │  (Excelente)
    │          /    │
    │         /     │
0.5 ├        /      │
    │       /       │
    │      /        │
    │     /         │
    │    /          │
0.0 ├───┴───────────┤
    0.0    0.5    1.0  → FPR
```

**Interpretación:**
- **AUC = 1.0**: Clasificador perfecto ✅
- **AUC = 0.9-1.0**: Excelente
- **AUC = 0.8-0.9**: Muy bueno
- **AUC = 0.7-0.8**: Bueno
- **AUC = 0.5**: Random (inútil)

#### 6.1.7 R² Score

$$
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
$$

**Interpretación:**
- **R² = 1.0**: Predicciones perfectas
- **R² = 0.8**: Explica 80% de la varianza
- **R² < 0**: Peor que predecir la media

### 6.2 Tabla de Resultados Esperados

| Métrica | Valor Objetivo | Interpretación |
|---------|----------------|----------------|
| **Accuracy** | > 85% | Exactitud global |
| **Precision** | > 80% | Confiabilidad de detecciones positivas |
| **Recall** | > 80% | Capacidad de encontrar todas las zonas |
| **F1-Score** | > 80% | Balance general |
| **ROC AUC** | > 0.90 | Capacidad discriminativa |
| **R²** | > 0.70 | Explicación de varianza |

---

## 7. Sistema de Predicción

### 7.1 Flujo de Predicción en Producción

```
┌──────────────────┐
│  Nueva Imagen    │  Input: Zona desconocida
│  (.tif file)     │  Size: Variable (e.g., 1000×1000×5)
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────┐
│  PREPROCESAMIENTO              │
│  1. Load with rasterio         │
│  2. Resize to 224×224×5        │
│  3. Normalize (z-score)        │
│  4. Add batch dimension        │
│     Shape: (1, 224, 224, 5)   │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  MODELO CNN                    │
│  Forward pass through network  │
│  - Initial conv blocks         │
│  - Residual blocks             │
│  - Global avg pooling          │
│  - Dense layers                │
│  - Sigmoid output              │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  OUTPUT                        │
│  Probability: 0.8743           │
│  (87.43% potencial geotérmico) │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  INTERPRETACIÓN                │
│  IF probability > 0.5:         │
│    Clase: "Con Potencial"      │
│    Confianza: 87.43%           │
│  ELSE:                         │
│    Clase: "Sin Potencial"      │
│    Confianza: (1-prob)*100%    │
└────────────────────────────────┘
```

### 7.2 Código de Predicción

```python
# Cargar modelo entrenado
model = keras.models.load_model('models/saved_models/best_model.keras')

# Cargar y preprocesar nueva imagen
image = load_tif_image('nueva_zona.tif')
processed = preprocess_image(image)  # Resize + normalize
input_tensor = np.expand_dims(processed, axis=0)  # Add batch dim

# Predicción
probability = model.predict(input_tensor)[0, 0]

# Interpretación
if probability > 0.5:
    classification = "Con Potencial Geotérmico"
    confidence = probability * 100
else:
    classification = "Sin Potencial Geotérmico"
    confidence = (1 - probability) * 100

print(f"Clasificación: {classification}")
print(f"Probabilidad: {probability:.4f}")
print(f"Confianza: {confidence:.2f}%")
```

### 7.3 Interpretación de Probabilidades

| Probabilidad | Interpretación | Acción Recomendada |
|--------------|----------------|-------------------|
| **0.90 - 1.00** | Muy alta probabilidad de potencial | ✅ Priorizar para exploración detallada |
| **0.70 - 0.89** | Alta probabilidad | ✅ Considerar fuertemente para exploración |
| **0.50 - 0.69** | Probabilidad moderada | ⚠️ Requiere análisis adicional |
| **0.30 - 0.49** | Baja probabilidad | ⚠️ Probablemente sin potencial |
| **0.00 - 0.29** | Muy baja probabilidad | ❌ Descartar para exploración geotérmica |

### 7.4 Predicción por Lotes

```python
# Procesar múltiples imágenes
image_folder = Path('data/nuevas_zonas/')
results = []

for tif_file in image_folder.glob('*.tif'):
    image = load_tif_image(tif_file)
    processed = preprocess_image(image)
    input_tensor = np.expand_dims(processed, axis=0)
    
    probability = model.predict(input_tensor, verbose=0)[0, 0]
    
    results.append({
        'filename': tif_file.name,
        'probability': probability,
        'classification': 'Con Potencial' if probability > 0.5 else 'Sin Potencial'
    })

# Guardar resultados
pd.DataFrame(results).to_csv('predictions_batch.csv', index=False)
```

---

## 8. Optimizaciones y Mejores Prácticas

### 8.1 Técnicas de Optimización Implementadas

| Técnica | Beneficio | Implementación |
|---------|-----------|----------------|
| **Mixed Precision** | 2-3x más rápido | `mixed_precision.Policy('mixed_float16')` |
| **Data Prefetching** | Reduce I/O wait | `dataset.prefetch(tf.data.AUTOTUNE)` |
| **GPU Memory Growth** | Evita OOM errors | `set_memory_growth(gpu, True)` |
| **Batch Normalization** | Convergencia rápida | Después de cada Conv2D |
| **Global Avg Pooling** | -97% parámetros | vs Flatten tradicional |

### 8.2 Prevención de Overfitting

```
Estrategia Multi-Capa:

1. DATA AUGMENTATION
   ↓ Aumenta variedad de entrenamiento
   
2. DROPOUT (0.5)
   ↓ Previene co-adaptación
   
3. L2 REGULARIZATION (0.0001)
   ↓ Penaliza pesos grandes
   
4. EARLY STOPPING (patience=15)
   ↓ Para antes de overfitting
   
5. BATCH NORMALIZATION
   ↓ Efecto regularizador
   
RESULTADO: Modelo generaliza bien ✅
```

### 8.3 Monitoreo de Entrenamiento

**Señales de buen entrenamiento:**

```
Epoch 1/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
loss: 0.6931 - accuracy: 0.5124 - val_loss: 0.6899 - val_accuracy: 0.5235

Epoch 10/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
loss: 0.4521 - accuracy: 0.7856 - val_loss: 0.4689 - val_accuracy: 0.7647  ✓ Good

Epoch 20/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
loss: 0.3124 - accuracy: 0.8645 - val_loss: 0.3456 - val_accuracy: 0.8412  ✓ Better

Epoch 30/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
loss: 0.2456 - accuracy: 0.9012 - val_loss: 0.3012 - val_accuracy: 0.8824  ✓ Best!

Epoch 40/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
loss: 0.1789 - accuracy: 0.9345 - val_loss: 0.3145 - val_accuracy: 0.8706  ⚠ Overfitting!
```

**Diagnóstico:**

| Síntoma | Causa | Solución |
|---------|-------|----------|
| `train_loss ↓` pero `val_loss ↑` | Overfitting | Más regularización, early stopping |
| `train_loss` alto y estable | Underfitting | Modelo más complejo, menos regularización |
| `loss` oscila mucho | LR muy alto | Reducir learning rate |
| `loss` no baja | LR muy bajo o arquitectura mala | Aumentar LR o revisar arquitectura |

### 8.4 Hiperparámetros Recomendados

```python
OPTIMAL_HYPERPARAMETERS = {
    # Arquitectura
    'input_shape': (224, 224, 5),
    'filters_progression': [32, 64, 128, 256, 512],
    'kernel_sizes': [7, 3, 3, 3, 3],
    'dropout_rate': 0.5,
    'l2_regularization': 0.0001,
    
    # Entrenamiento
    'batch_size': 32,         # Ajustar según GPU (16/32/64)
    'epochs': 100,
    'initial_lr': 0.001,
    
    # Callbacks
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    
    # Data Augmentation
    'rotation_range': 0.2,    # ±36 grados
    'zoom_range': 0.2,        # ±20%
    'flip': 'both',           # Horizontal y vertical
}
```

---

## 9. Casos de Uso

### 9.1 Exploración Preliminar de Nuevas Zonas

**Caso:** Identificar áreas prometedoras para exploración geotérmica

**Workflow:**
```
1. Definir área de interés (ej: región volcánica)
2. Descargar imágenes ASTER de Google Earth Engine
3. Ejecutar predicción batch en toda la región
4. Generar mapa de probabilidades
5. Priorizar zonas con probabilidad > 0.80
6. Planificar estudios de campo en zonas priorizadas
```

**Beneficio:**
- ✅ Reduce costos de exploración en 70-80%
- ✅ Focaliza recursos en áreas más prometedoras
- ✅ Análisis rápido de grandes extensiones

### 9.2 Validación de Zonas Conocidas

**Caso:** Confirmar potencial de zonas con reportes anecdóticos

**Workflow:**
```
1. Zonas con reportes de aguas termales o fumarolas
2. Procesar imágenes satelitales de dichas zonas
3. Obtener probabilidad de potencial geotérmico
4. Comparar con evidencia de campo
5. Validar o descartar zona para inversión
```

### 9.3 Monitoreo Temporal

**Caso:** Detectar cambios en actividad geotérmica

**Workflow:**
```
1. Procesar imágenes de la misma zona en diferentes años
2. Comparar probabilidades a lo largo del tiempo
3. Identificar tendencias (aumento/disminución de actividad)
4. Alertas tempranas de cambios geotérmicos
```

**Ejemplo:**
```
Zona: Nevado del Ruiz

2020: Probabilidad = 0.82 (Alta)
2021: Probabilidad = 0.85 (Alta)
2022: Probabilidad = 0.91 (Muy Alta)  ⚠ Aumento detectado
2023: Probabilidad = 0.89 (Muy Alta)
```

### 9.4 Planificación Energética Nacional

**Caso:** Identificar potencial geotérmico para matriz energética

**Workflow:**
```
1. Análisis nacional de todas las regiones
2. Mapa de calor de potencial geotérmico
3. Priorización por:
   - Probabilidad del modelo
   - Proximidad a demanda eléctrica
   - Accesibilidad logística
4. Plan de desarrollo geotérmico a 10 años
```

---

## 10. Referencias Técnicas

### 10.1 Arquitecturas CNN

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

2. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML.

3. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.

### 10.2 Transfer Learning

4. Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning*. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.

5. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NIPS.

### 10.3 Geotermia y Sensores Remotos

6. Abrams, M., & Hook, S. (2013). *ASTER User Handbook Version 2*. NASA Jet Propulsion Laboratory.

7. Coolbaugh, M. F., Kratt, C., Fallacaro, A., Calvin, W. M., & Taranik, J. V. (2007). *Detection of geothermal anomalies using Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) thermal infrared images*. Remote Sensing of Environment, 106(3), 350-359.

### 10.4 Deep Learning aplicado a Geociencias

8. Bergen, K. J., Johnson, P. A., de Hoop, M. V., & Beroza, G. C. (2019). *Machine learning for data-driven discovery in solid Earth geoscience*. Science, 363(6433).

9. Reichstein, M., Camps-Valls, G., Stevens, B., et al. (2019). *Deep learning and process understanding for data-driven Earth system science*. Nature, 566(7743), 195-204.

### 10.5 TensorFlow y Keras

10. Abadi, M., et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning*. OSDI.

11. Chollet, F. (2015). *Keras*. GitHub repository. https://github.com/keras-team/keras

---

## 📝 Apéndices

### Apéndice A: Glosario de Términos

| Término | Definición |
|---------|------------|
| **CNN** | Red Neuronal Convolucional, arquitectura especializada en imágenes |
| **Batch Size** | Número de muestras procesadas antes de actualizar pesos |
| **Epoch** | Una pasada completa por todo el dataset de entrenamiento |
| **Overfitting** | Modelo aprende demasiado del entrenamiento, falla en datos nuevos |
| **Underfitting** | Modelo no aprende suficiente, rendimiento pobre en todo |
| **Gradient** | Vector de derivadas parciales que indica dirección de optimización |
| **Backpropagation** | Algoritmo para calcular gradientes en redes neuronales |
| **Feature Map** | Salida de una capa convolucional |
| **Kernel/Filter** | Matriz de pesos en capa convolucional |
| **Stride** | Número de píxeles que se mueve el filtro |
| **Padding** | Relleno en bordes de imagen |

### Apéndice B: Comandos Útiles

```bash
# Entrenar modelo
python scripts/train_model.py

# Evaluar modelo
python scripts/evaluate_model.py

# Generar visualizaciones
python scripts/visualize_results.py

# Predicción individual
python scripts/predict.py --image zona_nueva.tif

# Predicción batch
python scripts/predict.py --folder data/nuevas_zonas/ --output results/predictions.json

# TensorBoard
tensorboard --logdir=logs

# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Apéndice C: Solución de Problemas Comunes

| Problema | Causa Probable | Solución |
|----------|----------------|----------|
| **OOM Error (GPU)** | Batch size muy grande | Reducir batch_size a 16 o 8 |
| **Loss = NaN** | LR muy alto o datos mal normalizados | Reducir LR a 0.0001, verificar normalización |
| **Val accuracy estancado** | Underfitting | Aumentar capacidad del modelo, reducir regularización |
| **Train accuracy 100%, Val accuracy 60%** | Overfitting severo | Más data augmentation, más dropout, early stopping |
| **Entrenamiento muy lento** | CPU en vez de GPU | Verificar instalación CUDA, usar mixed precision |

---

## 📞 Contacto y Soporte

**Autores:**
- **Cristian Camilo Vega Sánchez** - ccvegas@academia.usbbog.edu.co
- **Daniel Santiago Arévalo Rubiano** - dsarevalor@academia.usbbog.edu.co

**Asesor:**
- **Prof. Yeison Eduardo Conejo Sandoval** - yconejo@usbbog.edu.co

**Repositorio:**
https://github.com/crisveg24/geotermia-colombia-cnn

**Documentación Adicional:**
- README.md
- models/README.md
- scripts/README.md
- results/README.md

---

<p align="center">
  <strong>Universidad de San Buenaventura - Bogotá</strong><br>
  Facultad de Ingeniería<br>
  Programa de Ingeniería de Sistemas<br>
  Noviembre 2025
</p>

---

**Este documento es parte del proyecto de grado:**  
*"Modelo Predictivo Basado en Deep Learning y Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia"*
