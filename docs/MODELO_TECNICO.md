# Documentación Técnica Completa — CNN Geotermia Región Andina (v3)

**Proyecto de Grado — Universidad de San Buenaventura, Bogotá**
**Autores:** Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín
**Asesor:** Prof. Yeison Eduardo Conejo Sandoval
**Última actualización:** 7 de marzo de 2026

---

## Tabla de Contenidos

1. [Contexto: Energía Geotérmica](#1-contexto-energía-geotérmica)
2. [Datos Satelitales: Sensor ASTER](#2-datos-satelitales-sensor-aster)
3. [Dataset: Región Andina](#3-dataset-región-andina)
4. [Pipeline de Procesamiento](#4-pipeline-de-procesamiento)
5. [Conceptos de Deep Learning](#5-conceptos-de-deep-learning)
6. [Arquitectura del Modelo](#6-arquitectura-del-modelo)
7. [Entrenamiento](#7-entrenamiento)
8. [Métricas de Evaluación](#8-métricas-de-evaluación)
9. [Resultados](#9-resultados)
10. [Bugs Descubiertos y Corregidos](#10-bugs-descubiertos-y-corregidos)
11. [Evolución del Proyecto (v1 → v2 → v3)](#11-evolución-del-proyecto-v1--v2--v3)
12. [Catálogo de Campos Geotérmicos](#12-catálogo-de-campos-geotérmicos)

---

## 1. Contexto: Energía Geotérmica

### 1.1 ¿Qué es?

La energía geotérmica es el **calor almacenado en el interior de la Tierra**, proveniente del calor primordial de la formación del planeta y del decaimiento radiactivo de isótopos (uranio-238, torio-232, potasio-40). Este calor genera un gradiente geotérmico promedio de 25–30 °C/km de profundidad, que en zonas volcánicas puede superar los 100 °C/km.

### 1.2 El "Triángulo Geotérmico"

Para que un recurso geotérmico sea explotable se necesitan tres componentes:

| Componente | Descripción |
|------------|-------------|
| **Fuente de calor** | Intrusión magmática o gradiente geotérmico elevado |
| **Reservorio** | Roca con porosidad y permeabilidad suficientes |
| **Fluido** | Agua (líquida o vapor) que transporta el calor |

### 1.3 Tipos de sistemas geotérmicos

- **Hidrotermales convencionales:** Agua subterránea se calienta naturalmente. Se manifiestan como aguas termales, fumarolas, géiseres. Temperaturas: 150–350 °C a 1–3 km. Ejemplo: Nevado del Ruiz (Colombia), The Geysers (California).

- **Sistemas Geotérmicos Mejorados (EGS):** Roca caliente sin agua ni permeabilidad. Se inyecta agua a presión para crear fracturas. No requieren aguas termales. Temperaturas: 150–300 °C a 3–6 km. Ejemplo: FORGE (Utah, EE.UU.).

- **Uso directo y bombas de calor (GSHP):** Aprovechan la temperatura estable del subsuelo (~15 °C). Funcionan en cualquier lugar, sin vulcanismo.

### 1.4 ¿Por qué Colombia?

Colombia posee un potencial geotérmico estimado superior a **2.000 MW** (SGC, 2023) debido a su ubicación en el Cinturón de Fuego del Pacífico, donde convergen las placas tectónicas de Nazca, Sudamericana y del Caribe. A pesar de esto, **no existe ninguna planta geotérmica operativa** en el país.

**Zonas geotérmicas conocidas:**

| Zona | Departamento | Tipo | Temperatura |
|------|-------------|------|:-----------:|
| Nevado del Ruiz | Caldas/Tolima | Volcánico-hidrotermal | > 200 °C |
| Chiles–Cerro Negro | Nariño | Volcánico-hidrotermal | > 200 °C |
| Azufral | Nariño | Volcánico-hidrotermal | > 200 °C |
| Paipa–Iza | Boyacá | Hidrotermal no volcánico | 150–200 °C |
| Puracé–Coconucos | Cauca | Volcánico-hidrotermal | 150–200 °C |
| Santa Rosa de Cabal | Risaralda | Hidrotermal | ~150 °C |

### 1.5 Rol de la CNN en la exploración

La exploración geotérmica convencional tiene 4 fases: reconocimiento regional ($50K–200K), exploración de superficie ($200K–1M), exploración profunda ($5M–20M) y desarrollo ($50M–200M). Nuestro modelo CNN actúa en la **Fase 1 (Reconocimiento Regional)** como herramienta de **screening automatizado**:

- Recibe una imagen ASTER de cualquier punto de Colombia
- Analiza patrones espectrales asociados a actividad geotérmica
- Produce una **probabilidad (0–100%)** de potencial geotérmico

**No reemplaza** la exploración de campo ni confirma la existencia de un recurso explotable. Su valor es **priorizar zonas** para inversión en exploración detallada.

### 1.6 Contexto geológico compartido de la Región Andina

Los países andinos (Colombia, Ecuador, Perú y Chile) comparten el contexto geológico del Cinturón de Fuego del Pacífico: subducción de la placa de Nazca bajo la placa Sudamericana, generando actividad volcánica y geotérmica a lo largo de los Andes. Chile tiene la primera planta geotérmica de Sudamérica (Cerro Pabellón, 48 MW). Esta similitud geológica justifica expandir el dataset de entrenamiento a la región andina completa, mientras que la **aplicación** del modelo se circunscribe a Colombia.

---

## 2. Datos Satelitales: Sensor ASTER

### 2.1 ¿Qué es ASTER?

El sensor **ASTER** (Advanced Spaceborne Thermal Emission and Reflection Radiometer), a bordo del satélite Terra de la NASA, dispone de 14 bandas espectrales: VNIR (3), SWIR (6) y TIR (5). Las bandas TIR son particularmente relevantes para detectar anomalías térmicas superficiales y composición mineralógica.

### 2.2 Producto utilizado: ASTER GED AG100 v003

- **Proveedor:** NASA/METI/AIST/Japan Spacesystems
- **ID en Google Earth Engine:** `NASA/ASTER_GED/AG100_003`
- **Resolución espacial:** 100 metros (90 m/px en GEE)
- **Cobertura:** Global
- **Tipo:** Promedio temporal (no captura variaciones estacionales)

### 2.3 Las 7 bandas utilizadas

| # | Banda | Longitud de onda | Utilidad geotérmica |
|:-:|-------|-----------------|---------------------|
| 1 | emissivity_band10 | 8,125–8,475 μm | Detección de cuarzo caliente |
| 2 | emissivity_band11 | 8,475–8,825 μm | Identificación de feldespatos |
| 3 | emissivity_band12 | 8,925–9,275 μm | Detección de minerales arcillosos |
| 4 | emissivity_band13 | 10,25–10,95 μm | Temperatura superficial |
| 5 | emissivity_band14 | 10,95–11,65 μm | Anomalías térmicas |
| 6 | temperature | Temperatura superficial (°C × 100) | Indicador directo de calor |
| 7 | ndvi | Índice de vegetación normalizado | Proxy de cobertura vegetal |

Las 5 bandas de emisividad (TIR) detectan **anomalías térmicas y composición mineral**. La temperatura ofrece un **indicador directo** de calor superficial. El NDVI actúa como **proxy** de cobertura vegetal (que puede enmascarar señales térmicas).

### 2.4 Indicadores geotérmicos detectables por satélite

| Indicador | Detectable | Banda relevante |
|-----------|:----------:|-----------------|
| Anomalías térmicas superficiales | ✅ | TIR (bandas 10–14) |
| Alteración hidrotermal de minerales | ✅ | TIR |
| Composición mineralógica (arcillas, sílice) | ✅ | TIR |
| Gradiente geotérmico elevado | ❌ (indirecto) | — |
| Presencia de vulcanismo reciente | ✅ | TIR, temperature |

---

## 3. Dataset: Región Andina

### 3.1 Composición del dataset v3

| País | Positivas | Negativas | Total |
|------|:---------:|:---------:|:-----:|
| Colombia | 544 | 477 | 1.021 |
| Ecuador | 184 | 66 | 250 |
| Perú | 107 | 28 | 135 |
| Chile | 162 | 451 | 613 |
| **Total** | **997** | **1.022** | **2.019** |

Cada zona base se expande en una **grilla de 9 tiles** (center + N, S, E, W, NE, NW, SE, SW) desplazados ±0,045° (~4 km) para maximizar cobertura espacial.

### 3.2 Criterios de selección de zonas

**Positivas (label=1):**
1. Campo geotérmico confirmado por servicio geológico nacional
2. Volcán activo con fumarolas (actividad en el Holoceno, < 11.700 años)
3. Manifestación hidrotermal con T > 40 °C
4. Buffer de 5 km por zona, subdivisiones 3×3

**Negativas (label=0):**
1. Ausencia de vulcanismo en radio > 50 km
2. Estabilidad geológica: cuencas sedimentarias, llanuras costeras
3. Diversidad geomorfológica: costa, llanura, selva, altiplano
4. Distancia mínima > 20 km entre puntos

### 3.3 Aumento de datos (Data Augmentation)

Se aplican **10 técnicas offline** por imagen (reducidas de 30 en v2, siguiendo Shorten & Khoshgoftaar, 2019):

1. Rotación 90° / 180° / 270°
2. Volteo horizontal / vertical
3. Trasposición
4. Ajuste de brillo (±15 %)
5. Ajuste de contraste (±15 %)
6. Ajuste de gamma
7. Ruido gaussiano

**Resultado:** 2.019 originales → **22.209 imágenes** (11.016 pos / 11.193 neg). Balance casi perfecto.

### 3.4 División anti-fuga geográfica

La división se realiza con **GroupShuffleSplit** agrupando por **zona geográfica base** (eliminando sufijos `_aug*` y `_grid_*`), de modo que TODAS las variantes de una misma zona geográfica quedan en el mismo subconjunto:

| Subconjunto | Imágenes | Zonas base | Proporción |
|-------------|:--------:|:----------:|:----------:|
| Entrenamiento | 15.037 | 284 | 67,7 % |
| Validación | 3.553 | 61 | 16,0 % |
| Prueba | 3.619 | 62 | 16,3 % |
| **Total** | **22.209** | **407** | 100 % |

**Solapamiento de zonas entre subconjuntos: 0 (CERO — verificado por scripts independientes).**

### 3.5 Descarga desde Google Earth Engine

- Descarga paralela con **3 hilos concurrentes**
- Retardo de 0,5 s entre solicitudes (respeto de cuotas API)
- Cada imagen: buffer de 5 km, resolución 90 m/px → ~111 × 111 × 7 píxeles
- Formato: GeoTIFF 7 bandas
- Tiempo total de descarga: ~2 horas

---

## 4. Pipeline de Procesamiento

### 4.1 Flujo completo

```
Google Earth Engine (ASTER GED AG100 v003)
    │
    │  scripts/download_dataset.py (3 hilos, ~2h)
    ▼
data/raw/ → 2.019 imágenes .tif (7 bandas, ~115 MB)
    │
    │  scripts/augment_full_dataset.py (×10 técnicas)
    ▼
data/augmented/ → 22.209 imágenes .tif (~7,9 GB)
    │
    │  scripts/prepare_dataset.py
    │   ├── Filtrado NoData (-9999) → interpolación mediana por banda
    │   ├── Redimensionamiento a 224×224 (bicúbico, anti-aliasing)
    │   ├── Normalización z-score global por banda (Welford online)
    │   ├── GroupShuffleSplit por zona geográfica base
    │   └── Guardado en .npy particionado (~500 imgs/parte)
    ▼
data/processed/ → ~29,8 GB
    │   X_train_part00-30.npy, y_train.npy
    │   X_val_part00-07.npy, y_val.npy
    │   X_test_part00-07.npy, y_test.npy
    │   band_stats_v3.json, split_info.json
    │
    │  scripts/train_model_v7.py (GPU RTX 4070, WSL2)
    ▼
models/saved_models/geotermia_v7_phase2_best.keras (60 MB)
    │
    │  scripts/evaluate_model.py (Bootstrap CI 95%, 2.000 iteraciones)
    ▼
results/metrics/evaluation_metrics.json
    │
    │  app.py (Streamlit + Folium)
    ▼
Interfaz web → predicción en tiempo real por coordenadas
```

### 4.2 Filtrado de NoData

Los datos ASTER usan **-9999** como indicador de ausencia de datos. El filtrado:
1. Detecta valores -9999 en cada banda
2. Si > 50 % de una banda es NoData → imagen descartada
3. Valores NoData restantes → reemplazados por la mediana de valores válidos de la misma banda

### 4.3 Redimensionamiento

Imágenes originales (~111×111 px) → **224×224 px** mediante interpolación bicúbica con anti-aliasing (`skimage.transform.resize`, `mode='reflect'`). El tamaño 224×224 es estándar en deep learning y compatible con EfficientNetB0.

### 4.4 Normalización z-score global

Cada banda se normaliza con la **media y desviación estándar globales** calculadas sobre todo el dataset:

$$x_{\text{norm}} = \frac{x - \mu_{\text{banda}}^{\text{global}}}{\sigma_{\text{banda}}^{\text{global}}}$$

Las estadísticas globales se calculan en un solo pase con el **algoritmo de Welford** (online, O(1) RAM) y se guardan en `band_stats_v3.json` para reutilizar en inferencia.

> **Nota crítica:** La v1 y los primeros intentos de v3 normalizaban **por imagen individual**, lo que destruía toda información absoluta entre imágenes (una zona a 80 °C y otra a 20 °C quedaban idénticas). Este fue el Bug #5 de v3 — ver sección 10.

---

## 5. Conceptos de Deep Learning

### 5.1 Redes Neuronales Convolucionales (CNN)

Las CNN son arquitecturas de aprendizaje profundo especializadas en datos con estructura de cuadrícula (imágenes). Su poder radica en tres operaciones:

**Convolución (Conv2D):** Aplica filtros (kernels) aprendibles que detectan patrones locales:

$$\text{Output}(i,j) = \sum_{m,n} \text{Input}(i+m, j+n) \times \text{Kernel}(m,n) + b$$

Los filtros de las primeras capas aprenden bordes y texturas; los de capas profundas aprenden patrones de alto nivel (anomalías térmicas, composición mineral).

**Pooling (MaxPooling2D):** Reduce dimensiones espaciales conservando las características más relevantes. Aporta invariancia a traslaciones menores.

**Activación (ReLU):** Introduce no linealidad: $f(x) = \max(0, x)$. Permite al modelo aprender relaciones complejas.

### 5.2 Redes residuales (ResNet) — usadas en v2

He et al. (2016) introdujeron las **conexiones residuales (skip connections)**: el gradiente fluye directamente a través de las capas, evitando el problema de degradación en redes profundas.

En un bloque residual:
$$y = F(x, \{W_i\}) + x$$

donde $F$ es la transformación del camino principal y $x$ es la entrada transmitida por el atajo. La red aprende la función residual $F(x) = y - x$.

### 5.3 Transfer Learning — usado en v3

Consiste en **reutilizar pesos** de un modelo preentrenado en un dominio fuente (ImageNet, 1,2M imágenes) y adaptarlos al dominio objetivo. Las primeras capas aprenden características genéricas (bordes, texturas) que son **transferibles entre dominios**; las capas superiores se especializan.

**Estrategia de entrenamiento en 2 fases:**
1. **Backbone congelado:** Solo se entrenan el adapter y el clasificador
2. **Fine-tuning:** Se descongelan las últimas capas del backbone con un learning rate reducido

### 5.4 EfficientNet (Tan & Le, 2019)

Arquitectura que optimiza simultáneamente la **profundidad, ancho y resolución** mediante un coeficiente de escalado compuesto. EfficientNetB0 (la variante base) alcanza rendimiento comparable a redes mucho más grandes con solo 4,0M parámetros.

Su bloque fundamental es el **MBConv** (Mobile Inverted Bottleneck):
- Convoluciones depthwise separable (eficientes en parámetros)
- **Squeeze-and-Excitation (SE):** Calibra adaptativamente la importancia de cada canal
- Skip connections internas

### 5.5 Channel Adapter

Módulo convolucional diseñado para **proyectar las 7 bandas ASTER al espacio de 3 canales** esperado por EfficientNetB0 (preentrenado en RGB):

- Conv2D(16, 3×3) + BatchNorm + ReLU: expande 7 bandas a 16 mapas intermedios
- Conv2D(3, 1×1) + BatchNorm + ReLU: proyecta a 3 canales

Este adapter **aprende la proyección óptima** del espacio espectral ASTER al espacio RGB de ImageNet, en lugar de seleccionar o promediar bandas manualmente.

### 5.6 Técnicas de regularización

| Técnica | Qué hace | Parámetro en v3 |
|---------|----------|:---------------:|
| **Dropout** | Desactiva aleatoriamente neuronas durante entrenamiento | 0,3 (head), 0,5 (dense) |
| **Batch Normalization** | Normaliza activaciones de cada capa, estabiliza convergencia | Todas las capas |
| **Label Smoothing** | Suaviza etiquetas duras → $(0.05, 0.95)$ en vez de $(0, 1)$ | ε = 0,1 |
| **Weight Decay (AdamW)** | L2 desacoplado sobre los pesos | 1 × 10⁻³ |
| **MixUp** | Interpola pares de muestras y etiquetas: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$ | α = 0,2 |
| **CosineDecay** | Learning rate decae siguiendo una curva coseno | Hasta α = 1 × 10⁻⁴ |
| **EarlyStopping** | Detiene entrenamiento si no mejora | patience = 10/15 |

**MixUp en detalle:** Genera ejemplos virtuales de entrenamiento:
$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$
donde $\lambda \sim \text{Beta}(\alpha, \alpha)$. Suaviza la frontera de decisión y mejora la calibración.

### 5.7 Mixed Precision Training (float16)

Usa aritmética de 16 bits para las operaciones forward/backward y 32 bits para la acumulación de gradientes. Beneficios:
- **Duplica el throughput** en GPU modernas (RTX 4070)
- **Reduce consumo de VRAM** (~50 %)
- Sin pérdida de precisión (loss scaling automático)

### 5.8 Global Average Pooling

En lugar de aplanar (Flatten) la salida del backbone (que generaría millones de parámetros), se calcula el **promedio por canal**:

```
7×7×1280 → Promedio por canal → 1280 valores
```

Reduce parámetros drásticamente y es menos propenso a overfitting.

---

## 6. Arquitectura del Modelo

### 6.1 Modelo v3 (final): EfficientNetB0 + Channel Adapter

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: (224×224×7)                                              │
│ 5 bandas emisividad + temperatura + NDVI                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ CHANNEL ADAPTER                                                 │
│ Conv2D(16, 3×3, padding='same', no bias) + BatchNorm + ReLU   │
│ Conv2D(3, 1×1, padding='same', no bias) + BatchNorm + ReLU    │
│ Output: (224×224×3)                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKBONE: EfficientNetB0 (pesos ImageNet)                      │
│ 237 capas, bloques MBConv + Squeeze-and-Excitation             │
│ Fase 1: 100% congelado                                         │
│ Fase 2: últimas 39 capas descongeladas (BatchNorm congelado)   │
│ Output: (7×7×1280)                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ GLOBAL AVERAGE POOLING → 1280                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION HEAD                                             │
│ Dense(256) + BatchNorm + ReLU + Dropout(0.5)                   │
│ Dense(64) + BatchNorm + ReLU + Dropout(0.3)                    │
│ Dense(1, sigmoid) → Probabilidad [0, 1]                        │
└─────────────────────────────────────────────────────────────────┘

Total parámetros: 4.396.112
Nombre del modelo: GeotermiaCNN_V7
```

### 6.2 Modelo v2 (previo): CNN ResNet-inspired

Para referencia, la v2 empleaba una arquitectura personalizada con:
- Conv2D inicial (32 filtros, 7×7, stride=2) + BN + ReLU
- 4 bloques residuales (64, 128, 256, 512 filtros) con skip connections
- GlobalAveragePooling2D → Dense(256) → Dense(1, sigmoid)
- **5.032.385 parámetros**

### 6.3 ¿Por qué cambiar de v2 a v3?

| Aspecto | v2 (ResNet-inspired) | v3 (EfficientNetB0) |
|---------|:--------------------:|:-------------------:|
| Parámetros | 5.032.385 | **4.396.112** (−12,6 %) |
| Conocimiento previo | Ninguno (desde cero) | **ImageNet (1,2M imgs)** |
| Recall | 86,05 % | **93,17 %** (+7,12 pp) |
| Dataset de prueba | 1.017 imgs, 1 país | **3.619 imgs, 4 países** |
| Entrenamiento | CPU, 22 épocas | **GPU, 80 épocas (2 fases)** |

---

## 7. Entrenamiento

### 7.1 Estrategia de 2 fases

**Fase 1 — Backbone congelado (30 épocas):**

| Parámetro | Valor |
|-----------|-------|
| Capas entrenables | 345.863 (Channel Adapter + Head) |
| Backbone | 100 % congelado (pesos ImageNet) |
| Optimizador | AdamW (weight_decay = 1 × 10⁻³) |
| Learning rate | 1 × 10⁻³ con CosineDecay |
| Batch size | 32 |
| MixUp | α = 0,2 |
| Label smoothing | 0,1 |
| EarlyStopping | patience = 10, monitor = val_auc |
| Mejor época | 27 (val_auc = 0,9000) |

**Fase 2 — Fine-tuning (50 épocas):**

| Parámetro | Valor |
|-----------|-------|
| Capas descongeladas | Últimas 39 capas del backbone |
| BatchNormalization | Congelado (preserva estadísticas ImageNet) |
| Optimizador | AdamW (weight_decay = 1 × 10⁻³) |
| Learning rate | 1 × 10⁻⁴ con CosineDecay (**10× menor** que Fase 1) |
| Batch size | 32 |
| MixUp | α = 0,2 |
| Label smoothing | 0,1 |
| EarlyStopping | patience = 15, monitor = val_auc |
| Mejor época | 50 (val_auc = 0,9725) |

**Parámetros comunes:**

| Parámetro | Valor |
|-----------|-------|
| Función de pérdida | BinaryCrossentropy (label_smoothing = 0,1) |
| Métricas monitoreadas | Accuracy, Precision, Recall, AUC, PR-AUC, F1Score |
| ModelCheckpoint | monitor = val_auc, save_best_only, mode = max |
| Pesos de clase | Calculados automáticamente (~1,0 por equilibrio) |
| Semilla aleatoria | 42 |
| Hardware | NVIDIA RTX 4070, 12 GB VRAM, WSL2 Ubuntu 22.04 |
| Precisión mixta | float16 (mixed_float16) |
| Velocidad | ~347 ms/step (45× más rápido que CPU) |
| Total épocas | 80 (30 + 50) |
| Script | `scripts/train_model_v7.py` |

### 7.2 ¿Por qué 2 fases?

1. **Fase 1** entrena solo el adapter y el clasificador (~346K params) con un LR alto. El backbone ya tiene buenos filtros de ImageNet; no queremos destruirlos.

2. **Fase 2** desbloquea las últimas 39 capas del backbone para **ajustar finamente** los filtros al dominio ASTER. Se usa un LR 10× menor para no destruir los pesos preentrenados. Las BatchNorm se mantienen congeladas porque sus estadísticas de ImageNet siguen siendo útiles.

### 7.3 ¿Por qué mantener BatchNorm congelado?

Las capas de Batch Normalization almacenan estadísticas (media y varianza) calculadas durante el preentrenamiento en ImageNet. Si se descongelan, estas estadísticas se recalculan con los datos de nuestro dataset (mucho más pequeño), lo que puede desestabilizar el entrenamiento.

### 7.4 Generador de datos por particiones

Los 22.209 imágenes (~29,8 GB en .npy) no caben en RAM. El generador:
1. Pre-shuffle de datos en disco (seed=42, garantiza mezcla de clases)
2. Cada época: shuffle del **orden de las partes** (~500 imgs c/u)
3. Carga **una parte completa** a la vez (lectura secuencial, ~670 MB)
4. Shuffle de muestras **dentro** de cada parte
5. Yield batches de 32 imágenes

Esta estrategia mantiene excelente aleatoriedad con I/O óptimo.

---

## 8. Métricas de Evaluación

### 8.1 Definiciones

**Exactitud (Accuracy):** Proporción de predicciones correctas:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precisión (Precision):** De las predicciones positivas, cuántas son correctas:
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Sensibilidad (Recall):** De los positivos reales, cuántos detecta:
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score:** Media armónica de precisión y sensibilidad:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**ROC AUC:** Capacidad discriminativa del modelo en todos los umbrales (1,0 = perfecto, 0,5 = azar).

**MCC (Matthews Correlation Coefficient):** Métrica equilibrada que considera las 4 categorías de la matriz de confusión:
$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 8.2 Intervalos de confianza Bootstrap

Se estima la variabilidad de cada métrica generando **2.000 muestras con reemplazo** del conjunto de prueba, computando la métrica en cada muestra y tomando los percentiles 2,5 % y 97,5 % como intervalo al 95 %.

### 8.3 ¿Qué métrica importa más para screening geotérmico?

El **recall** (sensibilidad) es la métrica más crítica en un contexto de screening: es preferible investigar un falso positivo (zona señalada erróneamente) que **omitir** una zona con potencial real. El modelo v3 prioriza recall (93,17 %) sobre precisión (91,27 %).

---

## 9. Resultados

### 9.1 Métricas del modelo v3 (test set: 3.619 imágenes, 4 países)

| Métrica | Valor | Objetivo mínimo | Objetivo ideal | Estado |
|---------|:-----:|:---------------:|:--------------:|:------:|
| Exactitud | **92,28 %** | > 85 % | > 90 % | ✅ Superado |
| Precisión | **91,27 %** | > 80 % | > 85 % | ✅ Superado |
| Sensibilidad | **93,17 %** | > 80 % | > 85 % | ✅ Superado |
| F1-Score | **92,21 %** | > 80 % | > 85 % | ✅ Superado |
| ROC AUC | **0,9737** | > 0,90 | > 0,95 | ✅ Superado |
| PR AUC | **0,9693** | — | — | — |
| MCC | **0,8458** | > 0,50 | > 0,70 | ✅ Superado |

**Todas las métricas superan los objetivos ideales.**

### 9.2 Matriz de confusión v3

|  | Predicho Negativo | Predicho Positivo |
|---|:-:|:-:|
| **Real Negativo** | 1.686 (VN) | 158 (FP) |
| **Real Positivo** | 121 (FN) | 1.651 (VP) |

- **Especificidad:** 91,43 % (1.686/1.844 negativos correctos)
- **Sensibilidad:** 93,17 % (1.651/1.772 positivos correctos)
- **Tasa FP:** 8,57 % — mayor que v2 (2,15 %) por dataset más diverso
- **Tasa FN:** 6,83 % — mejora significativa respecto a v2 (13,95 %)

### 9.3 Contraste de hipótesis

Con accuracy 92,28 % sobre 3.619 imágenes y ROC AUC 0,9737, se **rechaza la hipótesis nula** ($H_0: \text{Accuracy} \leq 0{,}50$) con amplio margen. El modelo demuestra capacidad discriminativa significativamente superior al azar.

### 9.4 Comparativo v1 → v2 → v3

| Métrica | v1 (baseline) | v2 (ResNet) | v3 (EfficientNetB0) |
|---------|:---:|:---:|:---:|
| Exactitud | 68,43 % | 91,45 % | **92,28 %** |
| Precisión | 86,32 % | 97,94 % | **91,27 %**¹ |
| Sensibilidad | 48,10 % | 86,05 % | **93,17 %** |
| F1-Score | 61,77 % | 91,61 % | **92,21 %** |
| ROC AUC | 0,8198 | 0,983 | **0,9737**² |
| MCC | −0,2673 | 0,837 | **0,8458** |
| Arquitectura | CNN custom | ResNet-inspired | EfficientNetB0 + Adapter |
| Parámetros | ~2 M | 5.032.385 | **4.396.112** |
| Test set | 396 imgs (1 país) | 1.017 imgs (1 país) | **3.619 imgs (4 países)** |
| Fuga de datos | Sí | Parcial³ | **No (0 %)** |

¹ Precisión menor por test 3,6× mayor y más diverso. ² AUC sobre dataset más desafiante. ³ La v2 tenía 62,5 % leakage descubierto en v3.

### 9.5 Limitaciones conocidas

1. **Resolución temporal:** ASTER GED es un promedio temporal; no captura variaciones estacionales.
2. **Validación en campo:** Las predicciones no han sido contrastadas con prospección in situ.
3. **Búsqueda de hiperparámetros:** Manual. Optuna/Bayesian search podría mejorar resultados.
4. **Etiquetado:** Basado en literatura geológica → sesgo hacia zonas ya conocidas.
5. **Generalización fuera de la Región Andina:** No evaluada.

---

## 10. Bugs Descubiertos y Corregidos

### 10.1 Auditoría v1 → v2: 28 bugs (4 CRITICAL, 4 HIGH, 10 MEDIUM, 10 LOW)

#### Bugs CRITICAL

| # | Bug | Impacto | Corrección |
|:-:|-----|---------|------------|
| 1 | **NoData (-9999) no filtrado** | Media y std completamente distorsionadas | Filtrado + interpolación mediana por banda |
| 2 | **Doble normalización** (z-score + Rescaling 1/255) | Señal aplastada a rango [-0,008, +0,008] | Eliminada capa Rescaling; solo z-score |
| 3 | **Data leakage** (train_test_split sin agrupar) | Augmentaciones del mismo tile en train y test | GroupShuffleSplit por imagen original |
| 4 | **Doble regularización L2** (kernel_regularizer + AdamW weight_decay) | Modelo sobre-regularizado | Solo AdamW weight_decay |

#### Bugs HIGH

| # | Bug | Corrección |
|:-:|-----|------------|
| 5 | RandomContrast sobre datos z-score (espera [0,1]) | Eliminado RandomContrast |
| 6 | Doble augmentación (offline 30× + online Keras) | Desactivado aumento online |
| 7 | ReduceLROnPlateau conflicta con AdamW | Reemplazado por CosineDecay |
| 8 | Excepciones silenciosas en predicción | Logging completo con traceback |

#### Bugs MEDIUM (9–18)

Incluyen: hardcoded GEE project ID (centralizado en config.py), augmentaciones float64→float32, carga completa en RAM (particionado), distancia euclidiana→Haversine, curvas ROC sintéticas→datos reales, métricas hardcodeadas→lectura dinámica, keywords incompletas, BatchNorm faltante en shortcut, training=False hardcodeado.

#### Bugs LOW (19–28)

Incluyen: documentación incorrecta, valores hardcodeados en UI, rutas relativas, falta de retry en descargas GEE, R2 Score reemplazado por MCC, métricas None mostraban "0.0000".

### 10.2 Auditoría v2 → v3: 6 bugs (3 CRITICAL, 2 HIGH, 1 CRITICAL adicional)

| # | Sev. | Bug | Impacto | Corrección |
|:-:|:----:|-----|---------|------------|
| 1 | **CRITICAL** | Clases segregadas en particiones .npy | Batches de una sola clase → modelo no aprende | Pre-shuffle en disco (seed=42) |
| 2 | **CRITICAL** | CosineDecay con decay_steps hardcodeado | LR llegaba a mínimo en época 12 de 100 | CosineDecay dinámico basado en steps_per_epoch |
| 3 | **HIGH** | Val/test cargados completos en RAM | ~5 GB innecesarios, OOM | Generadores por partes |
| 4 | **HIGH** | Generador con global shuffle + LRU cache | 16 s/step por cache thrashing | Part-sequential con intra-part shuffle |
| 5 | **CRITICAL** | **Normalización z-score per-image** | Destruye info absoluta → AUC ≈ 0,51 | Normalización **global** por banda (Welford) |
| 6 | **CRITICAL** | **62,5 % data leakage entre splits** | V2 memorizaba augmentaciones, métricas infladas | `resplit_data.py`: GroupShuffleSplit por zona geográfica base (407 zonas, 0 % leakage) |

#### Detalle del Bug #5: Normalización per-image

```python
# ANTES (bug): Cada imagen queda con mean≈0, std≈1 → zonas calientes y frías idénticas
for i in range(bands):
    mean = np.mean(image[:,:,i])    # ← mean de ESTA imagen
    std  = np.std(image[:,:,i])     # ← std de ESTA imagen

# DESPUÉS (correcto): Estadísticas globales del dataset completo
for i in range(bands):
    normalized[:,:,i] = (image[:,:,i] - global_means[i]) / global_stds[i]
```

Diagnóstico: AUC ≈ 0,51 (azar) durante 11 épocas. El modelo no podía aprender porque las clases eran estadísticamente indistinguibles.

#### Detalle del Bug #6: Data Leakage 62,5 %

El split original no consideraba los sufijos de grilla (`_grid_r0_c1`), permitiendo que augmentaciones de la **misma zona geográfica** aparecieran en train, val y test simultáneamente. Diagnóstico con `_check_leakage.py`:

```
Leaking SAMPLES:
  Val samples from leaked zones:  2.136/3.414 → 62,5%
  Test samples from leaked zones: 2.088/3.342 → 62,5%
```

Corrección: `resplit_data.py` extrae la zona base eliminando sufijos `_aug*` y `_grid_*`, y aplica GroupShuffleSplit por zona. Resultado: **407 zonas, 0 % solapamiento**.

### 10.3 Modelos experimentales fallidos (antes de descubrir el leakage)

| Modelo | Arquitectura | Mejor val_acc | Problema |
|:------:|-------------|:---:|---------|
| V3 | Custom ResNet 5M params | 64 % | Data leakage |
| V4 | Custom CNN reducido | 56 % | Data leakage |
| V5 | EfficientNetB0 (primer intento) | 62 % | Data leakage + adapter pobre |
| V6 | MobileNetV2 | 58 % | Data leakage |
| **V7 (final)** | **EfficientNetB0 + Adapter** | **92,28 %** | **Datos limpios → primer intento exitoso** |

---

## 11. Evolución del Proyecto (v1 → v2 → v3)

| Aspecto | v1 (nov 2025) | v2 (feb 2026) | v3 (mar 2026) |
|---------|:---:|:---:|:---:|
| Imágenes originales | 85 | 200 | **2.019** |
| Imágenes augmentadas | 2.635 | 6.200 | **22.209** |
| Aug. por imagen | ~30 | ~30 | **10** |
| Países | Colombia | Colombia | **4 (CO+EC+PE+CL)** |
| Bandas | 7 (con bugs) | 7 (limpias) | 7 (limpias) |
| Split | train_test_split (leakage) | GroupShuffleSplit | **GroupShuffleSplit anti-leakage geográfico** |
| Normalización | z-score + Rescaling | Solo z-score (per-image) | **Solo z-score (global por banda)** |
| NoData | Sin filtrar | Filtrado | Filtrado |
| Arquitectura | CNN custom | ResNet-inspired (5M) | **EfficientNetB0 + Adapter (4,4M)** |
| Entrenamiento | CPU | CPU | **GPU RTX 4070 (WSL2)** |
| Épocas | 23 | 22 | **80 (30 + 50, 2 fases)** |
| Test Accuracy | 68,43 % | 91,45 % | **92,28 %** |
| Test Recall | 48,10 % | 86,05 % | **93,17 %** |
| ROC AUC | 0,8198 | 0,983 | **0,9737** |
| Fuga de datos | Sí | 62,5 % | **0 %** |

### Lecciones aprendidas

1. **La calidad de los datos importa más que la arquitectura.** Los 28 bugs de v1 tenían más impacto en el rendimiento que el cambio de arquitectura.
2. **La fuga de datos es silenciosa y catastrófica.** El leakage del 62,5 % no se manifestaba como un error obvio, sino como métricas artificialmente buenas.
3. **La normalización global es crítica.** Normalizar por imagen destruye la información absoluta que el modelo necesita para discriminar.
4. **Transfer Learning funciona en dominios no naturales.** Los filtros de ImageNet (bordes, texturas) son transferibles a imágenes de emisividad térmica.
5. **Auditar el pipeline completo es indispensable.** Cada bug corregido contribuyó a la mejora final.

---

## 12. Catálogo de Campos Geotérmicos

### 12.1 Colombia 🇨🇴

**Campos geotérmicos confirmados:**
| Zona | Coordenadas [lon, lat] | Tipo | Estado |
|------|------------------------|------|--------|
| Nevado del Ruiz | [-75.322, 4.895] | Alta entalpía, volcán activo | Exploración avanzada (ISAGEN/SGC) |
| Volcán Puracé | [-76.404, 2.321] | Confirmado, fumarolas | Exploración SGC |
| Paipa-Iza | [-73.112, 5.778] | Baja entalpía, aguas termales | Exploración preliminar |

**Volcanes activos con manifestaciones hidrotermales:**
Galeras [-77.360, 1.220], Cumbal [-77.880, 0.950], Azufral [-77.680, 1.085], Cerro Machín [-75.390, 4.490], Nevado del Tolima [-75.330, 4.660], Sotará [-76.590, 2.110], Doña Juana [-76.940, 1.500], Nevado del Huila [-76.030, 2.940], Chiles [-77.940, 0.820], Cerro Negro de Mayasquer [-77.960, 0.830], Cerro Bravo [-75.300, 5.090], Nevado de Santa Isabel [-75.370, 4.815], Romeral [-75.364, 4.960].

**Zonas termales:** Termales de Manizales, Coconucos, Santa Rosa de Cabal, Herveo, San Vicente Ferrer, Tabio, Choachí, Rivera (Huila), Guadalupe (Santander), San Agustín (Huila), Pitalito.

### 12.2 Ecuador 🇪🇨

**Campos geotérmicos:** Tufiño-Chiles-Cerro Negro (binacional), Chalupas, Chachimbiro, Baños de Agua Santa, El Placer, Papallacta.

**Volcanes activos:** Cotopaxi, Tungurahua, Guagua Pichincha, Reventador, Sangay, Cayambe, Antisana, Chimborazo, Quilotoa, Illiniza, Atacazo, Soche, Pululagua, Cuicocha, Imbabura, Mojanda, Cotacachi.

### 12.3 Perú 🇵🇪

**Campos geotérmicos:** Calientes (Tacna), Borateras (Tacna), Tutupaca, Ubinas, Salinas-Chivay (Colca), Río Jesús.

**Volcanes activos:** Misti, Ubinas, Sabancaya, Chachani, Huaynaputina, Ticsani, Yucamane, Tutupaca, Casiri, Coropuna, Ampato, Sara Sara, Hualca Hualca.

### 12.4 Chile 🇨🇱

**Campos geotérmicos operativos/en exploración:** Cerro Pabellón (**48 MW, operativo** — primera planta geotérmica de Sudamérica), El Tatio (mayor campo de géiseres del hemisferio sur), Puchuldiza-Tuja, Calabozos, Nevados de Chillán.

**Volcanes activos (norte Zona Volcánica Central):** Láscar, Putana, Ollagüe, San Pedro, Irruputuncu, Isluga, Guallatiri, Tacora, Parinacota, Licancabur, Surire.

**Volcanes activos (sur Zona Volcánica Sur):** Tolhuaca, Cordón Caulle, Villarrica, Llaima, Copahue, Antuco, Callaqui.

### 12.5 Zonas de control (negativas)

Para cada país se seleccionaron zonas sin actividad volcánica ni geotermal conocida, en regiones geológicamente estables:

- **Colombia (89 zonas):** Llanos Orientales, Amazonía, Costa Caribe, Altiplano Cundiboyacense, Chocó.
- **Ecuador (20 zonas):** Guayaquil, Machala, Esmeraldas, Quevedo, Lago Agrio, Puyo, Loja, Santa Elena.
- **Perú (20 zonas):** Lima, Piura, Chiclayo, Ica, Nasca, Cusco, Puno, Iquitos, Pucallpa.
- **Chile (20 zonas):** Santiago, Valparaíso, Concepción, La Serena, Copiapó, Antofagasta, Punta Arenas.

---

## Tecnologías y Herramientas

| Categoría | Herramientas |
|-----------|-------------|
| Deep Learning | TensorFlow 2.20.0, Keras 3.12+ |
| GPU / Aceleración | NVIDIA RTX 4070 12 GB VRAM, CUDA 12.x, cuDNN, Mixed Precision float16 |
| Procesamiento | NumPy, pandas, scikit-learn, scikit-image, OpenCV, SciPy, rasterio |
| Datos geoespaciales | Google Earth Engine API, NASA ASTER GED AG100 v003 |
| Visualización | Matplotlib, Seaborn, TensorBoard, Plotly, Folium |
| Interfaz web | Streamlit, streamlit-folium |
| Control de versiones | Git, GitHub |
| Lenguaje | Python 3.12.12 (WSL2 GPU), Python 3.10.11 (Windows desarrollo) |
| SO | Windows 11 (desarrollo), WSL2 Ubuntu 22.04 (entrenamiento GPU) |

---

## Referencias

- Abrams, M. y Hook, S. J. (2002). *ASTER User Handbook, Version 2*. JPL.
- Alfaro, C. (2015). Evaluación del potencial geotérmico de Colombia. SGC.
- Bona, P. y Coviello, M. (2016). *Valoración y gobernanza de los proyectos geotérmicos en América del Sur*. CEPAL.
- Coolbaugh, M. et al. (2007). Detection of geothermal anomalies using ASTER thermal infrared images. *Remote Sensing of Environment*, 106(3), 350–359.
- DiPippo, R. (2015). *Geothermal Power Plants*. 4ª ed. Elsevier.
- Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*, 770–778.
- Hulley, G. C. et al. (2015). The ASTER Global Emissivity Dataset. *Geophysical Research Letters*, 42(19), 7966–7976.
- LeCun, Y. et al. (2015). Deep learning. *Nature*, 521(7553), 436–444.
- Lahsen, A. (1982). Upper Cenozoic volcanism and tectonism in the Andes of northern Chile. *Earth-Science Reviews*, 18(3).
- Servicio Geológico Colombiano. (2023). *Mapa de potencial geotérmico de Colombia*.
- Shorten, C. y Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60.
- Tan, M. y Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*, 6105–6114.
- Zhang, H. et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
- Zhu, X. X. et al. (2017). Deep learning in remote sensing. *IEEE GRSM*, 5(4), 8–36.

---

*Universidad de San Buenaventura — Bogotá | Proyecto de Grado 2025–2026*
