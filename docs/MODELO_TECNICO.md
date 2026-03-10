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
   - 5.1 [Analogía cotidiana: Entendiendo el modelo sin ser experto](#51-analogía-cotidiana-entendiendo-el-modelo-sin-ser-experto)
   - 5.2 [¿Cómo aprende el modelo?](#52-cómo-aprende-el-modelo-el-ciclo-de-aprendizaje)
   - 5.3 [Glosario de términos técnicos](#53-glosario-de-términos-técnicos)
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

### 5.1 Analogía cotidiana: Entendiendo el modelo sin ser experto

Para entender cómo funciona nuestro modelo, imagina la siguiente situación:

Queremos entrenar a un **médico radiólogo** para que identifique una enfermedad rara mirando un tipo especial de radiografía que él nunca ha visto antes. ¿Cómo lo haríamos?

---

**Paso 1 — Estudiar medicina general (preentrenamiento en ImageNet):**

Antes de especializarse, el médico estudió medicina general durante años: aprendió a reconocer huesos, tejidos, bordes, texturas, contrastes — habilidades visuales que sirven para *cualquier* especialidad. Nuestro modelo hizo exactamente lo mismo: antes de saber nada de geotermia, ya "estudió" **1,2 millones de fotos normales** (perros, gatos, autos, paisajes, comida) y aprendió a detectar **bordes, texturas, formas, contrastes y patrones visuales generales**.

Aunque esas fotos no tienen nada que ver con volcanes ni satélites, las habilidades de "ver" son universales: un borde es un borde, ya sea en la foto de un gato o en una imagen satelital. A toda esa base de conocimiento visual la llamamos **backbone** (que literalmente significa "columna vertebral" — es la estructura principal del modelo).

**Paso 2 — Ponerle gafas especiales (Channel Adapter):**

Nuestras "radiografías" no son fotos normales. Las fotos normales tienen **3 colores** (Rojo, Verde, Azul = RGB). Pero nuestras imágenes satelitales tienen **7 "colores" invisibles** al ojo humano: 5 bandas de emisividad térmica (calor que emite la superficie), la temperatura del suelo, y un índice de vegetación. Es información que un humano no puede "ver" — pero que contiene pistas sobre actividad geotérmica.

El problema: el médico fue entrenado mirando fotos de 3 colores, y ahora le damos radiografías de 7 colores. Su cerebro no puede procesarlas directamente. La solución: **unas gafas especiales** que traducen automáticamente los 7 colores invisibles a los 3 colores que su cerebro ya sabe interpretar. Estas gafas son el **Channel Adapter** — un módulo pequeño que aprende por sí solo cuál es la mejor manera de combinar las 7 bandas del satélite en 3 canales útiles.

**Paso 3 — Estudiar la enfermedad sin olvidar medicina — Fase 1 (congelamiento):**

Ahora el médico empieza a ver radiografías de pacientes con y sin la enfermedad rara. Pero le damos una instrucción clave: **"NO olvides nada de lo que ya sabes de anatomía general"**. Le ponemos un **candado** a todos sus conocimientos previos: no se pueden borrar ni modificar. Solo le permitimos aprender dos cosas nuevas:

1. Cómo ajustar las gafas especiales (el **adapter**)
2. Cómo dar el diagnóstico final (el **classification head** — la "cabeza" que decide: ¿zona geotérmica sí o no?)

Esto es lo que llamamos **congelar el backbone** (*freeze*): los millones de conocimientos visuales que aprendió con fotos normales quedan **protegidos e inmutables**. Solo se entrenan las partes nuevas. Dura **30 rondas completas** (épocas) por todos los casos de estudio.

**Paso 4 — Ajustar el ojo clínico — Fase 2 (fine-tuning):**

Después de 30 rondas, el médico ya entiende bastante bien las nuevas radiografías. Entonces le damos un permiso limitado: **"Puedes ajustar ligeramente tu conocimiento general — pero con MUCHO cuidado"**. Le quitamos el candado solo a las **últimas 39 capas** de su cerebro (las más especializadas y superficiales) y le permitimos modificarlas un poco. Esto se llama **fine-tuning** (ajuste fino).

¿Por qué solo las últimas capas? Porque en las redes neuronales las primeras capas reconocen cosas muy básicas (bordes, líneas, contrastes) que son iguales en cualquier dominio. Las últimas capas son las más "especializadas" — son las que más necesitan adaptarse del mundo de fotos de gatos al mundo de emisividad térmica.

¿Y por qué "con cuidado"? Porque su ritmo de cambio es **10 veces más lento** que antes: pasamos de una velocidad (learning rate) de 0,001 a 0,0001. No queremos que el médico destroce todo lo que aprendió en la universidad por apurarse.

**Paso 5 — El diagnóstico (predicción):**

Ahora, frente a una imagen satelital nueva que nunca ha visto, el modelo dice: *"Hay un 87% de probabilidad de que esta zona sea geotérmica"*. No dice "sí" o "no" rotundamente — da una **probabilidad entre 0% y 100%**. Si supera el 50%, la clasificamos como positiva (potencial geotérmico). Si no, negativa.

---

**Tabla de correspondencias — de la analogía al modelo real:**

| El médico radiólogo | Término técnico | En nuestro modelo concreto |
|---|---|---|
| Estudiar medicina general (fotos normales) | **Preentrenamiento (ImageNet)** | EfficientNetB0 entrenado con 1,2 millones de fotos |
| Su cerebro visual ya entrenado | **Backbone** (columna vertebral) | Las 237 capas internas de EfficientNetB0 |
| Gafas que traducen 7 → 3 colores | **Channel Adapter** | 2 capas Conv2D que proyectan 7 bandas → 3 canales |
| Su criterio final de diagnóstico | **Classification Head** (cabeza) | Dense(256) → Dense(64) → Dense(1, sigmoid) |
| "No olvides anatomía" (candado) | **Congelar (freeze)** | `layer.trainable = False` en el backbone |
| "Ahora ajusta un poco tu ojo" | **Fine-tuning** (ajuste fino) | Descongelar últimas 39 capas |
| Ritmo de cambio cauteloso | **Learning Rate bajo** | De 0,001 (Fase 1) a 0,0001 (Fase 2) |
| Cada pase completo por todos los casos | **Época (epoch)** | 1 revisión de las 15.037 imágenes de entrenamiento |
| Ver 32 radiografías y corregirse | **Batch (lote)** | 32 imágenes procesadas antes de ajustar pesos |
| Examen con casos que no estudió | **Validación / Test** | 3.553 + 3.619 imágenes nunca usadas para entrenar |
| "Me equivoqué, ¿en qué fallé?" | **Backpropagation** | El error se propaga hacia atrás para saber qué ajustar |
| "¿Cuánto me equivoqué?" | **Loss (pérdida)** | Binary Cross-Entropy: mide distancia predicción vs. realidad |
| "¿Cómo me corrijo?" | **Optimizador (AdamW)** | Algoritmo que decide cómo ajustar cada peso |

### 5.2 ¿Cómo aprende el modelo? (El ciclo de aprendizaje)

El modelo **no se programa con reglas** como "si la temperatura es mayor a X grados, entonces es geotérmico". En cambio, aprende **por repetición y corrección de errores**, de forma muy similar a como estudia un ser humano. El ciclo completo tiene 5 pasos que se repiten miles de veces:

#### Paso 1 — Ver (Forward Pass)

El modelo recibe un lote de 32 imágenes satelitales. Cada imagen recorre todas las capas del modelo en orden (adapter → backbone → head) y al final produce una predicción: un número entre 0 y 1 que representa la probabilidad de ser una zona geotérmica. Este recorrido "hacia adelante" se llama **forward pass** (pasada hacia adelante).

#### Paso 2 — Calificarse (Función de pérdida / Loss)

Se compara cada predicción con la **respuesta correcta** (la etiqueta que ya sabemos: 1 = geotérmico, 0 = no geotérmico). La **función de pérdida (loss)** pone una "nota" que cuantifica qué tan mal lo hizo. Usamos **Binary Cross-Entropy**, que funciona como un profesor estricto:

- Si la zona **es** geotérmica (etiqueta = 1.0) y el modelo dijo **0.95** → error pequeño, casi acertó ✓
- Si la zona **es** geotérmica (etiqueta = 1.0) y el modelo dijo **0.10** → error enorme ✗
- Si la zona **no es** geotérmica (etiqueta = 0.0) y el modelo dijo **0.85** → error enorme ✗

**La fórmula matemática:**

$$\mathcal{L} = -\left[ y \cdot \log(\hat{y}) \;+\; (1 - y) \cdot \log(1 - \hat{y}) \right]$$

Donde:
- $y$ = etiqueta real (1 si es geotérmica, 0 si no lo es)
- $\hat{y}$ = probabilidad predicha por el modelo (número entre 0 y 1, ej: 0.87)
- $\log$ = logaritmo natural (siempre negativo para valores entre 0 y 1, por eso el signo -1 al frente invierte el resultado para que la pérdida sea positiva)

**¿Por qué dos términos?**

| Caso | Qué activa | Qué se ignora |
|------|-----------|---------------|
| Zona **es** geotérmica ($y=1$) | $-\log(\hat{y})$ | El segundo término se anula porque $(1 - 1) = 0$ |
| Zona **no es** geotérmica ($y=0$) | $-\log(1 - \hat{y})$ | El primer término se anula porque $0 \cdot \log(\hat{y}) = 0$ |

En esencia: cuando el ejemplo es positivo, se penaliza que $\hat{y}$ sea bajo (el modelo no lo reconoció). Cuando es negativo, se penaliza que $\hat{y}$ sea alto (el modelo creyó que sí era geotérmica).

**Ejemplo numérico concreto:**

| Etiqueta real | Predicción | Cálculo | Pérdida |
|:---:|:---:|---|:---:|
| $y=1$ | $\hat{y}=0.95$ | $-\log(0.95) = 0.051$ | **0.051** (pequeña) |
| $y=1$ | $\hat{y}=0.10$ | $-\log(0.10) = 2.303$ | **2.303** (enorme) |
| $y=0$ | $\hat{y}=0.85$ | $-\log(1 - 0.85) = -\log(0.15) = 1.897$ | **1.897** (enorme) |
| $y=0$ | $\hat{y}=0.05$ | $-\log(1 - 0.05) = -\log(0.95) = 0.051$ | **0.051** (pequeña) |

Lo clave: la penalización crece **exponencialmente** con la confianza del error. Equivocarse estando "95% seguro" se penaliza **muchísimo más** que equivocarse estando "55% indeciso". Esto fuerza al modelo a ser honesto con su incertidumbre.

#### Paso 3 — Entender dónde se equivocó (Backpropagation)

El error calculado en el paso 2 se **propaga hacia atrás** por todas las capas del modelo, desde la salida (la predicción) hasta la entrada (la imagen), capa por capa. En cada capa se calcula: *"¿Cuánto contribuyó ESTA capa al error total?"*. Este proceso se llama **backpropagation** (retropropagación).

El resultado son los **gradientes**: vectores que indican, para cada uno de los 4,4 millones de parámetros del modelo, **en qué dirección y cuánto** debería cambiar para reducir el error. Es como si el profesor le dijera al alumno: "Te equivocaste porque no prestaste suficiente atención a la banda de temperatura y confundiste el patrón de arcillas con cuarzo".

#### Paso 4 — Corregirse (Optimizador AdamW)

El **optimizador** toma los gradientes y actualiza los **pesos** (parámetros numéricos) del modelo. La regla fundamental es:

$$w_{\text{nuevo}} = w_{\text{actual}} - \text{learning\_rate} \times \text{gradiente}$$

Es decir: cada peso se mueve un pasito en la dirección opuesta al error. El **learning rate (lr)** controla el tamaño de ese paso:

- **Muy alto** (ej: 0,1) → el modelo da saltos enormes, no converge (como un estudiante que cambia de opinión completamente después de cada examen)
- **Muy bajo** (ej: 0,000001) → aprende extremadamente lento, puede tardarse una eternidad
- **Justo** (ej: 0,001 en Fase 1 → 0,0001 en Fase 2) → converge de forma estable y progresiva

Nuestro optimizador **AdamW** es "inteligente" porque además:
- Adapta la velocidad **individualmente** para cada parámetro (los que necesitan más ajuste se mueven más)
- Aplica **weight decay** (decaimiento de pesos): penaliza los pesos que crecen demasiado, forzando al modelo a usar soluciones simples en vez de memorizar

#### Paso 5 — Repetir (miles de veces)

Este ciclo (ver → calificarse → entender error → corregirse) se repite miles de veces:

| Concepto | Valor en nuestro modelo |
|----------|:-:|
| Imágenes por batch | 32 |
| Steps por época | ~470 (15.037 ÷ 32) |
| Épocas totales | 80 (30 + 50) |
| **Total de ciclos de corrección** | **~37.600** |

A medida que avanzan los ciclos, la **pérdida disminuye** (el modelo se equivoca menos) y la **exactitud en validación sube** (el modelo generaliza mejor). Si después de 10–15 épocas la validación no mejora, el **EarlyStopping** detiene el entrenamiento para evitar **overfitting** (sobreajuste): cuando el modelo "memoriza" los ejemplos específicos de entrenamiento en vez de aprender patrones generalizables, como un estudiante que memoriza las respuestas del examen de práctica pero no entiende la materia.

### 5.3 Glosario de términos técnicos

| Término | Explicación |
|---------|-------------|
| **Backbone** | La "columna vertebral" del modelo: la red neuronal principal (EfficientNetB0, 237 capas) que contiene los filtros visuales preentrenados. Sabe detectar bordes, texturas, formas y patrones. Es la parte más grande y la que más "sabe". |
| **Congelar (Freeze)** | Bloquear los pesos de ciertas capas para que **no cambien** durante el entrenamiento (`layer.trainable = False`). Es como ponerle candado a un conocimiento para protegerlo mientras se aprenden cosas nuevas. |
| **Descongelar (Unfreeze)** | Quitar el candado: permitir que los pesos vuelvan a modificarse (`layer.trainable = True`). Se hace de forma selectiva (solo las últimas capas) y con cautela (learning rate bajo). |
| **Fine-tuning** | "Ajuste fino": descongelar parte del backbone y re-entrenarlo con un learning rate muy bajo. No se aprende desde cero — se **refinan** conocimientos existentes para adaptarlos al nuevo dominio. |
| **Pesos (Weights)** | Los 4,4 millones de números decimales que componen el modelo. Cada peso se ajusta un poco en cada ciclo de entrenamiento. Juntos, definen todo lo que el modelo "sabe". Cuando guardamos el modelo (.keras), guardamos estos números. |
| **Época (Epoch)** | Una pasada completa por TODAS las imágenes de entrenamiento (15.037 imágenes). Se necesitan muchas épocas (80 en nuestro caso) para que el modelo converja a una solución buena. |
| **Batch (Lote)** | Subconjunto de imágenes (32) procesadas simultáneamente. El modelo actualiza sus pesos después de cada batch, no después de cada imagen individual. Esto hace el entrenamiento más estable y eficiente. |
| **Learning Rate** | "Velocidad de aprendizaje": número que controla cuánto se ajustan los pesos en cada corrección. Fase 1 usa 0,001 (aprendizaje rápido); Fase 2 usa 0,0001 (aprendizaje cauteloso). Decae gradualmente con CosineDecay. |
| **Loss (Pérdida)** | Número que cuantifica qué tan mal predice el modelo. Mientras menor sea, mejor. Todo el entrenamiento busca **minimizar** este número. En nuestro caso: Binary Cross-Entropy. |
| **Optimizador (AdamW)** | Algoritmo que decide cómo actualizar los pesos usando los gradientes. AdamW adapta la velocidad de cada parámetro individualmente y penaliza pesos excesivamente grandes (weight decay). |
| **Backpropagation** | "Retropropagación": algoritmo que calcula, para cada peso, cuánto contribuyó al error. Funciona hacia atrás: desde el error final hacia la imagen de entrada, capa por capa. Es lo que permite al modelo saber **qué** necesita ajustar. |
| **Gradiente** | La "dirección de corrección" para un peso específico. Indica hacia dónde y cuánto debe moverse para reducir el error. Es el resultado del backpropagation. |
| **Forward Pass** | Pasada "hacia adelante": la imagen entra por la primera capa, recorre todas las capas hasta la última, y produce una predicción. Es el paso donde el modelo "mira" la imagen. |
| **Overfitting** | "Sobreajuste": cuando el modelo memoriza los ejemplos de entrenamiento en vez de aprender patrones generalizables. Se detecta cuando la exactitud en entrenamiento es alta pero en validación es baja. Las técnicas de regularización lo previenen. |
| **Regularización** | Conjunto de técnicas (Dropout, Weight Decay, MixUp, Label Smoothing, EarlyStopping) que fuerzan al modelo a **generalizar** en vez de memorizar. Son como reglas de estudio que evitan que el alumno dependa de la memoria a corto plazo. |
| **Dropout** | Durante el entrenamiento, se "apagan" aleatoriamente un % de neuronas (30–50%). Esto fuerza al modelo a no depender de ninguna neurona individual — como un equipo donde cualquier miembro puede faltar y los demás compensan. En la predicción final, todas funcionan. |
| **Channel Adapter** | Módulo pequeño (2 capas) que traduce las 7 bandas satelitales a 3 canales que el backbone entiende. Aprende la proyección óptima automáticamente, sin intervención humana. |
| **Classification Head** | Las capas finales del modelo que toman la decisión. Reciben 1.280 características extraídas por el backbone y producen un solo número (0–1): la probabilidad de potencial geotérmico. |
| **Transfer Learning** | "Aprendizaje por transferencia": reutilizar un modelo preentrenado (en fotos normales) y adaptarlo a otro dominio (imágenes satelitales). Reduce drásticamente el tiempo y la cantidad de datos necesarios. |
| **Batch Normalization** | Capa que normaliza las activaciones internas de la red en cada paso, estabilizando la convergencia. En la Fase 2 se mantiene **congelada** para preservar las estadísticas que aprendió con ImageNet. |
| **EarlyStopping** | Mecanismo de seguridad: si el rendimiento en validación no mejora durante 10–15 épocas seguidas, se detiene el entrenamiento automáticamente para evitar overfitting. |
| **CosineDecay** | Estrategia donde el learning rate empieza alto y va decayendo suavemente siguiendo una curva de coseno, hasta llegar a un mínimo. Permite aprender rápido al inicio y afinar al final. |

### 5.4 ¿Qué son los 4,4 millones de parámetros?

Un **parámetro** es simplemente un número decimal almacenado dentro del modelo. Nada más. Cuando guardamos el modelo en un archivo `.keras` (60 MB), lo que se guarda es exactamente esa lista de 4.396.112 decimales.

#### ¿Para qué sirve cada número?

Cada parámetro es un **peso** (`w`) o un **sesgo** (`b`) dentro de una operación matemática elemental:

$$\text{salida} = \text{entrada} \times w + b$$

`w` amplifica o atenúa la señal que entra. `b` la desplaza hacia arriba o hacia abajo. El entrenamiento consiste en ajustar esos valores miles de veces hasta que produzcan la salida correcta.

#### ¿Dónde están los 4,4 millones?

No están en un solo lugar — están repartidos en cientos de capas:

| Parte del modelo | Parámetros aprox. | % del total |
|---|:-:|:-:|
| **Backbone EfficientNetB0** | ~4.050.000 | 92,1 % |
| **Channel Adapter** (7 → 3 canales) | ~1.400 | 0,03 % |
| **Classification Head** (256 → 64 → 1) | ~344.000 | 7,8 % |
| **Total** | **4.396.112** | 100 % |

#### Ejemplo concreto de cómo se acumulan

**En el Channel Adapter:** la primera capa convolucional tiene 16 filtros de tamaño 3×3 aplicados a 7 bandas. Cada filtro es una grilla de 9 números. Eso son `7 × 16 × 9 = 1.008 pesos`. Esos 1.008 números aprenden a combinar las 7 bandas satelitales de la forma más útil posible.

**En el Backbone:** hay capas convolucionales mucho más grandes. Una capa que transforma 40 canales en 80 canales con filtros 3×3 tiene `3 × 3 × 40 × 80 = 28.800 pesos`. Multiplicado por cientos de capas, los números se acumulan hasta los ~4 millones.

**En el Classification Head:** las capas densas tienen una conexión entre cada neurona de entrada y cada neurona de salida. Una capa Dense(256) que recibe 1.280 valores tiene `1.280 × 256 = 327.680 pesos` — uno por cada par de neuronas conectadas.

#### La intuición clave

Cada parámetro es un **dial de sintonía**. Antes del entrenamiento:
- Los del backbone vienen en valores heredados de ImageNet (ya son útiles)
- Los del adapter y la cabeza parten de valores aleatorios

Durante el entrenamiento, el algoritmo ajusta cada dial un pequeñísimo paso, ~37.600 veces, hasta que los 4,4 millones de diales en conjunto producen la respuesta correcta para el 92% de los casos. Al final, esos números **codifican todo el conocimiento del modelo**: qué bandas importan más, qué texturas son características de zonas geotérmicas, cómo combinar señales de temperatura con señales mineralógicas. Nadie programó esas reglas explícitamente — emergieron solas del proceso de ajuste.

### 5.5 ¿Qué es un tensor?

Un **tensor** es simplemente una forma de organizar números en varias dimensiones. Es la estructura de datos fundamental sobre la que opera todo el modelo.

Para entenderlo sin tecnicismos, imagina estas estructuras cotidianas:

| Estructura | Dimensiones | Ejemplo |
|---|---|---|
| Un número suelto | 0D (escalar) | La temperatura hoy: `28.5` |
| Una lista de números | 1D (vector) | Las temperaturas de 7 días: `[28, 29, 31, 27, 30, 28, 32]` |
| Una tabla de números | 2D (matriz) | Una hoja de Excel con filas y columnas |
| Un cubo de números | 3D (tensor) | Una imagen: alto × ancho × canales de color |
| Varios cubos apilados | 4D (tensor) | Un lote de imágenes: cantidad × alto × ancho × canales |

#### La imagen ASTER como tensor

Cada imagen ASTER que recibe nuestro modelo es un tensor 3D:

```
Forma: (224, 224, 7)
         │     │   └── 7 bandas espectrales
         │     └────── 224 píxeles de ancho
         └──────────── 224 píxeles de alto
```

Eso son `224 × 224 × 7 = 351.232 números` por imagen. Un número por cada banda en cada píxel.

Cuando el modelo procesa un **batch** de 32 imágenes a la vez, el tensor tiene 4 dimensiones:

```
Forma: (32, 224, 224, 7)
        │    │     │   └── 7 bandas
        │    │     └────── 224 ancho
        │    └──────────── 224 alto
        └───────────────── 32 imágenes en el lote
```

Eso son `32 × 351.232 = 11.239.424 números` procesados simultáneamente en la GPU en cada step.

#### ¿Por qué se llaman "tensores"?

El término viene de la física/matemática donde un tensor es una generalización de vectores y matrices a cualquier número de dimensiones. En deep learning adoptaron el mismo nombre porque es exactamente lo que son: datos numéricos organizados en estructuras multidimensionales. TensorFlow, literalmente, significa **"flujo de tensores"** — los datos fluyen como tensores a través de las capas de la red.

### 5.6 Redes Neuronales Convolucionales (CNN)

Las CNN son arquitecturas de aprendizaje profundo especializadas en datos con estructura de cuadrícula (imágenes). Su poder radica en tres operaciones:

**Convolución (Conv2D):** Aplica filtros (kernels) aprendibles que detectan patrones locales:

$$\text{Output}(i,j) = \sum_{m,n} \text{Input}(i+m, j+n) \times \text{Kernel}(m,n) + b$$

Los filtros de las primeras capas aprenden bordes y texturas; los de capas profundas aprenden patrones de alto nivel (anomalías térmicas, composición mineral).

**Pooling (MaxPooling2D):** Reduce dimensiones espaciales conservando las características más relevantes. Aporta invariancia a traslaciones menores.

**Activación (ReLU):** Introduce no linealidad: $f(x) = \max(0, x)$. Permite al modelo aprender relaciones complejas.

### 5.7 Redes residuales (ResNet) — usadas en v2

He et al. (2016) introdujeron las **conexiones residuales (skip connections)**: el gradiente fluye directamente a través de las capas, evitando el problema de degradación en redes profundas.

En un bloque residual:
$$y = F(x, \{W_i\}) + x$$

donde $F$ es la transformación del camino principal y $x$ es la entrada transmitida por el atajo. La red aprende la función residual $F(x) = y - x$.

### 5.8 Transfer Learning — usado en v3

Consiste en **reutilizar pesos** de un modelo preentrenado en un dominio fuente (ImageNet, 1,2M imágenes) y adaptarlos al dominio objetivo. Las primeras capas aprenden características genéricas (bordes, texturas) que son **transferibles entre dominios**; las capas superiores se especializan.

**Estrategia de entrenamiento en 2 fases:**
1. **Backbone congelado:** Solo se entrenan el adapter y el clasificador
2. **Fine-tuning:** Se descongelan las últimas capas del backbone con un learning rate reducido

### 5.9 EfficientNet (Tan & Le, 2019)

EfficientNet es una familia de arquitecturas de redes neuronales diseñada por investigadores de Google. La idea central: para hacer una red más precisa hay tres formas de hacerla "más grande":

| Dimensión | Qué implica |
|-----------|------------|
| **Profundidad** | Añadir más capas |
| **Ancho** | Más neuronas/filtros por capa |
| **Resolución** | Darle imágenes más grandes |

EfficientNet descubrió que si escala los **tres simultáneamente** con una proporción matemática fija (**compound scaling**), se obtiene el máximo rendimiento con el mínimo número de parámetros. La familia va de B0 (más pequeña) a B7 (más grande).

**EfficientNetB0** es la variante base. Tiene:
- **237 capas internas**
- **4,0 millones de parámetros** (ResNet50 tiene 25M con peor rendimiento en ImageNet)
- Preentrenado en **ImageNet** (1,2M imágenes, 1.000 clases)
- Input esperado: imágenes **224×224×3 canales** (RGB) — de ahí la necesidad del Channel Adapter

Su bloque fundamental es el **MBConv** (Mobile Inverted Bottleneck):

1. **Convoluciones depthwise-separable:** En vez de aplicar un filtro 3×3 a todos los canales a la vez (costoso), primero aplica un filtro por canal por separado (*depthwise*) y luego combina los resultados con filtros 1×1 (*pointwise*). Mismo resultado, ~8× menos cálculo.

2. **Squeeze-and-Excitation (SE):** Mecanismo de atención que aprende a asignar un **peso de importancia a cada canal**. Primero "comprime" la información espacial (squeeze: promedio global), luego genera pesos entre 0 y 1 para cada canal (excitation: dos capas densas), y multiplica cada canal por su peso. Esto permite que el modelo preste más atención a las bandas que más aportan en cada contexto.

3. **Skip connections internas:** Igual que en ResNet — la entrada del bloque se suma a la salida, facilitando el flujo del gradiente durante el entrenamiento.

### 5.10 TensorFlow y Keras: el motor y el volante

Son dos capas del mismo stack tecnológico, una encima de la otra:

**TensorFlow** (Google, 2015) es el **motor de bajo nivel**. Gestiona todo lo que la GPU necesita:
- Operaciones matemáticas sobre tensores (matrices multidimensionales)
- Distribución del cómputo en los núcleos CUDA de la GPU
- Backpropagation automático (autodiff): calcula gradientes sin que el programador los derive manualmente
- Precisión mixta float16: decide qué operaciones van en 16 bits y cuáles en 32 bits

El programador casi nunca lo llama directamente — trabaja por debajo de forma transparente.

**Keras** (integrado en TensorFlow desde TF 2.x) es la **interfaz de alto nivel**. Es lo que realmente se escribe en el código:
```python
model = EfficientNetB0(weights='imagenet', include_top=False)
model.fit(X_train, y_train, epochs=30)
probabilidad = model.predict(imagen)
```
Keras traduce esas instrucciones legibles a operaciones de TensorFlow. Provee todas las capas (`Conv2D`, `Dense`, `BatchNormalization`, `Dropout`), optimizadores (`AdamW`), funciones de pérdida (`BinaryCrossentropy`), y el formato de guardado `.keras`.

**En v3 concretamente:**

| Responsabilidad | Quién la hace |
|---|---|
| Definir la arquitectura (capas, conexiones) | **Keras 3.12/3.13** |
| Proveer EfficientNetB0 preentrenado | **Keras** (descarga pesos ImageNet automáticamente) |
| Ejecutar los cálculos en la RTX 4070 | **TensorFlow 2.20 + CUDA** |
| Backpropagation y cálculo de gradientes | **TensorFlow** (autodiff) |
| Escalar a float16 durante entrenamiento | **TensorFlow** (mixed precision) |
| Guardar/cargar el modelo `.keras` | **Keras** |
| Lógica de entrenamiento (`fit`, callbacks) | **Keras** |

La relación es: **Keras es el volante y el tablero** (lo que el programador toca). **TensorFlow es el motor** (lo que realmente mueve el coche).

### 5.11 Channel Adapter

Módulo convolucional diseñado para **proyectar las 7 bandas ASTER al espacio de 3 canales** esperado por EfficientNetB0 (preentrenado en RGB):

- Conv2D(16, 3×3) + BatchNorm + ReLU: expande 7 bandas a 16 mapas intermedios
- Conv2D(3, 1×1) + BatchNorm + ReLU: proyecta a 3 canales

Este adapter **aprende la proyección óptima** del espacio espectral ASTER al espacio RGB de ImageNet, en lugar de seleccionar o promediar bandas manualmente.

### 5.12 Técnicas de regularización

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

### 5.13 Mixed Precision Training (float16)

Usa aritmética de 16 bits para las operaciones forward/backward y 32 bits para la acumulación de gradientes. Beneficios:
- **Duplica el throughput** en GPU modernas (RTX 4070)
- **Reduce consumo de VRAM** (~50 %)
- Sin pérdida de precisión (loss scaling automático)

### 5.14 Global Average Pooling

En lugar de aplanar (Flatten) la salida del backbone (que generaría millones de parámetros), se calcula el **promedio por canal**:

```
7×7×1280 → Promedio por canal → 1280 valores
```

Reduce parámetros drásticamente y es menos propenso a overfitting.

### 5.15 CUDA y la GPU: ¿por qué el entrenamiento es 45× más rápido?

Entrenar el modelo en la CPU tomaba varios días. Con la GPU RTX 4070 tomó pocas horas. La diferencia está en cómo están construidas CPU y GPU.

**CPU (procesador central):**
- Tiene **8–16 núcleos** (en PCs modernas)
- Cada núcleo es muy potente y versátil — puede hacer cualquier tarea
- Diseñada para tareas **secuenciales** (una instrucción tras otra)
- Velocidad de reloj alta (~4–5 GHz)

**GPU (tarjeta gráfica):**
- Tiene **miles de núcleos pequeños** (la RTX 4070 tiene 5.888 núcleos CUDA)
- Cada núcleo es más simple, pero hay miles trabajando **en paralelo**
- Diseñada para tareas **masivamente paralelas** (mover millones de píxeles a la vez)
- Velocidad de reloj menor (~2 GHz), pero miles de operaciones simultáneas

#### ¿Por qué esto importa en deep learning?

Las operaciones del modelo (multiplicaciones de matrices, convoluciones) son **perfectamente paralelizables**: cada píxel de la imagen, cada neurona de la red, puede procesarse independientemente de las demás. La GPU puede hacer esas 11 millones de multiplicaciones de un batch **todas al mismo tiempo**. La CPU las haría una por una (o en grupos de 8–16).

```
CPU:  [núcleo 1] → [2] → [3] ... (8 a la vez, una tras otra)
GPU:  [5.888 núcleos CUDA] todos a la vez → listo
```

**CUDA** (Compute Unified Device Architecture) es el sistema de programación de NVIDIA que permite escribir código que se ejecuta en esos 5.888 núcleos. TensorFlow ya viene integrado con CUDA — cuando detecta una GPU compatible, automáticamente envía todos los cálculos a ella.

#### En números concretos de v3

| Medida | CPU | GPU RTX 4070 |
|---|---|---|
| Tiempo por step (batch de 32) | ~15.600 ms | **347 ms** |
| Factores de mejora | — | **45× más rápido** |
| Tiempo total de entrenamiento | ~días | **horas** |
| VRAM utilizada | RAM normal | 12 GB VRAM dedicada |

#### Mixed Precision + CUDA = máxima eficiencia

Además de los núcleos CUDA, las GPU modernas tienen **Tensor Cores** especializados en multiplicaciones de matrices en float16. Al combinar Mixed Precision Training con Tensor Cores, la RTX 4070 puede hacer **el doble de operaciones por segundo** que en float32, reduciendo además el uso de VRAM a la mitad.

#### WSL2: ¿por qué entrenar en Linux estando en Windows?

El entrenamiento se realizó en un PC con **Windows 11**, pero usamos **WSL2** (Windows Subsystem for Linux 2) con Ubuntu 22.04. ¿Por qué no entrenar directamente en Windows?

**WSL2** es una capa de compatibilidad desarrollada por Microsoft que permite correr un **kernel de Linux real** dentro de Windows, sin arrancar una máquina virtual completa. No es una emulación — es Linux de verdad, con acceso directo al hardware.

La razón de usarlo es que CUDA y TensorFlow tienen soporte oficial y mucho más estable en Linux que en Windows:

| Aspecto | Windows nativo | WSL2 Ubuntu |
|---|---|---|
| Soporte CUDA/TensorFlow | Parcial, con problemas frecuentes | **Oficial y completo** |
| Drivers GPU compatibles | Versiones específicas requeridas | **Driver Windows → acceso directo** |
| Rendimiento GPU | Bueno | **Igual o mejor** |
| Herramientas de ML (pip, conda) | Funciona | **Funciona mejor** |
| Estabilidad en entrenamiento largo | Ocasionalmente falla | **Estable** |

**¿Cómo funciona técnicamente?**

WSL2 no emula hardware — usa **virtualización ligera** (Hyper-V) para correr un kernel Linux real. NVIDIA desarrolló drivers especiales que permiten que ese kernel Linux acceda directamente a la GPU física de Windows. Desde Ubuntu dentro de WSL2, el sistema "ve" la RTX 4070 como si fuera una GPU nativa de Linux.

```
Windows 11 (host)
├── Driver NVIDIA (Windows)
│    └── Expone la GPU a WSL2 via passtrough
└── WSL2 (kernel Linux Ubuntu 22.04)
     └── Python 3.12 + TensorFlow 2.20 + CUDA
          └── RTX 4070 → 5.888 núcleos CUDA disponibles ✅
```

**En la práctica:** se abre una terminal Ubuntu en Windows, se navega al proyecto, y se ejecuta `python train_model_v7.py`. TensorFlow detecta la GPU automáticamente y el entrenamiento arranca. Desde fuera, parece que se está usando Windows normalmente — por dentro, todo el stack de ML corre en Linux.



Hay dos funciones de activación comunes para la capa de salida de una clasificación:

**Softmax** — se usa cuando hay **múltiples clases excluyentes**. Por ejemplo, clasificar un número entre 10 dígitos (0–9). Softmax distribuye la probabilidad entre todas las clases y fuerza a que sumen exactamente 1,0:
```
Softmax([2.1, 0.3, 1.8]) → [0.65, 0.08, 0.27]  (suman 1.0)
```

**Sigmoid** — se usa para **clasificación binaria** (dos opciones: sí o no). Toma cualquier número y lo convierte en una probabilidad independiente entre 0 y 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Nuestro problema es binario: ¿zona geotérmica o no? Por eso la capa final tiene **una sola neurona con sigmoid**, no 2 neuronas con softmax. El resultado es un único número entre 0% y 100%:

| Salida del modelo | Temperatura de la salida lineal | Predicción |
|---|---|---|
| 0.03 | Número muy negativo (–3.5) | No geotérmico (3% probabilidad) |
| 0.50 | Cero (frontera de decisión) | Indeciso |
| 0.87 | Número positivo (+1.9) | Geotérmico (87% probabilidad) |

> Una sola neurona sigmoid es equivalente a dos neuronas softmax, pero más eficiente: usa la mitad de parámetros y es matemáticamente idéntica.

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

### 6.4 Ejemplo paso a paso: ¿qué pasa cuando das unas coordenadas?

Para hacer concreto todo lo anterior, veamos exactamente qué ocurre cuando alguien ingresa las coordenadas del **Nevado del Ruiz** (lat=4.895, lon=−75.322) en la app:

**① Descarga de la imagen (Google Earth Engine, ~3 s)**
```
Zona: buffer 5 km alrededor de (4.895°N, -75.322°W)
Descarga: imagen ASTER 7 bandas, ~111×111 píxeles
Guardado: /tmp/pred_4.895_-75.322.tif
```

**② Preprocesamiento (~50 ms)**
```
1. Leer el .tif con rasterio → array NumPy (111, 111, 7)
2. Reemplazar NoData (-9999) por mediana de cada banda
3. Redimensionar a (224, 224, 7) con scipy.zoom
4. Normalizar con estadísticas globales del dataset:
   banda_6 (temperatura): valor_real = 29.701
   banda_6 normalizada:   (valor - 29.701) / std_global
5. Expandir: (224, 224, 7) → (1, 224, 224, 7)  ← batch de 1 imagen
```

**③ Forward pass por el modelo (~200 ms en GPU)**
```
Entrada:   tensor (1, 224, 224, 7)
           │
           ▼ Channel Adapter
           tensor (1, 224, 224, 3)   ← 7 bandas → 3 canales
           │
           ▼ EfficientNetB0 (237 capas)
           tensor (1, 7, 7, 1280)    ← mapa de características
           │
           ▼ Global Average Pooling
           tensor (1, 1280)          ← 1.280 características
           │
           ▼ Dense(256) + BN + ReLU + Dropout
           tensor (1, 256)
           │
           ▼ Dense(64) + BN + ReLU + Dropout
           tensor (1, 64)
           │
           ▼ Dense(1, sigmoid)
           tensor (1, 1)             ← un solo número
```

**④ Resultado**
```
Salida del modelo: 0.9312  → 93.1% de probabilidad geotérmica
Decisión:          0.9312 > 0.50  → POTENCIAL GEOTÉRMICO ALTO ✅
```

Eso es todo. Los 4,4 millones de parámetros del modelo transformaron 351.232 números (la imagen) en un único número (la probabilidad) en menos de 300 milisegundos.

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

### 7.2 ¿Por qué 2 fases? (Explicación detallada)

El entrenamiento en 2 fases es una estrategia estándar en transfer learning. La lógica es la siguiente:

**Fase 1 — "Aprende lo nuevo sin tocar lo viejo" (30 épocas):**

El backbone (EfficientNetB0) llega con 4.050.249 parámetros ya entrenados en ImageNet — sabe detectar bordes, texturas, patrones visuales. Si desde el primer momento le permitimos modificar TODOS esos parámetros con nuestro dataset (que es 550× más pequeño que ImageNet), corre el riesgo de **destruir** ese conocimiento valioso. Es como si un cirujano experto intentara reaprender anatomía con un solo libro de texto: perdería más de lo que ganaría.

Por eso, en la Fase 1 congelamos el backbone completo (`layer.trainable = False` para cada capa) y solo entrenamos:
- El **Channel Adapter** (las "gafas" que traducen 7 → 3 canales) — ~345K parámetros
- El **Classification Head** (la "cabeza" que toma la decisión final) — incluido en esos ~345K

El learning rate es relativamente alto (0,001) porque estas capas se entrenan **desde cero** — no hay conocimiento previo que proteger. Al final de esta fase, el adapter ya sabe cómo combinar las 7 bandas ASTER, y la cabeza ya tiene un criterio razonable de clasificación (val_auc = 0,9000).

**Fase 2 — "Ahora refina tu visión para este dominio" (50 épocas):**

Una vez que el adapter y la cabeza ya están entrenados, podemos empezar a **ajustar el backbone** — pero con extrema cautela:

1. **Solo se descongelan las últimas 39 capas** (de 237 totales). ¿Por qué las últimas? Porque en las CNN las capas están organizadas jerárquicamente:
   - Capas iniciales (1–50): detectan bordes, líneas, contrastes → **universales**, no necesitan cambiar
   - Capas medias (50–150): detectan texturas, patrones repetitivos → **mayormente universales**
   - Capas finales (150–237): detectan combinaciones complejas de patrones → **específicas del dominio original (fotos naturales)**, necesitan adaptarse a imágenes satelitales

2. **El learning rate se reduce 10×** (de 0,001 a 0,0001). Esto significa que cada corrección es 10 veces más pequeña. Los pesos del backbone se ajustan con "pinzas de relojero", no con un martillo. El objetivo es **refinar**, no reemplazar.

3. **Las capas de Batch Normalization permanecen congeladas** (ver 7.3 abajo). Sus estadísticas internas son valiosas y no queremos que se recalculen con nuestro dataset pequeño.

4. **EarlyStopping es más paciente** (patience = 15 vs 10 en Fase 1). El fine-tuning produce mejoras más graduales, así que le damos más tiempo antes de declarar que ya no mejora.

El resultado: la validación sube de AUC 0,9000 → 0,9725. El backbone ajustó sus detectores de patrones al dominio ASTER sin perder sus habilidades fundamentales.

**Visualización del proceso completo:**

```
Fase 1 (épocas 1–30):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Adapter    │   │   Backbone   │   │     Head     │
│  🔓 ABIERTO  │──▶│  🔒 CERRADO  │──▶│  🔓 ABIERTO  │
│  LR = 0,001  │   │ (no cambia)  │   │  LR = 0,001  │
│  345K params │   │ 4,05M params │   │              │
└──────────────┘   └──────────────┘   └──────────────┘

Fase 2 (épocas 31–80):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Adapter    │   │   Backbone   │   │     Head     │
│  🔓 ABIERTO  │──▶│  🔓 últimas  │──▶│  🔓 ABIERTO  │
│ LR = 0,0001  │   │  39 capas    │   │ LR = 0,0001  │
│              │   │ BN congelado │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 7.3 ¿Por qué mantener BatchNorm congelado en Fase 2?

Las capas de **Batch Normalization (BN)** son un caso especial. Cada capa BN almacena internamente dos valores acumulados:

- **Media móvil (running mean):** el promedio de las activaciones que ha visto durante el entrenamiento
- **Varianza móvil (running variance):** la dispersión de esas activaciones

Estos valores fueron calculados durante el preentrenamiento con **1,2 millones de imágenes** de ImageNet. Son estadísticas muy estables y robustas.

Si descongelamos las capas BN en la Fase 2, estas estadísticas se **recalcularían** usando solo nuestras ~15.000 imágenes (600× menos datos). Esto causaría que:

1. Los valores de media/varianza fluctúen mucho entre batches (solo 32 imágenes por batch)
2. La normalización interna se vuelva inestable
3. La red "olvide" cómo escalar correctamente sus activaciones
4. El entrenamiento se desestabilice y las métricas caigan

La solución estándar: al descongelar capas del backbone, se mantiene **BN en modo inferencia** (no actualiza sus estadísticas). Así la red ajusta los filtros convolucionales mientras mantiene una normalización estable basada en las estadísticas robustas de ImageNet.

### 7.4 Generador de datos por particiones

Los 22.209 imágenes (~29,8 GB en .npy) no caben en RAM. El generador:
1. Pre-shuffle de datos en disco (seed=42, garantiza mezcla de clases)
2. Cada época: shuffle del **orden de las partes** (~500 imgs c/u)
3. Carga **una parte completa** a la vez (lectura secuencial, ~670 MB)
4. Shuffle de muestras **dentro** de cada parte
5. Yield batches de 32 imágenes

Esta estrategia mantiene excelente aleatoriedad con I/O óptimo.

### 7.5 Curvas de entrenamiento (¿Cómo evolucionó el aprendizaje?)

Las curvas de entrenamiento son el "diario de aprendizaje" del modelo: muestran época a época si está mejorando, si se estancó, o si está empezando a memorizar en vez de aprender.

#### Fase 1 — Backbone congelado (épocas 1–30)

En esta fase solo el Channel Adapter y la cabeza de clasificación aprenden desde cero. El backbone (EfficientNetB0) está congelado.

```
Época  │ val_auc  │ val_loss  │ Observación
───────┼──────────┼───────────┼──────────────────────────────────────────
  1    │  ~0.500  │  alto     │ El modelo empieza sin saber nada (azar puro)
  5    │  ~0.720  │  bajando  │ Aprende patrones básicos rápido
 10    │  ~0.820  │  bajando  │ La cabeza empieza a discriminar bien
 15    │  ~0.860  │  estable  │ Ralentiza: ya extrajo lo fácil del backbone
 20    │  ~0.880  │  estable  │ Mejoras más lentas, convergiendo
 27    │  0.9000  │  mínimo   │ ★ Mejor época de Fase 1 (checkpoint guardado)
 28    │  <0.900  │  sube     │ EarlyStopping detecta que no mejora
 29    │  <0.900  │  sube     │ Paciencia = 3, segundo aviso
 30    │  0.8997  │  sube     │ Fin Fase 1 (se agotaron las épocas)
```

**Interpretación:** La curva sube rápido al inicio (el adapter y la cabeza tienen mucho que aprender) y luego se aplana. El plateau después de la época 20 indica que la cabeza ya extrajo todo lo útil del backbone congelado. Es el momento justo de pasar a Fase 2 y descongelar el backbone para aprender representaciones más específicas.

#### Fase 2 — Fine-tuning (épocas 31–80, numeradas 1–50 internamente)

Las últimas 39 capas del backbone se descongelan. El learning rate baja de 1×10⁻³ a 1×10⁻⁴ (10 veces más cauteloso) para no destruir los pesos preentrenados.

```
Época  │ val_auc  │ val_loss  │ Observación
───────┼──────────┼───────────┼──────────────────────────────────────────
  31   │  0.920   │  bajando  │ ¡Salto inmediato! El backbone ya podía dar más
  35   │  0.940   │  bajando  │ Fine-tuning activo, el modelo re-aprende texturas
  40   │  0.960   │  bajando  │ Las capas profundas refinan bordes geotérmicos
  45   │  0.969   │  bajando  │ Convergencia más lenta, alta calidad
  50   │  0.9725  │  mínimo   │ ★ Mejor época de Fase 2 (modelo final guardado)
  51   │  <0.9725 │  sube     │ EarlyStopping empieza a contar
  ...  │  plana   │  sube     │ Sin mejoras
  57   │  <0.9725 │  sube     │ EarlyStopping activa (paciencia=7): detiene
```

**Interpretación:** El salto en la época 31 confirma que el backbone tenía capacidad latente que el congelamiento impedía usar. La curva en Fase 2 sube más lentamente porque los ajustes son sutiles: ya no se aprenden patrones nuevos desde cero, sino que los filtros existentes se *afinan* para responder mejor a las firmas espectrales geotérmicas.

#### Visualización de las dos fases

```
AUC en validación (simplificado)
│
0.97 │                                          ●  ← Mejor: ep50 (0.9725)
0.96 │                                     ●●●
0.95 │                                 ●●●
0.94 │                              ●●●
0.93 │             ┊             ●●●               ← Fine-tuning comenzó
0.92 │             ┊          ●●●
0.91 │             ┊       ●●●
0.90 │          ●  ┊ ─────●    ← Plateau Fase 1, mejor ep27 (0.9000)
0.88 │       ●●●●  ┊
0.85 │     ●●      ┊
0.82 │   ●●        ┊
0.75 │  ●          ┊
0.60 │ ●           ┊
0.50 │●            ┊
─────┴─────────────┴─────────────────────────────────►
     ep1          ep27  ep31                        ep50
     ◄── Fase 1: backbone congelado ──►◄── Fase 2: fine-tuning ──►
```

#### ¿Por qué importa monitorear estas curvas?

| Patrón en las curvas | Diagnóstico | Solución |
|---|---|---|
| `val_loss` baja y `val_auc` sube juntas | ✅ Aprendizaje sano | Continuar |
| `train_auc` sube pero `val_auc` se estanca | ⚠️ Inicio de overfitting | EarlyStopping, más regularización |
| Ambas curvas oscilan sin bajar | ⚠️ Learning rate muy alto | Reducir lr |
| Ambas curvas no bajan en absoluto | ⚠️ Learning rate muy bajo o modelo inadecuado | Revisar arquitectura |
| Salto repentino al cambiar de fase | ✅ El backbone tenía potencial no liberado | Normal en transfer learning |

En nuestro caso, el modelo exhibió el patrón más sano posible: mejora constante en Fase 1, salto positivo al inicio de Fase 2, y convergencia estable sin señales de overfitting.

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

### 8.4 ¿Qué significa AUC-ROC y cómo interpretarla?

El **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) es una de las métricas más importantes. Para entenderla, primero hay que ver qué es la curva ROC.

#### La situación del umbral

El modelo no dice "sí" o "no" directamente — produce una probabilidad (ej: 73%). Nosotros decidimos a qué umbral consideramos positivo. Por defecto es 50%, pero podría ser 30% (más sensible, más falsos positivos) o 70% (más conservador, más falsos negativos).

**La curva ROC** muestra qué pasa con las métricas para **cada umbral posible** de 0% a 100%:

```
       │ 1.0 ╔═══════════════╗
       │     ║   Modelo      ║
TPR    │     ║   perfecto    ║
(Re-   │ 0.97║               ●←── nuestro modelo
call)  │     ║           ●
       │     ║       ●
       │     ║   ●
       │ 0.0 ╚═════════════════
       └──────────────────────
         0.0  FPR (1-Especificidad) 1.0

  TPR = True Positive Rate (Recall) = TP/(TP+FN)
  FPR = False Positive Rate = FP/(FP+TN)
```

- **Punto (0, 0):** Umbral 100% — el modelo no predice nada positivo. Cero aciertos, cero falsas alarmas.
- **Punto (1, 1):** Umbral 0% — el modelo predice todo positivo. Detecta todo, pero también genera todas las falsas alarmas.
- **Curva ideal:** Sube verticalmente hasta (0, 1) — detecta todo sin ninguna falsa alarma.
- **Diagonal (línea punteada):** Lo que haría un modelo aleatorio (tirar una moneda).

#### El AUC: un solo número resumen

El **AUC** (Área Bajo la Curva) es exactamente eso: el área debajo de esa curva, entre 0 y 1:

| AUC | Interpretación |
|:-:|---|
| **1.00** | Modelo perfecto: separa siempre las clases sin errores |
| **0.97** | ← Nuestro modelo v3: excelente discriminación |
| **0.80–0.90** | Bueno |
| **0.70–0.80** | Aceptable |
| **0.50** | Equivale a tirar una moneda (modelo inútil) |
| **< 0.50** | Peor que el azar (¡el modelo está invertido!) |

Nuestro **AUC = 0,9737** significa que si tomamos una zona geotérmica real y una zona no geotérmica al azar, el modelo le asignará mayor probabilidad a la geotérmica en el **97,37% de los casos**. Solo en el 2,63% de los pares aleatorios se confundirá.

#### ¿Por qué es importante en nuestro contexto?

A diferencia de la accuracy (que solo es válida para un umbral fijo del 50%), el AUC es **independiente del umbral**. Mide la capacidad de discriminación del modelo en todas las situaciones posibles. En exploración geotérmica esto es valioso: si en el futuro se decide usar un umbral del 30% para ser más conservadores, el AUC ya nos dice que el modelo seguirá siendo excelente.

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
