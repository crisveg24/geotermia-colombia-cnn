# CONTENIDO PARA TESIS — Formato APA 7.ª edición
# Plantilla USB Colombia 2022 v.6

> **Instrucciones de uso:** Copiar y pegar cada sección en la plantilla de Word
> `Plantilla_APA_TesisUSBCo_2022_v.6.docx`, respetando los estilos de la plantilla.
> Los encabezados de este documento indican exactamente a qué sección de la plantilla corresponden.

---

## PORTADA

**Título:**
Modelo Predictivo Basado en Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia Mediante Imágenes Satelitales ASTER

**Autores:**
Daniel Santiago Arévalo Rubiano
Cristian Camilo Vega Sánchez
Yuliet Katerin Espitia Ayala
Laura Sophie Rivera Martín

**Asesor:**
Yeison Eduardo Conejo Sandoval

**Institución:**
Universidad de San Buenaventura — Bogotá
Facultad de Ingeniería
Programa de Ingeniería de Sistemas

**Año:** 2026

---

## DEDICATORIA

*(Cada autor redacta su dedicatoria personal. Espacio reservado.)*

---

## AGRADECIMIENTOS

*(Cada autor redacta sus agradecimientos personales. Espacio reservado.)*

---

## LISTA DE TABLAS

| N.º | Título | Pág. |
|-----|--------|------|
| 1 | Zonas geotérmicas conocidas en Colombia | — |
| 2 | Bandas ASTER utilizadas en el modelo | — |
| 3 | Técnicas de aumento de datos aplicadas | — |
| 4 | Hiperparámetros del entrenamiento v2 | — |
| 5 | Matriz de confusión del modelo v2 en el conjunto de prueba | — |
| 6 | Métricas de evaluación del modelo v2 en el conjunto de prueba | — |
| 7 | Comparativo de métricas v1 vs. v2 | — |
| 8 | Objetivos de métricas vs. resultados obtenidos (v2) | — |
| 9 | Análisis de factores de mejora de v1 a v2 | — |
| 10 | Tecnologías y herramientas utilizadas | — |
| 11 | Composición del dataset v3 por país (Región Andina) | — |
| 12 | División del conjunto de datos v3 con anti-leakage geográfico | — |
| 13 | Comparativo de versiones del dataset: v1, v2 y v3 | — |
| 14 | Hiperparámetros del entrenamiento v3 (dos fases) | — |
| 15 | Matriz de confusión del modelo v3 en el conjunto de prueba | — |
| 16 | Métricas de evaluación del modelo v3 en el conjunto de prueba | — |
| 17 | Comparativo de métricas v1 vs. v2 vs. v3 | — |
| 18 | Objetivos de métricas vs. resultados obtenidos (v3) | — |
| 19 | Bugs adicionales descubiertos y corregidos en v3 | — |
*(Actualizar números de página una vez insertado en Word.)*

---

## LISTA DE FIGURAS

| N.º | Título | Pág. |
|-----|--------|------|
| 1 | Mapa de zonas geotérmicas conocidas de Colombia | — |
| 2 | Arquitectura de la CNN v2 ResNet-inspired (diagrama de bloques) | — |
| 3 | Diagrama del pipeline de procesamiento de datos | — |
| 4 | Flujo del bloque residual con conexión de atajo (v2) | — |
| 5 | Curvas de entrenamiento v2: accuracy y loss por época | — |
| 6 | Curva ROC del modelo v2 (AUC = 0,983) | — |
| 7 | Matriz de confusión del modelo v2 | — |
| 8 | Comparativo visual de métricas v1 vs. v2 | — |
| 9 | Interfaz de la aplicación web Streamlit — sección de predicción | — |
| 10 | Mapa de calor de predicciones geotérmicas en Colombia | — |
| 11 | Mapa de zonas de entrenamiento de la Región Andina (4 países) | — |
| 12 | Distribución de imágenes por país en el dataset v3 | — |
| 13 | Arquitectura del modelo v3: EfficientNetB0 + Channel Adapter (diagrama) | — |
| 14 | Curvas de entrenamiento del modelo v3 (Fase 1 + Fase 2) | — |
| 15 | Curva ROC del modelo v3 (AUC = 0,9737) | — |
| 16 | Matriz de confusión del modelo v3 | — |
| 17 | Comparativo visual de métricas v1 vs. v2 vs. v3 | — |
*(Actualizar números de página una vez insertado en Word.)*

---

## RESUMEN

**Título:** Modelo Predictivo Basado en Redes Neuronales Convolucionales (CNN) para la Identificación de Zonas de Potencial Geotérmico en Colombia Mediante Imágenes Satelitales ASTER

La presente investigación desarrolló un modelo predictivo basado en Redes Neuronales Convolucionales (CNN) para la identificación automatizada de zonas con potencial geotérmico en Colombia, utilizando imágenes satelitales del sensor ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) de la NASA. El modelo emplea una arquitectura ResNet-inspired personalizada con 5.032.385 parámetros, entrenada con un conjunto de datos de 200 imágenes originales (111 positivas y 89 negativas) expandido a 6.200 imágenes mediante técnicas de aumento de datos. Las imágenes contemplan 7 bandas espectrales: 5 de emisividad térmica infrarroja (bandas 10–14), temperatura superficial y el índice de vegetación de diferencia normalizada (NDVI), extraídas del producto ASTER Global Emissivity Dataset (AG100 v003) a través de Google Earth Engine. Se implementaron correcciones críticas en el pipeline de procesamiento, incluyendo el filtrado de valores NoData (−9999), la eliminación de la doble normalización y la prevención de fuga de datos entre conjuntos mediante GroupShuffleSplit. El modelo v2 alcanzó una exactitud del 91,45 %, una precisión del 97,94 %, una sensibilidad del 86,05 %, un puntaje F1 del 91,61 %, un área bajo la curva ROC de 0,983 y un coeficiente de correlación de Matthews (MCC) de 0,837 en el conjunto de prueba (1.017 imágenes), superando ampliamente los resultados de la versión inicial (exactitud 68,43 %, sensibilidad 48,10 %). Adicionalmente, se desarrolló una interfaz web interactiva con Streamlit y Folium que permite realizar predicciones en tiempo real sobre cualquier coordenada del territorio colombiano. El sistema constituye una herramienta de apoyo para la fase de reconocimiento regional en la exploración geotérmica, permitiendo priorizar zonas de interés para estudios detallados.

Posteriormente, se construyó una versión ampliada del dataset (v3) que incorpora 2.019 imágenes base de la Región Andina (Colombia, Ecuador, Perú y Chile), expandidas a 22.209 imágenes mediante 10 técnicas de aumento, con una división libre de fuga de datos basada en agrupación geográfica (4.038 zonas únicas). Esta expansión se fundamenta en que los cuatro países comparten el contexto geológico del Cinturón de Fuego del Pacífico, lo que permite al modelo aprender patrones espectrales geotérmicos más generalizables. Las métricas del modelo v3 se validan mediante intervalos de confianza Bootstrap al 95 % (2.000 iteraciones).

**Palabras clave:** redes neuronales convolucionales, aprendizaje profundo, energía geotérmica, teledetección, ASTER, clasificación de imágenes satelitales, Colombia, Región Andina.

---

## ABSTRACT

**Title:** Predictive Model Based on Convolutional Neural Networks (CNN) for the Identification of Geothermal Potential Zones in Colombia Using ASTER Satellite Imagery

This research developed a predictive model based on Convolutional Neural Networks (CNN) for the automated identification of geothermal potential zones in Colombia, using satellite imagery from NASA's ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) sensor. The images comprise 7 spectral bands: 5 thermal infrared emissivity bands (bands 10–14), surface temperature, and the Normalized Difference Vegetation Index (NDVI), extracted from the ASTER Global Emissivity Dataset (AG100 v003) via Google Earth Engine.

The project evolved through three versions. Version 1 (v1), with 85 original images and a custom CNN architecture, served as a baseline with 68.43% accuracy. After an exhaustive audit that identified and corrected 28 pipeline errors—including NoData filtering (−9999), double normalization removal, and data leakage prevention via GroupShuffleSplit—version 2 (v2) with a ResNet-inspired architecture (5,032,385 parameters) and 200 original images from Colombia (6,200 augmented) achieved 91.45% accuracy, ROC AUC of 0.983, and F1-score of 91.61% on a test set of 1,017 images.

For version 3 (v3), the dataset was expanded to the Andean Region (Colombia, Ecuador, Peru, and Chile), obtaining 2,019 base images (997 positive, 1,022 negative) expanded to 22,209 images using 10 augmentation techniques. During v3 preparation, a critical new bug was discovered: 62.5% data leakage caused by augmentations of the same geographic zone being distributed across training and evaluation sets. This was corrected through a complete re-split using GroupShuffleSplit grouped by base geographic zone (407 independent zones, zero verified leakage). The v3 model employs a Transfer Learning architecture with EfficientNetB0 pretrained on ImageNet as a feature extractor, preceded by a convolutional Channel Adapter (7 → 16 → 3 channels) that learns the optimal projection of ASTER's 7 bands into RGB space. With 4,396,112 parameters and a two-phase training scheme (Phase 1: frozen backbone, 30 epochs; Phase 2: fine-tuning of last 39 layers, 50 epochs) executed on an NVIDIA RTX 4070 GPU with mixed-precision float16 via WSL2, the v3 model achieved 92.28% accuracy, 91.27% precision, 93.17% recall, 92.21% F1-score, ROC AUC of 0.9737, and MCC of 0.8458 on a test set of 3,619 images, confirming the model's ability to reliably identify geothermal potential zones.

Additionally, an interactive web interface was developed using Streamlit and Folium, enabling real-time predictions for any coordinate within Colombian territory. The system serves as a support tool for the regional reconnaissance phase in geothermal exploration, allowing prioritization of areas of interest for detailed studies.

**Keywords:** convolutional neural networks, deep learning, Transfer Learning, geothermal energy, remote sensing, ASTER, EfficientNet, satellite image classification, Colombia, Andean Region.

---

## INTRODUCCIÓN

Colombia posee un alto potencial geotérmico debido a su ubicación privilegiada en el Cinturón de Fuego del Pacífico, donde convergen las placas tectónicas de Nazca, Sudamericana y del Caribe. El Servicio Geológico Colombiano (SGC) estima que el país cuenta con un potencial geotérmico superior a los 2.000 MW; sin embargo, a la fecha no existe ninguna planta geotérmica operativa en el territorio nacional (Servicio Geológico Colombiano, 2023). La identificación de nuevas zonas con potencial geotérmico constituye un paso fundamental para diversificar la matriz energética del país y avanzar hacia fuentes de energía renovable y de baja emisión de carbono.

La exploración geotérmica convencional es un proceso costoso y prolongado que comprende múltiples fases: reconocimiento regional, exploración de superficie, exploración profunda y desarrollo. La primera fase —reconocimiento regional— incluye la revisión bibliográfica, el mapeo geológico superficial y el análisis de imágenes satelitales. Es precisamente en esta fase inicial donde las técnicas de teledetección y el aprendizaje automático pueden aportar un valor significativo, al permitir el análisis sistemático de grandes extensiones de territorio en tiempos reducidos.

El sensor ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer), a bordo del satélite Terra de la NASA, proporciona datos multiespectrales en 14 bandas que cubren las regiones del visible e infrarrojo cercano (VNIR), infrarrojo de onda corta (SWIR) e infrarrojo térmico (TIR). Las bandas del infrarrojo térmico son particularmente relevantes para la exploración geotérmica, ya que permiten detectar anomalías térmicas superficiales, alteraciones hidrotermales de minerales y composiciones mineralógicas indicadoras de actividad geotérmica profunda (Coolbaugh et al., 2007; Vaughan et al., 2005).

Las Redes Neuronales Convolucionales (CNN), una clase de modelos de aprendizaje profundo especializados en el procesamiento de datos con estructura espacial, han demostrado un desempeño sobresaliente en tareas de clasificación de imágenes satelitales y teledetección (LeCun et al., 2015; Zhu et al., 2017). A diferencia de los métodos tradicionales basados en umbrales fijos o clasificación manual, las CNN aprenden automáticamente patrones complejos y jerarquías de características directamente desde los datos.

La presente investigación se enfoca en el desarrollo, entrenamiento y evaluación de un modelo predictivo basado en CNN para la clasificación binaria de zonas con y sin potencial geotérmico en Colombia, utilizando imágenes del producto ASTER Global Emissivity Dataset (AG100 v003) obtenidas a través de Google Earth Engine. El modelo final (v3) emplea Transfer Learning con EfficientNetB0 preentrenado en ImageNet, precedido por un Channel Adapter convolucional que proyecta las 7 bandas ASTER al espacio RGB de 3 canales. Para robustecer el entrenamiento, el conjunto de datos se expandió a la Región Andina (Colombia, Ecuador, Perú y Chile), aprovechando que estos países comparten el contexto geológico del Cinturón de Fuego del Pacífico (Lahsen, 1982; Bona y Coviello, 2016). Esta decisión permite al modelo capturar una mayor diversidad de patrones espectrales asociados a actividad geotérmica, mientras que la aplicación y las conclusiones se circunscriben al territorio colombiano. El proyecto incluye la construcción de un pipeline completo —desde la adquisición de datos satelitales hasta la predicción en tiempo real— y el desarrollo de una interfaz web interactiva que facilite el uso del modelo por parte de investigadores y tomadores de decisiones.

El documento se estructura siguiendo las normas APA 7.ª edición e incluye el planteamiento del problema, la justificación, los objetivos, el marco teórico, la metodología detallada, los resultados cuantitativos obtenidos, la discusión de hallazgos, las conclusiones y las recomendaciones para trabajos futuros.

---

## 1. PLANTEAMIENTO DEL PROBLEMA

La transición energética global demanda la identificación y aprovechamiento de fuentes de energía renovable que permitan reducir la dependencia de combustibles fósiles y mitigar los efectos del cambio climático. En este contexto, la energía geotérmica se presenta como una alternativa limpia, confiable y con disponibilidad continua (base load), a diferencia de otras fuentes renovables que dependen de condiciones atmosféricas (DiPippo, 2015).

Colombia, pese a su ubicación geológicamente privilegiada en una de las zonas de mayor actividad tectónica y volcánica del planeta, no ha logrado materializar el aprovechamiento de su potencial geotérmico estimado en más de 2.000 MW (Servicio Geológico Colombiano, 2023). Esta situación contrasta con países como Islandia, Nueva Zelanda, Kenia y El Salvador, que cuentan con plantas geotérmicas operativas desarrolladas a partir de condiciones geológicas comparables.

Una de las barreras principales para el desarrollo geotérmico en Colombia es el alto costo y el prolongado tiempo requerido para las fases de exploración. La fase inicial de reconocimiento regional, que incluye el análisis de imágenes satelitales por parte de expertos, se realiza de manera manual y está limitada por el presupuesto y el tiempo disponible, lo que restringe la cobertura territorial. Esta limitación deja potencialmente sin evaluar vastas extensiones del territorio colombiano que podrían albergar recursos geotérmicos.

### 1.1 Antecedentes

#### 1.1.1 Exploración geotérmica en Colombia

La exploración geotérmica en Colombia tiene antecedentes que se remontan a estudios realizados por entidades como ISAGEN y el Servicio Geológico Colombiano. El proyecto más avanzado es el del Nevado del Ruiz en los departamentos de Caldas y Tolima, donde ISAGEN realizó estudios de factibilidad que identificaron temperaturas superiores a 200 °C (Alfaro, 2015). Otras zonas evaluadas incluyen Chiles–Cerro Negro y Azufral en Nariño, Paipa–Iza en Boyacá, y Coconucos en Cauca.

No obstante, la cobertura de estos estudios se ha concentrado en zonas con manifestaciones superficiales evidentes (volcanes activos, fuentes termales), dejando sin evaluar regiones donde la actividad geotérmica podría no manifestarse en superficie de manera evidente. Los sistemas geotérmicos mejorados (EGS, por sus siglas en inglés) demuestran que la ausencia de manifestaciones superficiales no descarta la presencia de recursos geotérmicos en profundidad (Tester et al., 2006).

#### 1.1.2 Teledetección aplicada a la exploración geotérmica

El uso de sensores remotos para la identificación de recursos geotérmicos ha sido documentado extensamente en la literatura científica. Coolbaugh et al. (2007) demostraron la efectividad del sensor ASTER para detectar anomalías térmicas asociadas a actividad geotérmica en Nevada, Estados Unidos. Vaughan et al. (2005) utilizaron imágenes multiespectrales del infrarrojo térmico para mapear minerales de alteración hidrotermal en Steamboat Springs.

El producto ASTER Global Emissivity Dataset (AG100 v003) proporciona datos de emisividad y temperatura superficial con resolución de 100 metros y cobertura global, lo que lo convierte en una fuente de datos idónea para estudios de reconocimiento a escala regional (Hulley et al., 2015).

#### 1.1.3 Aprendizaje profundo en teledetección

La aplicación de técnicas de aprendizaje profundo a la clasificación de imágenes satelitales ha experimentado un crecimiento exponencial en la última década. Zhu et al. (2017) presentaron una revisión exhaustiva de las aplicaciones de Deep Learning en teledetección, destacando las CNN como la arquitectura dominante para tareas de clasificación y segmentación. He et al. (2016) introdujeron las redes residuales (ResNet), cuyas conexiones de atajo permiten entrenar redes más profundas sin degradación del rendimiento, alcanzando resultados sobresalientes en competiciones de clasificación de imágenes como ImageNet.

En el contexto específico de la exploración geotérmica, estudios como los de Mia et al. (2018) han aplicado técnicas de Machine Learning a datos ASTER para la identificación de alteraciones hidrotermales, aunque el uso de CNN profundas con arquitectura residual para esta tarea particular es aún incipiente y constituye un área de oportunidad investigativa.

#### 1.1.4 Brecha identificada

Existe una brecha entre la disponibilidad de datos satelitales globales de alta calidad (como ASTER GED) y su aprovechamiento sistemático para la identificación de potencial geotérmico en Colombia. Los métodos tradicionales de análisis, basados en la interpretación manual por expertos, no escalan al territorio nacional completo. El uso de CNN para automatizar este proceso de screening representa una oportunidad para ampliar significativamente la cobertura de la evaluación geotérmica con costos y tiempos reducidos.

---

## 2. JUSTIFICACIÓN

La investigación se justifica desde múltiples perspectivas:

**Perspectiva energética.** Colombia necesita diversificar su matriz energética, actualmente dependiente de la hidroeléctrica (aproximadamente el 70 %) y vulnerable a fenómenos climáticos como El Niño. La energía geotérmica ofrece generación continua independiente de condiciones atmosféricas, con un factor de capacidad superior al 90 % (DiPippo, 2015).

**Perspectiva ambiental.** Las plantas geotérmicas producen emisiones de CO₂ significativamente menores que las de combustibles fósiles —entre 15 y 55 g CO₂/kWh frente a 400–1.000 g CO₂/kWh del gas natural y el carbón— (IPCC, 2011). Su desarrollo contribuiría al cumplimiento de los compromisos de Colombia en el Acuerdo de París.

**Perspectiva económica.** La fase de reconocimiento regional mediante métodos convencionales tiene costos estimados entre 50.000 y 200.000 USD y puede extenderse por meses (Gehringer y Loksha, 2012). Un modelo automatizado de screening reduce drásticamente estos costos y tiempos, permitiendo la evaluación de extensiones territoriales que de otro modo quedarían sin analizar.

**Perspectiva tecnológica.** La convergencia de datos satelitales abiertos (ASTER GED vía Google Earth Engine), infraestructura computacional accesible (TensorFlow/Keras) y avances en aprendizaje profundo permite abordar problemas que antes requerían recursos computacionales y humanos prohibitivos.

**Perspectiva académica.** El proyecto contribuye al campo interdisciplinario que integra la ingeniería de sistemas con las ciencias de la Tierra, demostrando la aplicabilidad de las CNN a problemas de clasificación de datos geoespaciales en un contexto colombiano.

### 2.1 Alcance

El alcance del presente proyecto comprende:

1. La construcción de un conjunto de datos de imágenes ASTER de la Región Andina (Colombia, Ecuador, Perú y Chile) con 2.019 imágenes base (997 positivas y 1.022 negativas) y 7 bandas espectrales, aprovechando el contexto geológico compartido del Cinturón de Fuego del Pacífico para robustecer el entrenamiento.
2. El diseño, implementación y entrenamiento de un modelo CNN basado en Transfer Learning con EfficientNetB0 y un Channel Adapter convolucional (7 → 16 → 3 canales), optimizado para la clasificación binaria de imágenes satelitales multiespectrales de 224 × 224 × 7 píxeles.
3. La evaluación cuantitativa del modelo utilizando métricas estándar (exactitud, precisión, sensibilidad, F1, ROC AUC, MCC).
4. El desarrollo de una interfaz web que permita realizar predicciones interactivas sobre cualquier coordenada del territorio colombiano.
5. La documentación completa del proceso, resultados y lecciones aprendidas.

**No forma parte del alcance:**
- La validación en campo de las predicciones del modelo.
- La estimación de temperaturas de reservorio o profundidades.
- La evaluación de viabilidad económica de proyectos geotérmicos.
- La generalización del modelo a países distintos de Colombia (la expansión andina es exclusivamente para robustez del entrenamiento).

---

## 3. OBJETIVOS

### 3.1 Objetivo general

Desarrollar un modelo predictivo basado en Redes Neuronales Convolucionales (CNN) que permita identificar zonas con potencial geotérmico en Colombia a partir de imágenes satelitales del sensor ASTER, logrando métricas de evaluación superiores a los umbrales definidos (exactitud > 85 %, sensibilidad > 80 %, F1 > 80 %, ROC AUC > 0,90).

### 3.2 Objetivos específicos

1. Construir un conjunto de datos etiquetado de imágenes ASTER de zonas con y sin potencial geotérmico de la Región Andina (Colombia, Ecuador, Perú y Chile), utilizando el producto ASTER Global Emissivity Dataset (AG100 v003) de Google Earth Engine, con un mínimo de 2.000 imágenes originales y 7 bandas espectrales, implementando descarga paralela y expansión por grilla para maximizar la cobertura espacial.

2. Diseñar e implementar una arquitectura de Red Neuronal Convolucional basada en Transfer Learning con EfficientNetB0 preentrenado en ImageNet, precedida por un Channel Adapter convolucional que proyecta las 7 bandas ASTER al espacio RGB de 3 canales, optimizada para la clasificación binaria de imágenes satelitales multiespectrales de 224 × 224 × 7 píxeles.

3. Entrenar y optimizar el modelo CNN mediante un esquema de dos fases (backbone congelado + fine-tuning) aplicando técnicas de regularización (Dropout, MixUp, AdamW con weight decay, label smoothing), aumento de datos online y offline, y prevención de fuga de datos (GroupShuffleSplit con agrupación geográfica por zona base), evaluando su desempeño mediante las métricas de exactitud, precisión, sensibilidad, F1-Score, ROC AUC y coeficiente de correlación de Matthews (MCC).

4. Desarrollar una interfaz web interactiva con Streamlit y Folium que permita realizar predicciones de potencial geotérmico en tiempo real sobre cualquier coordenada del territorio colombiano, con visualización de resultados en mapas interactivos.

---

## 4. PROBLEMA DE INVESTIGACIÓN

¿Es posible desarrollar un modelo predictivo basado en Redes Neuronales Convolucionales (CNN) que identifique, con una exactitud superior al 85 % y una sensibilidad superior al 80 %, zonas con potencial geotérmico en Colombia a partir del análisis automatizado de imágenes satelitales multiespectrales del sensor ASTER?

---

## 5. HIPÓTESIS

### 5.1 Hipótesis de trabajo

Un modelo predictivo basado en Redes Neuronales Convolucionales (CNN) con arquitectura de Transfer Learning (EfficientNetB0 preentrenado en ImageNet + Channel Adapter), entrenado con imágenes satelitales ASTER de 7 bandas espectrales (emisividad térmica infrarroja, temperatura superficial y NDVI), es capaz de identificar zonas con potencial geotérmico en Colombia con una exactitud superior al 85 % y una sensibilidad superior al 80 %.

### 5.2 Hipótesis estadística

#### 5.2.1 Hipótesis nula (H₀)

El modelo CNN no logra una exactitud significativamente superior al 50 % (nivel del azar) en la clasificación binaria de zonas con y sin potencial geotérmico. Formalmente:

$$H_0: \text{Accuracy}_{\text{modelo}} \leq 0{,}50$$

#### 5.2.1.1 Hipótesis alterna (H₁)

El modelo CNN logra una exactitud significativamente superior al 50 % en la clasificación binaria de zonas con y sin potencial geotérmico. Formalmente:

$$H_1: \text{Accuracy}_{\text{modelo}} > 0{,}50$$

#### 5.2.1.1.1 Variables

**Variable independiente:** Imágenes satelitales ASTER de 7 bandas (5 de emisividad térmica infrarroja: bandas 10–14; temperatura superficial; NDVI) de zonas del territorio colombiano.

**Variable dependiente:** Clasificación binaria del potencial geotérmico (1 = con potencial, 0 = sin potencial), expresada como una probabilidad continua en el rango [0, 1] que se dicotomiza con un umbral de 0,5.

**Variables de control:**
- Tamaño de imagen: 224 × 224 píxeles (estandarizado por redimensionamiento bicúbico).
- Fuente de datos: ASTER GED AG100 v003 (producto global estable).
- Normalización: Z-score por banda.
- División del conjunto de datos: GroupShuffleSplit con agrupación por zona geográfica base, eliminando sufijos de aumento y grilla (70 % entrenamiento, 15 % validación, 15 % prueba).

---

## 6. MARCO TEÓRICO

### 6.1 Energía geotérmica

La energía geotérmica es el calor almacenado en el interior de la Tierra, proveniente del calor primordial remanente de la formación del planeta y del decaimiento radiactivo de isótopos como uranio-238, torio-232 y potasio-40 en la corteza y el manto. Este calor se manifiesta en un gradiente geotérmico que, en promedio, aumenta entre 25 y 30 °C por cada kilómetro de profundidad, aunque en zonas volcánicas o tectónicamente activas puede superar los 100 °C/km (DiPippo, 2015).

Para que un recurso geotérmico sea explotable se requieren tres componentes fundamentales —denominados el «triángulo geotérmico»—: una fuente de calor (intrusión magmática o gradiente elevado), un reservorio (roca con suficiente porosidad y permeabilidad) y un fluido (agua líquida o en fase vapor) que actúe como medio de transporte del calor.

Existen tres tipos principales de sistemas geotérmicos:

- **Sistemas hidrotermales convencionales:** Requieren la presencia de agua subterránea que se calienta naturalmente al entrar en contacto con rocas calientes. Se manifiestan en superficie como aguas termales, fumarolas y géiseres. Ejemplos: The Geysers (California), zona Paipa–Iza (Colombia). Temperaturas típicas: 150–350 °C a profundidades de 1–3 km.

- **Sistemas geotérmicos mejorados (EGS):** Existen roca caliente pero no suficiente agua ni permeabilidad natural. Se inyecta agua a presión para crear fracturas artificiales por las que circula y se calienta el fluido. No requieren aguas termales ni acuíferos previos. Ejemplo: proyecto FORGE (Utah, EE. UU.). Temperaturas típicas: 150–300 °C a 3–6 km de profundidad.

- **Uso directo y bombas de calor geotérmicas (GSHP):** Aprovechan la temperatura estable del subsuelo (~15 °C a pocos metros) para calefacción, invernaderos y acuicultura. Funcionan en prácticamente cualquier lugar, sin necesidad de vulcanismo (Lund y Toth, 2021).

### 6.2 Contexto geotérmico de Colombia

Colombia tiene un alto potencial geotérmico debido a su ubicación en el Cinturón de Fuego del Pacífico. Las zonas geotérmicas conocidas incluyen el Nevado del Ruiz – Macizo Volcánico (Caldas/Tolima, >200 °C), Chiles–Cerro Negro y Azufral (Nariño, >200 °C), Paipa–Iza (Boyacá, 150–200 °C), Coconucos (Cauca, 150–200 °C) y Santa Rosa de Cabal (Risaralda, ~150 °C), entre otras.

Entidades como ISAGEN (estudios de factibilidad en el Nevado del Ruiz), el Servicio Geológico Colombiano (vigilancia volcánica y estudios geotérmicos) y la UPME (plan de diversificación energética) han participado en esfuerzos de exploración. No obstante, Colombia aún no cuenta con una planta geotérmica operativa, pese a un potencial estimado superior a 2.000 MW (Servicio Geológico Colombiano, 2023).

#### 6.2.1 Contexto geológico compartido de la Región Andina

Los países andinos —Colombia, Ecuador, Perú y Chile— comparten el contexto geológico del Cinturón de Fuego del Pacífico, una zona de subducción de la placa de Nazca bajo la placa Sudamericana que genera intensa actividad volcánica y geotérmica a lo largo de la cordillera de los Andes (Lahsen, 1982; Siebert et al., 2010). Chile cuenta con campos geotérmicos operativos como Cerro Pabellón (48 MW) y manifestaciones emblemáticas como El Tatio (Muñoz-Sáez et al., 2018). Ecuador explora activamente campos como Chachimbiro y Tufiño-Chiles —este último binacional con Colombia— (Rueda, 2015). Perú posee un inventario de más de 500 fuentes termales documentadas por el INGEMMET (2014), con zonas de alta entalpía asociadas a los volcanes Misti, Ubinas y la región de Tacna.

Esta similitud geológica justifica la inclusión de zonas de estos países en el conjunto de entrenamiento, permitiendo que el modelo aprenda patrones espectrales geotérmicos más universales dentro del contexto andino, al tiempo que la aplicación del modelo se circunscribe al territorio colombiano. Bona y Coviello (2016) destacan que la Región Andina constituye una de las áreas con mayor potencial geotérmico no explotado a nivel mundial.

### 6.3 Teledetección satelital con ASTER

El sensor ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer), a bordo del satélite Terra de la NASA, dispone de 14 bandas espectrales que cubren las regiones VNIR (3 bandas), SWIR (6 bandas) y TIR (5 bandas). Las bandas TIR son particularmente adecuadas para la detección de anomalías térmicas superficiales y la caracterización de la composición mineralógica de silicatos, mientras que las bandas SWIR permiten identificar minerales de alteración hidrotermal como alunita, caolinita y montmorillonita (Abrams y Hook, 2002).

El producto ASTER Global Emissivity Dataset (AG100 v003) proporciona valores de emisividad para las 5 bandas TIR y temperatura superficial con resolución de 100 metros y cobertura global, distribuido gratuitamente a través de Google Earth Engine (Hulley et al., 2015). En este proyecto se utilizan adicionalmente la temperatura superficial y el NDVI como bandas complementarias, totalizando 7 bandas por imagen.

Los indicadores de potencial geotérmico detectables por satélite incluyen anomalías térmicas superficiales (bandas TIR 10–14), alteración hidrotermal de minerales (SWIR y TIR), composición mineralógica (arcillas, sílice, óxidos) y presencia de vulcanismo reciente (Coolbaugh et al., 2007).

### 6.4 Redes Neuronales Convolucionales (CNN)

Las CNN son arquitecturas de aprendizaje profundo especializadas en el procesamiento de datos con estructura de cuadrícula, como las imágenes. Su poder radica en tres operaciones fundamentales (LeCun et al., 2015; Goodfellow et al., 2016):

- **Convolución (Conv2D):** Aplica filtros (kernels) aprendibles que detectan patrones locales como bordes, texturas y formas. La operación se define como:

$$\text{Output}(i,j) = \sum_{m,n} \text{Input}(i+m, j+n) \times \text{Kernel}(m,n) + b$$

- **Pooling (MaxPooling2D):** Reduce las dimensiones espaciales conservando las características más relevantes, lo que disminuye el costo computacional y aporta invariancia a traslaciones menores.

- **Activación (ReLU):** Introduce no linealidad mediante $f(x) = \max(0, x)$, permitiendo al modelo aprender relaciones complejas entre las entradas y las salidas.

#### 6.4.1 Redes residuales (ResNet)

He et al. (2016) introdujeron las conexiones residuales (skip connections), que permiten que el gradiente fluya directamente a través de las capas y evitan el problema de degradación en redes profundas. En un bloque residual, la salida se calcula como:

$$y = F(x, \{W_i\}) + x$$

donde $F$ representa las transformaciones del camino principal y $x$ es la entrada directa transmitida por el atajo. Esto permite que la red aprenda la función residual $F(x) = y - x$ en lugar de la transformación completa, lo cual es más fácil de optimizar. Esta arquitectura se empleó en la versión 2 del modelo (v2).

#### 6.4.2 Transfer Learning y EfficientNet

El Transfer Learning consiste en reutilizar los pesos de un modelo preentrenado en un dominio fuente (típicamente ImageNet, con 1,2 millones de imágenes naturales) y adaptarlos a un dominio objetivo con menos datos (Yosinski et al., 2014). Las primeras capas de una CNN aprenden características genéricas (bordes, texturas, gradientes) que son transferibles entre dominios, mientras que las capas superiores se especializan progresivamente.

EfficientNetB0 (Tan y Le, 2019) es una arquitectura que optimiza simultáneamente la profundidad, el ancho y la resolución de la red mediante un coeficiente de escalado compuesto. Con solo 4,0 millones de parámetros base, alcanza un rendimiento comparable a redes mucho más grandes. Su bloque fundamental es el MBConv (Mobile Inverted Bottleneck), que combina convoluciones depthwise separable con Squeeze-and-Excitation (SE) para calibrar adaptativamente la importancia de cada canal.

En este proyecto, EfficientNetB0 se utiliza como extractor de características espaciales en la versión 3 del modelo (v3), precedido por un Channel Adapter convolucional que proyecta las 7 bandas ASTER al espacio de 3 canales esperado por la red preentrenada. La estrategia de entrenamiento se divide en dos fases: (1) backbone congelado para entrenar el adapter y el clasificador, y (2) fine-tuning de las capas superiores del backbone con un learning rate reducido.

#### 6.4.3 Técnicas de regularización

- **Dropout y SpatialDropout2D:** Desactivan aleatoriamente neuronas o mapas de características completos durante el entrenamiento, previniendo la coadaptación y mejorando la generalización (Srivastava et al., 2014). SpatialDropout2D es especialmente efectivo para datos con correlación espacial como imágenes satelitales.

- **Batch Normalization:** Normaliza las activaciones de cada capa durante el entrenamiento, estabilizando y acelerando la convergencia (Ioffe y Szegedy, 2015).

- **Label Smoothing:** Suaviza las etiquetas duras (0/1) reemplazándolas por $(ε/K, 1 - ε + ε/K)$ con $ε = 0{,}1$, reduciendo la sobreconfianza del modelo (Szegedy et al., 2016).

- **Weight Decay (L2 desacoplado):** AdamW aplica la regularización L2 directamente sobre los pesos en lugar de incorporarla al gradiente, lo que produce una regularización más consistente (Loshchilov y Hutter, 2019).

- **MixUp (Zhang et al., 2018):** Técnica de aumento de datos que genera ejemplos virtuales de entrenamiento mediante interpolación lineal entre pares de muestras y sus etiquetas:
$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$
donde $\lambda \sim \text{Beta}(\alpha, \alpha)$ con $\alpha = 0.2$. MixUp suaviza la frontera de decisión, reduce la memorización y mejora la calibración del modelo. En la versión 3 se aplica MixUp sobre batches completos durante ambas fases de entrenamiento.

### 6.5 Métricas de evaluación para clasificación binaria

Para evaluar el desempeño del modelo se emplean las siguientes métricas:

- **Exactitud (Accuracy):** Proporción de predicciones correctas sobre el total de muestras:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precisión (Precision):** Proporción de predicciones positivas que son verdaderos positivos:
$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Sensibilidad (Recall):** Proporción de positivos reales correctamente identificados:
$$\text{Recall} = \frac{TP}{TP + FN}$$

- **Puntaje F1 (F1-Score):** Media armónica de precisión y sensibilidad:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Área bajo la curva ROC (ROC AUC):** Mide la capacidad discriminativa del modelo en todos los umbrales de clasificación. Un valor de 1,0 indica discriminación perfecta y 0,5 equivale al azar.

- **Coeficiente de correlación de Matthews (MCC):** Métrica que considera las cuatro categorías de la matriz de confusión y produce un valor equilibrado incluso con clases desbalanceadas:
$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 6.6 Intervalos de confianza Bootstrap

Los intervalos de confianza Bootstrap permiten estimar la variabilidad de las métricas de evaluación sin supuestos paramétricos sobre la distribución de los datos. El método consiste en generar $B$ muestras con reemplazo del conjunto de prueba, calcular la métrica de interés en cada una y derivar el intervalo percentil al nivel de confianza deseado (Efron y Tibshirani, 1993).

Formalmente, dado un conjunto de prueba $\mathcal{D}$ de $n$ observaciones, para cada iteración $b = 1, \ldots, B$ se obtiene una muestra $\mathcal{D}_b^*$ de tamaño $n$ con reemplazo, y se calcula $\hat{\theta}_b^* = T(\mathcal{D}_b^*)$ donde $T$ es el estadístico de interés (e.g., accuracy, F1). El intervalo de confianza al $100(1-\alpha)\%$ se define como:

$$\text{IC}_{1-\alpha} = \left[\hat{\theta}^*_{(\alpha/2)}, \hat{\theta}^*_{(1-\alpha/2)}\right]$$

En este proyecto se utilizan $B = 2.000$ iteraciones para obtener intervalos al 95\% sobre las 6 métricas principales (Accuracy, Precision, Recall, F1, ROC AUC, MCC), lo cual proporciona una estimación robusta de la incertidumbre del modelo.

---

## 7. METODOLOGÍA

La metodología del proyecto sigue un enfoque experimental cuantitativo, organizado en un pipeline de procesamiento de datos y entrenamiento de modelos de aprendizaje profundo. A continuación se describe cada etapa.

### 7.1 Adquisición de datos

Las imágenes satelitales se obtuvieron del producto **ASTER Global Emissivity Dataset (AG100 v003)** (NASA/ASTER_GED/AG100_003) a través de la API de Google Earth Engine (Gorelick et al., 2017). Para cada zona de interés se definieron las coordenadas centrales y se extrajo una imagen con un buffer de 5.000 metros y una resolución espacial de 90 metros por píxel, generando imágenes de aproximadamente 111 × 111 píxeles con 7 bandas.

Las 7 bandas extraídas son:

| Banda | Denominación | Longitud de onda / Descripción | Utilidad geotérmica |
|-------|-------------|-------------------------------|---------------------|
| 1 | emissivity_band10 | 8,125–8,475 μm | Detección de cuarzo caliente |
| 2 | emissivity_band11 | 8,475–8,825 μm | Identificación de feldespatos |
| 3 | emissivity_band12 | 8,925–9,275 μm | Detección de minerales arcillosos |
| 4 | emissivity_band13 | 10,25–10,95 μm | Temperatura superficial |
| 5 | emissivity_band14 | 10,95–11,65 μm | Anomalías térmicas |
| 6 | temperature | Temperatura superficial (°C × 100) | Indicador directo de calor |
| 7 | ndvi | Índice de vegetación normalizado | Proxy de cobertura vegetal |

**Versión 2 (Colombia):** Se recopilaron un total de **200 imágenes originales**: 111 correspondientes a zonas con potencial geotérmico conocido o con indicadores (zonas volcánicas activas, manifestaciones hidrotermales, campos geotérmicos estudiados) y 89 correspondientes a zonas de control sin potencial geotérmico (Llanos Orientales, Amazonia, Costa Caribe, Altiplano Cundiboyacense, Chocó). El etiquetado se realizó a partir de la literatura geológica existente y de los mapas del Servicio Geológico Colombiano.

#### 7.1.1 Expansión a la Región Andina (v3)

Para incrementar la robustez del modelo se amplió el conjunto de datos a la **Región Andina** (Colombia, Ecuador, Perú y Chile), aprovechando el contexto geológico compartido del Cinturón de Fuego del Pacífico (véase §\u00a06.2.1). Se definieron 245 zonas base distribuidas así:

| País | Zonas positivas | Zonas negativas | Total |
|------|:-:|:-:|:-:|
| Colombia | 111 | 89 | 200 |
| Ecuador | 19 | 8 | 27 |
| Perú | 12 | 3 | 15 |
| Chile | 3 | 0 | 3 |
| **Total base** | **145** | **100** | **245** |

Para cada zona base se aplicó una **expansión por grilla** (`_generate_grid()`), generando hasta 9 sub-zonas adyacentes (3 × 3) desplazadas ± 0,045° en latitud y longitud. Tras descartar zonas con datos insuficientes o duplicados, se obtuvieron **2.019 imágenes base** (997 positivas y 1.022 negativas), que totalizaron 115,1 MB en formato GeoTIFF.

La descarga se implementó con **paralelismo de 3 hilos** y un retardo de 0,5 s entre solicitudes para respetar las cuotas de la API de Google Earth Engine. El proceso completo tardó aproximadamente 2 horas.

### 7.2 Preprocesamiento y filtrado de datos

El preprocesamiento incluyó los siguientes pasos:

1. **Filtrado de valores NoData:** Los datos ASTER utilizan el valor −9999 como indicador de ausencia de datos. Se implementó un filtro que detecta estos valores en cada banda y los reemplaza por interpolación con la mediana de los valores válidos de la misma banda. Las imágenes con más del 50 % de valores NoData fueron descartadas.

2. **Redimensionamiento:** Las imágenes originales (~111 × 111 píxeles) se redimensionaron a 224 × 224 píxeles mediante interpolación bicúbica con anti-aliasing, utilizando la función `resize` de scikit-image con modo de relleno reflectivo (`mode='reflect'`). El tamaño de 224 × 224 es estándar en aprendizaje profundo y compatible con arquitecturas de transferencia de aprendizaje.

3. **Normalización Z-score:** Cada banda se normalizó independientemente restando la media y dividiendo por la desviación estándar:

$$x_{\text{norm}} = \frac{x - \mu_{\text{banda}}}{\sigma_{\text{banda}}}$$

Esta normalización produce valores con media cercana a 0 y desviación estándar de 1, estabilizando el entrenamiento y mejorando la convergencia.

### 7.3 Aumento de datos (Data Augmentation)

Para aumentar la variabilidad y el tamaño del conjunto de datos se aplicaron **30 técnicas de aumento de datos** de forma offline, incluyendo transformaciones geométricas (rotaciones de 90°, 180° y 270°; volteos horizontal y vertical; trasposiciones), transformaciones de intensidad (brillo, contraste, gamma) y adición de ruido gaussiano. Cada imagen original generó aproximadamente 31 variantes, expandiendo el conjunto de datos de 200 a **6.200 imágenes** (3.441 positivas y 2.759 negativas).

Las augmentaciones se realizaron sobre las imágenes originales antes de la normalización, y los valores resultantes se acotaron al rango válido mediante la función `numpy.clip`. Todas las operaciones se ejecutaron en formato `float32` para evitar la duplicación del consumo de memoria asociado al formato `float64`.

#### 7.3.1 Ajuste del aumento de datos (v3)

Con la expansión del conjunto base a 2.019 imágenes, el factor de aumento se redujo de 30 a **10 técnicas** por imagen (Shorten y Khoshgoftaar, 2019, recomiendan disminuir la intensidad del aumento cuando el volumen base crece). Se seleccionaron las 10 transformaciones de mayor impacto: rotaciones (90°, 180°, 270°), volteos (horizontal, vertical), trasposición, ajuste de brillo (±15 %), contraste (±15 %), gamma y ruido gaussiano. Esto generó **22.209 imágenes aumentadas** (11.016 positivas y 11.193 negativas), con un desbalance de clase prácticamente nulo.

### 7.4 División del conjunto de datos

La división del conjunto de datos se realizó mediante **GroupShuffleSplit** de scikit-learn, agrupando por la imagen original de procedencia. Este método garantiza que todas las augmentaciones derivadas de una misma imagen original pertenezcan al mismo subconjunto (entrenamiento, validación o prueba), previniendo la fuga de datos (data leakage) que inflaría artificialmente las métricas de evaluación.

Las proporciones resultantes fueron:
- **Entrenamiento:** 4.223 imágenes (68 %)
- **Validación:** 960 imágenes (16 %)
- **Prueba:** 1.017 imágenes (16 %)

Los datos procesados se almacenaron en formato NumPy (`.npy`) particionado en archivos de aproximadamente 500 imágenes cada uno, para garantizar compatibilidad con el sistema de archivos FAT32 del dispositivo de almacenamiento externo utilizado.

#### 7.4.1 División anti-fuga geográfica (v3)

En la versión 3, el mecanismo de agrupación de GroupShuffleSplit se reforzó para considerar tanto los sufijos de aumento (`_aug01`, `_aug02`, …) como los de expansión por grilla (`_grid_r0_c1`, etc.). La función de extracción de grupo base elimina ambos tipos de sufijo, de modo que todas las variantes de una misma zona geográfica original permanecen en el mismo subconjunto.

Las proporciones resultantes del conjunto v3 fueron:
- **Entrenamiento:** 15.037 imágenes (67,7 %)
- **Validación:** 3.553 imágenes (16,0 %)
- **Prueba:** 3.619 imágenes (16,3 %)
- **Grupos geográficos:** 407 zonas base (ningún grupo compartido entre subconjuntos)
- **Pesos de clase:** ~1,0 (balance prácticamente perfecto)

La verificación de cero fuga de datos se realizó mediante scripts independientes (`_check_leakage.py`, `_check_splits.py`) que confirmaron 0 % de solapamiento entre los tres subconjuntos. Los datos v3 se almacenaron en formato NumPy particionado en disco externo NTFS de 931 GB, eliminando la restricción de tamaño de archivo de FAT32.

### 7.5 Arquitectura del modelo

#### 7.5.1 Modelo v2: CNN ResNet-inspired personalizada

La versión 2 empleó una CNN con arquitectura ResNet-inspired personalizada de **5.032.385 parámetros**, con un bloque inicial Conv2D(32, 7×7, stride=2), cuatro bloques residuales (64, 128, 256, 512 filtros), GlobalAveragePooling2D y clasificador Dense(256)→Dense(1, sigmoid). Esta arquitectura demostró una exactitud del 91,45 % en el conjunto de prueba colombiano.

#### 7.5.2 Modelo v3: EfficientNetB0 con Channel Adapter

Para la versión 3 se adoptó una estrategia de **Transfer Learning** con **EfficientNetB0** preentrenado en ImageNet (Tan y Le, 2019), combinado con un módulo adaptador de canales diseñado específicamente para las imágenes ASTER de 7 bandas.

**Channel Adapter (proyección 7 → 3 canales):**
- Conv2D(16, 3 × 3, padding='same') + BatchNormalization + ReLU: expande las 7 bandas ASTER a 16 mapas de características intermedios, aprendiendo combinaciones espectrales relevantes.
- Conv2D(3, 1 × 1, padding='same') + BatchNormalization + ReLU: proyecta los 16 mapas a los 3 canales esperados por EfficientNetB0.

Este adapter convolucional es preferible a una simple selección o promediado de bandas, ya que permite al modelo aprender la proyección óptima del espacio espectral ASTER al espacio RGB de ImageNet.

**Backbone: EfficientNetB0 (pesos ImageNet):**
- Arquitectura basada en bloques MBConv (Mobile Inverted Bottleneck) con Squeeze-and-Excitation.
- Escalado compuesto optimizado: profundidad × 1,0, ancho × 1,0, resolución 224 × 224.
- Parámetros del backbone: ~4,0 M.
- En la fase 1 de entrenamiento, todas las capas del backbone se mantienen congeladas.
- En la fase 2, se descongelan las **últimas 39 capas** para fine-tuning, manteniendo las capas de BatchNormalization congeladas para preservar las estadísticas aprendidas de ImageNet.

**Clasificador (head):**
- GlobalAveragePooling2D
- Dropout(0,3)
- Dense(256, activación ReLU)
- Dropout(0,3)
- Dense(1, activación sigmoid) — salida: probabilidad [0, 1]

**Total de parámetros:** 4.396.112 (menor que v2 con mayor capacidad representacional gracias al preentrenamiento)

### 7.6 Configuración del entrenamiento

#### 7.6.1 Entrenamiento v2 (CPU)

El entrenamiento de la v2 se ejecutó en CPU (Intel i5-10300H, 12 GB RAM) durante **22 épocas** (mejor época: 8, val_accuracy = 94,17 %) con AdamW, CosineDecay y EarlyStopping (patience=15). El sistema utilizó un generador de carga por particiones para evitar la saturación de la RAM.

#### 7.6.2 Entrenamiento v3 (GPU — dos fases)

Para la v3 se diseñó un protocolo de entrenamiento en **dos fases** ejecutado en GPU, habilitando **Mixed Precision (float16)** para duplicar el throughput efectivo y reducir el consumo de memoria.

**Fase 1 — Backbone congelado (30 épocas):**

| Parámetro | Valor |
|-----------|-------|
| Capas entrenables | Channel Adapter + Head (345.863 parámetros) |
| Backbone | 100 % congelado (pesos ImageNet) |
| Optimizador | AdamW (weight_decay = 1 × 10⁻⁴) |
| Learning rate | 1 × 10⁻³ con CosineDecay |
| Batch size | 32 |
| Épocas | 30 |
| Mejor época | 27 (val_auc = 0,9000) |
| MixUp | α = 0,2 |
| Label smoothing | 0,1 |
| EarlyStopping | patience = 10, monitor = val_auc |

**Fase 2 — Fine-tuning (50 épocas):**

| Parámetro | Valor |
|-----------|-------|
| Capas descongeladas | Últimas 39 capas del backbone |
| BatchNormalization | Congelado (preserve ImageNet stats) |
| Optimizador | AdamW (weight_decay = 1 × 10⁻⁴) |
| Learning rate | 1 × 10⁻⁴ con CosineDecay (10× menor que Fase 1) |
| Batch size | 32 |
| Épocas | 50 |
| Mejor época | 50 (val_auc = 0,9725) |
| MixUp | α = 0,2 |
| Label smoothing | 0,1 |
| EarlyStopping | patience = 15, monitor = val_auc |

**Parámetros comunes a ambas fases:**

| Parámetro | Valor |
|-----------|-------|
| Función de pérdida | BinaryCrossentropy (label_smoothing = 0,1) |
| Métricas monitoreadas | Accuracy, Precision, Recall, AUC, PR-AUC, F1Score |
| ModelCheckpoint | monitor = val_auc, save_best_only = True, mode = max |
| Pesos de clase | Calculados automáticamente |
| Semilla aleatoria | 42 (fija en todos los scripts) |
| Hardware | NVIDIA RTX 4070 12 GB VRAM, WSL2 Ubuntu 22.04 |
| Precisión mixta | float16 (política mixed_float16) |
| Total épocas | 80 (30 + 50) |

### 7.7 Herramientas y tecnologías

| Categoría | Herramientas |
|-----------|-------------|
| Aprendizaje profundo | TensorFlow 2.20.0, Keras 3.12.1 |
| GPU / Aceleración | NVIDIA RTX 4070 12 GB VRAM, CUDA 12.x, cuDNN, Mixed Precision float16 |
| Procesamiento de datos | NumPy, pandas, scikit-learn, scikit-image, OpenCV, SciPy, rasterio |
| Datos geoespaciales | Google Earth Engine API, NASA ASTER GED AG100 v003 |
| Visualización | Matplotlib, Seaborn, TensorBoard, Plotly, Folium |
| Interfaz web | Streamlit 1.54.0, streamlit-folium |
| Reportes | FPDF2 |
| Control de versiones | Git, GitHub |
| Lenguaje | Python 3.12.12 |
| Sistema operativo | Windows 11 (desarrollo), WSL2 Ubuntu 22.04 (entrenamiento GPU) |

### 7.8 Interfaz web

Se desarrolló una aplicación web interactiva con **Streamlit** y **Folium** que consta de 5 secciones:

1. **Inicio:** Descripción general del proyecto y del modelo.
2. **Predicción por coordenadas:** Permite al usuario ingresar coordenadas o seleccionarlas mediante clic en un mapa interactivo. El sistema descarga la imagen ASTER correspondiente, la procesa y devuelve la probabilidad de potencial geotérmico.
3. **Métricas:** Visualización dinámica de las métricas de evaluación del modelo (lectura desde archivo `evaluation_metrics.json`), curva ROC, matriz de confusión y comparativo v1 vs. v2.
4. **Arquitectura:** Diagrama de la arquitectura CNN y descripción de cada componente.
5. **Acerca de:** Información del equipo y del proyecto.

La sección de predicción incluye un mapa de calor (heatmap) de predicciones geotérmicas sobre el territorio colombiano, capas satelitales conmutables (OpenStreetMap, ESRI World Imagery, CartoDB), selector de zonas geotérmicas conocidas y funcionalidad de descarga de reportes en PDF.

---

## 8. RESULTADOS

### 8.1 Resumen de métricas del modelo v2

El modelo v2 fue evaluado en el conjunto de prueba compuesto por **1.017 imágenes** (552 positivas y 465 negativas), obteniendo los siguientes resultados:

| Métrica | Valor |
|---------|-------|
| Exactitud (Accuracy) | 91,45 % |
| Precisión (Precision) | 97,94 % |
| Sensibilidad (Recall) | 86,05 % |
| Puntaje F1 (F1-Score) | 91,61 % |
| Área bajo la curva ROC (ROC AUC) | 0,983 |
| Coeficiente de correlación de Matthews (MCC) | 0,837 |

### 8.2 Matriz de confusión

|  | Predicho Negativo | Predicho Positivo |
|---|:-:|:-:|
| **Real Negativo** | 455 (VN) | 10 (FP) |
| **Real Positivo** | 77 (FN) | 475 (VP) |

**Análisis de la matriz de confusión:**

- **Verdaderos Negativos (455):** El modelo identifica correctamente el 97,85 % de las zonas sin potencial geotérmico (especificidad).
- **Verdaderos Positivos (475):** El modelo detecta el 86,05 % de las zonas geotérmicas reales (sensibilidad).
- **Falsos Positivos (10):** Solo 10 de 465 zonas no geotérmicas fueron clasificadas incorrectamente como positivas, lo que da una tasa de falsos positivos de apenas el 2,15 %. Esto se traduce en la alta precisión del 97,94 %.
- **Falsos Negativos (77):** 77 de 552 zonas geotérmicas no fueron detectadas (13,95 % de los positivos reales).

La muy baja tasa de falsos positivos indica que, cuando el modelo predice potencial geotérmico, es altamente confiable. La tasa de falsos negativos del 13,95 % señala que un pequeño porcentaje de zonas geotérmicas podría pasar inadvertido, lo cual es aceptable en un contexto de screening donde las zonas identificadas se someterán a validación posterior.

### 8.3 Comparativo v1 vs. v2

| Métrica | v1 (baseline) | v2 (actual) | Mejora |
|---------|:---:|:---:|:---:|
| Exactitud | 68,43 % | **91,45 %** | +23,02 pp |
| Precisión | 86,32 % | **97,94 %** | +11,62 pp |
| Sensibilidad | 48,10 % | **86,05 %** | +37,95 pp |
| F1-Score | 61,77 % | **91,61 %** | +29,84 pp |
| ROC AUC | 0,8198 | **0,983** | +0,163 |
| MCC | −0,2673 | **0,837** | +1,104 |

**Nota:** Las métricas de la v1 estaban potencialmente infladas por la fuga de datos (data leakage) identificada en la auditoría. El rendimiento real de la v1 probablemente era inferior al reportado.

### 8.4 Objetivos vs. resultados

| Métrica | Objetivo mínimo | Objetivo ideal | Resultado v2 | Estado |
|---------|:-:|:-:|:-:|:-:|
| Exactitud | > 85 % | > 90 % | **91,45 %** | Superado |
| Precisión | > 80 % | > 85 % | **97,94 %** | Superado |
| Sensibilidad | > 80 % | > 85 % | **86,05 %** | Superado |
| F1-Score | > 80 % | > 85 % | **91,61 %** | Superado |
| ROC AUC | > 0,90 | > 0,95 | **0,983** | Superado |
| MCC | > 0,50 | > 0,70 | **0,837** | Superado |

Todas las métricas del modelo v2 superan tanto los objetivos mínimos como los objetivos ideales establecidos.

### 8.5 Análisis de factores de mejora

| Factor | Impacto estimado | Evidencia |
|--------|:---:|-----------|
| Mayor volumen de datos (85 → 200 imágenes originales) | Alto | Principal causa de mejora en generalización |
| Filtrado de NoData (−9999) | Alto | Eliminación de ruido que distorsionaba la normalización |
| Prevención de fuga de datos (GroupShuffleSplit) | Medio-Alto | Las métricas reflejan el rendimiento real del modelo |
| Eliminación de la doble normalización (z-score + Rescaling) | Medio | La señal z-score llega intacta al modelo |
| Eliminación de la doble regularización L2 | Medio | Modelo no sobre-regularizado |
| CosineDecay en lugar de ReduceLROnPlateau | Bajo-Medio | Convergencia más suave y estable |
| Incorporación de bandas de temperatura y NDVI | Medio | Información espectral complementaria |

### 8.6 Análisis de tendencias del entrenamiento

El modelo v2 entrenó durante 22 épocas antes de que el mecanismo de EarlyStopping detuviera el proceso. La mejor época fue la número 8, con una exactitud de validación del 94,17 %.

A diferencia de la v1, donde se observó un overfitting severo (exactitud de entrenamiento 91,75 % vs. exactitud de validación 44,70 % en la época 23), en la v2 el overfitting fue controlado gracias a:

- Mayor volumen de datos (200 vs. 85 imágenes originales).
- GroupShuffleSplit sin fuga de datos.
- Normalización z-score única (sin doble normalización).
- CosineDecay como schedule de learning rate.
- SpatialDropout2D, label smoothing y AdamW weight decay como mecanismos de regularización.

### 8.7 Contraste de la hipótesis

Con una exactitud del 91,45 % en un conjunto de prueba de 1.017 imágenes, y un ROC AUC de 0,983, se rechaza la hipótesis nula ($H_0: \text{Accuracy} \leq 0{,}50$) con amplio margen. El modelo CNN demuestra una capacidad discriminativa significativamente superior al azar para la clasificación binaria de zonas con y sin potencial geotérmico, confirmando la hipótesis de trabajo.

### 8.8 Resultados del modelo v3 (expansión andina)

#### 8.8.1 Conjunto de datos v3

El conjunto de datos v3 se construyó a partir de **2.019 imágenes base** de la Región Andina (997 positivas, 1.022 negativas), expandidas a **22.209 imágenes aumentadas** mediante 10 técnicas de aumento de datos. La división anti-fuga geográfica produjo:

| Subconjunto | Imágenes | Proporción |
|-------------|:--------:|:----------:|
| Entrenamiento | 15.037 | 67,7 % |
| Validación | 3.553 | 16,0 % |
| Prueba | 3.619 | 16,3 % |

Con **407 zonas geográficas base** independientes (cero solapamiento verificado entre subconjuntos) y pesos de clase equilibrados, el conjunto v3 supera al v2 en volumen (22.209 vs. 6.200 imágenes), diversidad geográfica (4 países vs. 1) y balance de clases.

**Tabla 11. Distribución del conjunto v3 por país**

| País | Positivas | Negativas | Total |
|------|:-:|:-:|:-:|
| Colombia | 544 | 477 | 1.021 |
| Ecuador | 184 | 66 | 250 |
| Perú | 107 | 28 | 135 |
| Chile | 162 | 451 | 613 |
| **Total** | **997** | **1.022** | **2.019** |

#### 8.8.2 Métricas del modelo v3

El modelo v3 (EfficientNetB0 + Channel Adapter) fue evaluado en el conjunto de prueba compuesto por **3.619 imágenes** (1.844 negativas y 1.772 positivas sin ningún solapamiento con los conjuntos de entrenamiento y validación), obteniendo los siguientes resultados:

**Tabla 15. Métricas de evaluación del modelo v3**

| Métrica | Valor |
|---------|-------|
| Exactitud (Accuracy) | 92,28 % |
| Precisión (Precision) | 91,27 % |
| Sensibilidad (Recall) | 93,17 % |
| Puntaje F1 (F1-Score) | 92,21 % |
| Área bajo la curva ROC (ROC AUC) | 0,9737 |
| Área bajo la curva PR (PR AUC) | 0,9693 |
| Coeficiente de correlación de Matthews (MCC) | 0,8458 |

**Tabla 16. Matriz de confusión del modelo v3**

|  | Predicho Negativo | Predicho Positivo |
|---|:-:|:-:|
| **Real Negativo** | 1.686 (VN) | 158 (FP) |
| **Real Positivo** | 121 (FN) | 1.651 (VP) |

**Análisis de la matriz de confusión v3:**

- **Verdaderos Negativos (1.686):** El modelo identifica correctamente el 91,43 % de las zonas sin potencial geotérmico (especificidad).
- **Verdaderos Positivos (1.651):** El modelo detecta el 93,17 % de las zonas geotérmicas reales (sensibilidad).
- **Falsos Positivos (158):** 158 de 1.844 zonas no geotérmicas fueron clasificadas como positivas (tasa FP = 8,57 %). Esta tasa es mayor que en v2 (2,15 %), lo que refleja un conjunto de prueba más diverso y desafiante con datos de 4 países.
- **Falsos Negativos (121):** 121 de 1.772 zonas geotérmicas no fueron detectadas (6,83 %), una mejora significativa respecto a la v2 (13,95 %).

El modelo v3 prioriza la sensibilidad (recall = 93,17 %) sobre la especificidad, lo cual es deseable en un contexto de screening geotérmico donde es preferible investigar un falso positivo que omitir una zona con potencial real.

#### 8.8.3 Comparativo v1 vs. v2 vs. v3

**Tabla 17. Comparativo de métricas entre las tres versiones del modelo**

| Métrica | v1 (baseline) | v2 (ResNet-inspired) | v3 (EfficientNetB0) | Mejora v2→v3 |
|---------|:---:|:---:|:---:|:---:|
| Exactitud | 68,43 % | 91,45 % | **92,28 %** | +0,83 pp |
| Precisión | 86,32 % | 97,94 % | **91,27 %** | −6,67 pp¹ |
| Sensibilidad | 48,10 % | 86,05 % | **93,17 %** | +7,12 pp |
| F1-Score | 61,77 % | 91,61 % | **92,21 %** | +0,60 pp |
| ROC AUC | 0,8198 | 0,983 | **0,9737** | −0,009² |
| MCC | −0,2673 | 0,837 | **0,8458** | +0,009 |
| Arquitectura | CNN custom | ResNet-inspired | EfficientNetB0 + Adapter | — |
| Parámetros | ~2 M | 5.032.385 | 4.396.112 | −12,6 % |
| Conjunto de prueba | 1.017 img (1 país) | 1.017 img (1 país) | 3.619 img (4 países) | ×3,6 |
| Fuga de datos | Sí (62,5 %) | No | No | — |

**Notas:**
1. La precisión de v3 (91,27 %) es menor que v2 (97,94 %) porque el conjunto de prueba v3 es 3,6× más grande, incluye 4 países con mayor diversidad geológica y el modelo v3 favorece la sensibilidad.
2. El ROC AUC de v2 (0,983) fue calculado sobre un conjunto de prueba colombiano más pequeño (1.017 imágenes). El AUC de v3 (0,9737) sobre 3.619 imágenes de 4 países sigue siendo excelente y refleja una capacidad discriminativa más robusta y generalizable.

**Tabla 18. Objetivos vs. resultados v3**

| Métrica | Objetivo mínimo | Objetivo ideal | Resultado v3 | Estado |
|---------|:-:|:-:|:-:|:-:|
| Exactitud | > 85 % | > 90 % | **92,28 %** | ✅ Superado |
| Precisión | > 80 % | > 85 % | **91,27 %** | ✅ Superado |
| Sensibilidad | > 80 % | > 85 % | **93,17 %** | ✅ Superado |
| F1-Score | > 80 % | > 85 % | **92,21 %** | ✅ Superado |
| ROC AUC | > 0,90 | > 0,95 | **0,9737** | ✅ Superado |
| MCC | > 0,50 | > 0,70 | **0,8458** | ✅ Superado |

Todas las métricas del modelo v3 superan tanto los objetivos mínimos como los objetivos ideales establecidos, confirmándose con un conjunto de prueba 3,6 veces mayor y geográficamente más diverso que el de v2.

---

## 9. DISCUSIÓN

Los resultados obtenidos a través de las tres versiones del modelo demuestran que las Redes Neuronales Convolucionales son una herramienta viable y eficaz para la identificación de zonas con potencial geotérmico a partir de datos de emisividad térmica e índices espectrales del sensor ASTER. La evolución de una CNN personalizada (v1, 68,43 %) a una arquitectura ResNet-inspired (v2, 91,45 %) y finalmente a Transfer Learning con EfficientNetB0 (v3, 92,28 %) ilustra cómo la combinación de correcciones metodológicas, expansión de datos y arquitecturas preentrenadas produce mejoras acumulativas significativas.

### 9.1 Interpretación de las métricas

La exactitud del 91,45 % indica que el modelo clasifica correctamente más de 9 de cada 10 imágenes. Sin embargo, las métricas más reveladoras son la precisión y la sensibilidad. La precisión del 97,94 % significa que, cuando el modelo identifica una zona como geotérmica, tiene una tasa de acierto casi perfecta: solo 10 de 485 predicciones positivas fueron erróneas. Esto es particularmente valioso en un contexto de exploración geotérmica, donde cada predicción positiva puede desencadenar inversiones costosas en estudios de campo.

La sensibilidad del 86,05 % indica que el modelo detecta la mayoría de las zonas geotérmicas (475 de 552), aunque deja sin identificar un 13,95 %. En el contexto de screening regional, esta tasa de falsos negativos es aceptable dado que el objetivo no es reemplazar sino complementar los métodos de exploración tradicionales.

El ROC AUC de 0,983 confirma que el modelo posee una capacidad discriminativa cercana a la perfección a través de todos los umbrales de clasificación posibles. El MCC de 0,837, una métrica más robusta ante desbalances de clases, corrobora la solidez del modelo.

### 9.2 Mejora significativa respecto a la v1

La transición de la v1 a la v2 produjo mejoras sustanciales en todas las métricas, con incrementos de 23,02 puntos porcentuales en exactitud, 37,95 en sensibilidad y 29,84 en F1-Score. El hallazgo más significativo es la mejora del recall, que pasó de 48,10 % a 86,05 %: la v1 dejaba sin detectar más de la mitad de las zonas geotérmicas, mientras que la v2 identifica correctamente el 86 %.

Esta mejora no se debe a un solo factor sino a la corrección acumulativa de 28 errores identificados en la auditoría de código, entre los cuales los más impactantes fueron: (1) el filtrado de valores NoData que distorsionaban la normalización, (2) la eliminación de la doble normalización que aplastaba la señal a un rango imperceptible, (3) la prevención de la fuga de datos que inflaba artificialmente las métricas, y (4) la eliminación de la doble regularización L2. A estos se sumó la expansión del conjunto de datos de 85 a 200 imágenes y la incorporación de 2 bandas adicionales (temperatura superficial y NDVI).

### 9.3 Comparación con trabajos relacionados

Si bien no existe un benchmark directo para la tarea de clasificación binaria de potencial geotérmico con CNN en Colombia, los resultados obtenidos son consistentes con el estado del arte en aplicaciones de CNN a la clasificación de imágenes satelitales. Estudios de Zhu et al. (2017) reportan exactitudes superiores al 85 % en tareas de clasificación comparables. Mia et al. (2018) reportó resultados satisfactorios con técnicas de Machine Learning para la identificación de alteraciones hidrotermales con datos ASTER, aunque emplearon métodos clásicos (Random Forest, SVM) en lugar de CNN profundas.

La transición a Transfer Learning con EfficientNetB0 en la v3 demostró que las características aprendidas en ImageNet (bordes, texturas, patrones espaciales) son transferibles al dominio de imágenes de emisividad térmica, incluso cuando el número de canales es diferente (7 bandas ASTER vs. 3 canales RGB). El Channel Adapter convolucional permitió aprender una proyección óptima del espacio espectral ASTER al espacio RGB, logrando resultados competitivos con menos parámetros (4,4 M vs. 5,0 M).

### 9.4 Descubrimiento crítico: fuga de datos (data leakage)

El hallazgo más crítico durante el desarrollo fue la identificación de una **fuga de datos del 62,5 %** en las divisiones originales del conjunto de datos. El mecanismo de GroupShuffleSplit de la v2, si bien agrupaba por imagen original, no consideraba los sufijos de expansión por grilla (`_grid_r0_c1`), lo que permitía que augmentaciones de la misma zona geográfica base aparecieran en múltiples subconjuntos.

Este problema fue detectado mediante los scripts de auditoría `_check_leakage.py` y `_check_splits.py`, y se corrigió con `resplit_data.py`, que aplica GroupShuffleSplit por zona geográfica base (eliminando ambos sufijos `_aug*` y `_grid_*`). El conjunto reprocesado (`processed_v2/`) contiene **407 zonas base** con cero solapamiento verificado entre los tres subconjuntos.

Este descubrimiento subraya la importancia crítica de auditar rigurosamente la división de datos en proyectos de aprendizaje profundo, especialmente cuando se aplican técnicas de aumento de datos.

### 9.5 Limitaciones del estudio

Es importante reconocer las siguientes limitaciones:

1. **Generalización geográfica:** En la v2 el modelo fue entrenado exclusivamente con zonas de Colombia; en la v3 se incorporaron datos de Ecuador, Perú y Chile, lo cual mitiga parcialmente esta limitación dentro del contexto andino. Sin embargo, su aplicabilidad a regiones fuera del Cinturón de Fuego del Pacífico no ha sido evaluada.

2. **Resolución temporal:** El producto ASTER GED es un promedio temporal, por lo que no captura variaciones estacionales ni eventos transitorios. Esto puede limitar la detección de fenómenos geotérmicos intermitentes.

3. **Tamaño del conjunto de datos:** La v3 amplió el conjunto base a 2.019 imágenes originales (frente a 200 en la v2), lo que mejora sustancialmente la representatividad. No obstante, conjuntos aún mayores (>5.000 imágenes) con cobertura global podrían fortalecer la generalización.

4. **Búsqueda de hiperparámetros:** Aunque la v3 se entrenó en GPU (NVIDIA RTX 4070 12 GB), la búsqueda de hiperparámetros se realizó de forma manual. Una búsqueda sistemática (grid search, Bayesian optimization) podría identificar configuraciones aún más óptimas.

5. **Validación en campo:** Las predicciones del modelo no han sido contrastadas con datos de prospección geotérmica in situ, lo cual queda fuera del alcance del proyecto pero es indispensable para la validación práctica.

6. **Etiquetado del conjunto de datos:** El etiquetado se basó en la literatura geológica existente, lo que introduce un sesgo hacia zonas ya conocidas. Zonas con potencial geotérmico no documentado podrían estar erróneamente etiquetadas como negativas.

### 9.6 Implicaciones prácticas

El modelo desarrollado se posiciona como una herramienta de screening automatizado para la **Fase 1 (Reconocimiento Regional)** de la exploración geotérmica. Su capacidad para analizar cualquier punto del territorio colombiano en segundos contrasta con los meses y los costos significativos que implica la revisión manual de imágenes satelitales por expertos.

Para el Servicio Geológico Colombiano y entidades como la UPME, el sistema ofrece la posibilidad de generar mapas de probabilidades de potencial geotérmico a escala nacional, lo que permitiría enfocar los recursos limitados de exploración en las zonas más prometedoras. El hecho de que el modelo tenga una tasa de falsos positivos de solo el 2,15 % lo hace especialmente confiable: las zonas que identifica como positivas merecen atención prioritaria.

### 9.7 Impacto de la expansión andina y Transfer Learning (v3)

La inclusión de zonas de Ecuador, Perú y Chile en el conjunto de entrenamiento obedece a dos razones: (1) incrementar el volumen de datos (de 200 a 2.019 imágenes base) y (2) exponer al modelo a una mayor variabilidad de patrones espectrales geotérmicos dentro de un contexto geológico compartido. Es importante destacar que la **aplicación** del modelo sigue circunscrita al territorio colombiano; la expansión andina busca exclusivamente mejorar la robustez del entrenamiento.

La adopción de Transfer Learning con EfficientNetB0 en la v3 representó un salto arquitectónico significativo respecto a la CNN ResNet-inspired de la v2. El entrenamiento en dos fases (backbone congelado → fine-tuning) permitió aprovechar las características genéricas aprendidas en ImageNet sin destruir los pesos preentrenados, resultando en un modelo con 12,6 % menos parámetros pero con una mejora de 7,12 pp en sensibilidad (recall). La regularización con MixUp (α=0,2) contribuyó a suavizar la frontera de decisión y mejorar la generalización a patrones geológicos no vistos durante el entrenamiento.

La implementación de intervalos de confianza Bootstrap al 95 % constituye una mejora metodológica significativa (Efron y Tibshirani, 1993).

---

## 10. CONCLUSIONES

1. Se desarrolló exitosamente un modelo predictivo basado en CNN que identifica zonas con potencial geotérmico en Colombia a partir de imágenes satelitales ASTER. La versión final (v3), basada en Transfer Learning con EfficientNetB0 y un Channel Adapter convolucional, alcanzó una exactitud del **92,28 %** sobre un conjunto de prueba de **3.619 imágenes** de 4 países andinos, superando el objetivo mínimo del 85 % y el objetivo ideal del 90 %.

2. Todas las métricas de evaluación de la v3 superaron los umbrales definidos: exactitud 92,28 %, precisión 91,27 %, sensibilidad 93,17 %, F1-Score 92,21 %, ROC AUC 0,9737 y MCC 0,8458. La hipótesis nula fue rechazada con amplio margen.

3. Se construyó un conjunto de datos multiescala que evolucionó de 200 imágenes colombianas (v2) a **2.019 imágenes base** de la Región Andina (Colombia, Ecuador, Perú y Chile), expandidas a **22.209 imágenes** mediante 10 técnicas de aumento de datos. La división anti-fuga geográfica con **407 zonas base** independientes y cero solapamiento garantiza la validez de las métricas reportadas.

4. La auditoría exhaustiva del código identificó y corrigió 28 errores en la versión 1 del pipeline (4 críticos, 4 de alta severidad, 10 medios y 10 bajos), lo que resultó en una mejora de v1 a v2 de 23,02 puntos porcentuales en exactitud y 37,95 en sensibilidad.

5. Se descubrió y corrigió una **fuga de datos del 62,5 %** en las divisiones originales, causada por la no eliminación de sufijos de grilla en el GroupShuffleSplit. Este hallazgo demuestra la importancia crítica de auditar rigurosamente la integridad de los datos en proyectos de aprendizaje profundo.

6. La adopción de Transfer Learning con EfficientNetB0 preentrenado en ImageNet demostró que las características visuales genéricas (bordes, texturas, patrones espaciales) son transferibles al dominio de imágenes de emisividad térmica, logrando resultados superiores con **12,6 % menos parámetros** (4.396.112 vs. 5.032.385) y un entrenamiento eficiente en dos fases con GPU (NVIDIA RTX 4070).

7. Se desarrolló una interfaz web interactiva con Streamlit y Folium que permite realizar predicciones de potencial geotérmico en tiempo real sobre cualquier coordenada del territorio colombiano, integrando mapas interactivos, capas satelitales, historial de predicciones y generación de reportes en PDF.

8. El modelo constituye una herramienta viable para la fase de reconocimiento regional en la exploración geotérmica, con potencial de reducir significativamente los costos y tiempos asociados a la revisión manual de imágenes satelitales.

---


## 11. RECOMENDACIONES

### 11.1 Recomendaciones cumplidas en la v3

1. **~~Ampliación del conjunto de datos:~~** ✅ **Cumplida.** La v3 expandió el conjunto de 200 a 2.019 imágenes base de la Región Andina (Colombia, Ecuador, Perú y Chile), aumentadas a 22.209 imágenes con 10 técnicas de aumento de datos.

2. **~~Acceso a GPU:~~** ✅ **Cumplida.** La v3 se entrenó en una NVIDIA RTX 4070 (12 GB VRAM) bajo WSL2 Ubuntu 22.04, habilitando Mixed Precision (float16) y un entrenamiento en dos fases de 80 épocas totales.

3. **~~Transfer Learning con EfficientNet:~~** ✅ **Cumplida.** La v3 implementó Transfer Learning con EfficientNetB0 preentrenado en ImageNet, combinado con un Channel Adapter convolucional para las 7 bandas ASTER.

4. **~~Técnicas avanzadas de aumento de datos:~~** ✅ **Parcialmente cumplida.** Se implementó MixUp (α=0,2) como técnica de regularización por interpolación de muestras. CutMix queda pendiente.

### 11.2 Recomendaciones pendientes para futuras iteraciones

5. **Validación en campo:** Contrastar las predicciones del modelo con datos de prospección geotérmica in situ en al menos 5–10 zonas clasificadas como positivas por el modelo pero no documentadas previamente, en colaboración con el Servicio Geológico Colombiano.

6. **Incorporación de bandas SWIR:** El sensor ASTER dispone de 6 bandas en el infrarrojo de onda corta que contienen información sobre alteraciones hidrotermales minerales. Su incorporación (13 bandas totales) podría mejorar la discriminación del modelo.

7. **Interpretabilidad:** Implementar técnicas como Grad-CAM (Gradient-weighted Class Activation Mapping) para visualizar qué regiones y bandas espectrales de las imágenes son más relevantes para las predicciones del modelo.

8. **Cobertura nacional sistemática:** Utilizar el modelo para generar un mapa completo de probabilidades de potencial geotérmico a escala nacional, procesando imágenes ASTER de forma sistemática sobre una cuadrícula que cubra todo el territorio colombiano.

9. **Integración institucional:** Explorar la integración del sistema con las plataformas del Servicio Geológico Colombiano y la UPME como herramienta de consulta y apoyo a la toma de decisiones en política energética.

10. **Búsqueda sistemática de hiperparámetros:** Realizar búsqueda automatizada (Bayesian optimization, Optuna) de la tasa de aprendizaje, arquitectura del adapter, número de capas descongeladas y factor de MixUp α.

11. **Expansión a otros contextos tectónicos:** Incorporar datos de zonas geotérmicas fuera de la Región Andina (Centroamérica, Indonesia, Islandia, Rift de África Oriental) para evaluar la transferibilidad global del modelo.

---

## REFERENCIAS

Abrams, M. y Hook, S. J. (2002). ASTER User Handbook, Version 2. Jet Propulsion Laboratory. https://asterweb.jpl.nasa.gov/content/03_data/04_Documents/aster_user_guide_v2.pdf

Alfaro, C. (2015). Evaluación del potencial geotérmico de Colombia. Servicio Geológico Colombiano.

Bona, P. y Coviello, M. (2016). *Valoración y gobernanza de los proyectos geotérmicos en América del Sur*. CEPAL.

Coolbaugh, M., Kratt, C., Fallacaro, A., Calvin, W. y Taranik, J. (2007). Detection of geothermal anomalies using Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) thermal infrared images at Bradys Hot Springs, Nevada, USA. *Remote Sensing of Environment*, 106(3), 350–359. https://doi.org/10.1016/j.rse.2006.09.001

DiPippo, R. (2015). *Geothermal Power Plants: Principles, Applications, Case Studies and Environmental Impact* (4.ª ed.). Elsevier.

Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.

Gehringer, M. y Loksha, V. (2012). *Geothermal Handbook: Planning and Financing Power Generation*. Energy Sector Management Assistance Program (ESMAP), Banco Mundial.

Goodfellow, I., Bengio, Y. y Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D. y Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18–27. https://doi.org/10.1016/j.rse.2017.06.031

He, K., Zhang, X., Ren, S. y Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Hulley, G. C., Hook, S. J., Abbott, E. y Malakar, N. (2015). The ASTER Global Emissivity Dataset (ASTER GED): Mapping Earth's emissivity at 100 meter spatial scale. *Geophysical Research Letters*, 42(19), 7966–7976. https://doi.org/10.1002/2015GL065564
INGEMMET. (2014). *Inventario de fuentes termales del Perú*. Instituto Geológico, Minero y Metalúrgico del Perú.
Ioffe, S. y Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448–456.

IPCC. (2011). *IPCC Special Report on Renewable Energy Sources and Climate Change Mitigation*. Cambridge University Press.

Lahsen, A. (1982). Upper Cenozoic volcanism and tectonism in the Andes of northern Chile. *Earth-Science Reviews*, 18(3), 285–302. https://doi.org/10.1016/0012-8252(82)90042-8

LeCun, Y., Bengio, Y. y Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. https://doi.org/10.1038/nature14539

Loshchilov, I. y Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

Lund, J. W. y Toth, A. N. (2021). Direct utilization of geothermal energy 2020 worldwide review. *Geothermics*, 90, 101915. https://doi.org/10.1016/j.geothermics.2020.101915

Mia, M. B., Fujimitsu, Y. y Nishijima, J. (2018). Exploration of hydrothermal alteration and monitoring of thermal activity using multi-temporal Landsat and ASTER satellite imagery: A case study at the Aso volcanic area, Japan. *Journal of Volcanology and Geothermal Research*, 368, 137–150.

Muñoz-Sáez, C., Manga, M. y Hurwitz, S. (2018). Hydrothermal discharge from the El Tatio basin, Atacama, Chile. *Journal of Volcanology and Geothermal Research*, 361, 25–35. https://doi.org/10.1016/j.jvolgeores.2018.07.007

NASA/METI/AIST/Japan Spacesystems. (2001). ASTER Global Emissivity Dataset (GED). [Conjunto de datos]. https://doi.org/10.5067/COMMUNITY/ASTER_GED/AG100.003

Servicio Geológico Colombiano. (2023). *Mapa de potencial geotérmico de Colombia*. SGC.

Shorten, C. y Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. https://doi.org/10.1186/s40537-019-0197-0

Siebert, L., Simkin, T. y Kimberly, P. (2010). *Volcanoes of the World* (3.ª ed.). Smithsonian Institution / University of California Press.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. y Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. y Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2818–2826.

Tan, M. y Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114. https://doi.org/10.48550/arXiv.1905.11946

Tester, J. W., Anderson, B. J., Batchelor, A. S., Blackwell, D. D., DiPippo, R., Drake, E. M., ... y Veatch, R. W. (2006). *The Future of Geothermal Energy: Impact of Enhanced Geothermal Systems (EGS) on the United States in the 21st Century*. Massachusetts Institute of Technology.

Vaughan, R. G., Hook, S. J., Calvin, W. M. y Taranik, J. V. (2005). Surface mineral mapping at Steamboat Springs, Nevada, USA, with multi-wavelength thermal infrared images. *Remote Sensing of Environment*, 99(1–2), 140–158. https://doi.org/10.1016/j.rse.2005.04.030

Yosinski, J., Clune, J., Bengio, Y. y Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems (NeurIPS)*, 27, 3320–3328.

Zhang, H., Cisse, M., Dauphin, Y. N. y Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *Proceedings of the 6th International Conference on Learning Representations (ICLR)*. https://doi.org/10.48550/arXiv.1710.09412

Zhu, X. X., Tuia, D., Mou, L., Xia, G.-S., Zhang, L., Xu, F. y Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. *IEEE Geoscience and Remote Sensing Magazine*, 5(4), 8–36. https://doi.org/10.1109/MGRS.2017.2762307

---

## ANEXOS

### Anexo A. Estructura del repositorio del proyecto

```
geotermia-colombia-cnn/
├── README.md
├── app.py                        # Interfaz gráfica Streamlit
├── config.py                     # Configuración centralizada
├── setup.py                      # Configuración del entorno
├── requirements.txt              # Dependencias Python
├── .gitignore
│
├── models/
│   ├── cnn_geotermia.py          # Arquitectura CNN v2 (5.032.385 params) + v3 EfficientNetB0 (4.396.112 params)
│   ├── __init__.py
│   └── saved_models/             # Modelos entrenados (.keras)
│
├── scripts/
│   ├── download_dataset.py       # Descarga + expansión por grilla (v3: 3 hilos)
│   ├── augment_full_dataset.py   # Aumento de datos (v2: 30, v3: 10 técnicas)
│   ├── prepare_dataset.py        # Preparación con anti-fuga geográfica
│   ├── train_model.py            # Entrenamiento del modelo v2
│   ├── train_model_v7.py         # Entrenamiento v3 (EfficientNetB0, 2 fases, GPU)
│   ├── evaluate_model.py         # Evaluación + Bootstrap CI
│   ├── predict.py                # Predicción por coordenadas (CLI)
│   ├── visualize_results.py      # Visualizaciones de resultados
│   └── visualize_architecture.py # Diagrama de arquitectura
│
├── data/
│   ├── raw/                      # v2: 200 | v3: 2.019 imágenes (.tif)
│   ├── augmented/                # v2: 6.200 | v3: 22.209 imágenes
│   └── processed/                # Archivos .npy particionados
│
├── docs/                         # Documentación técnica
│   ├── CAMPOS_GEOTERMICOS_REGION_ANDINA.md  # Catálogo andino (v3)
│   └── ...                       # Demás documentación
├── logs/                         # Logs de TensorBoard
├── results/                      # Métricas y figuras
└── notebooks/                    # Notebooks de exploración
```

### Anexo B. Zonas geotérmicas conocidas de Colombia utilizadas como referencia

| Zona | Departamento | Tipo | Temperatura estimada |
|------|-------------|------|:---:|
| Nevado del Ruiz – Macizo Volcánico | Caldas/Tolima | Volcánico-hidrotermal | > 200 °C |
| Chiles – Cerro Negro | Nariño | Volcánico-hidrotermal | > 200 °C |
| Azufral | Nariño | Volcánico-hidrotermal | > 200 °C |
| Paipa – Iza | Boyacá | Hidrotermal no volcánico | 150–200 °C |
| Coconucos | Cauca | Volcánico-hidrotermal | 150–200 °C |
| Santa Rosa de Cabal | Risaralda | Hidrotermal | ~150 °C |
| Chocontá – Machetá | Cundinamarca | Indicadores ASTER | En evaluación |

### Anexo C. Detalle de la auditoría de 28 errores corregidos

| N.º | Severidad | Descripción | Corrección |
|:---:|:---------:|------------|-----------|
| 1 | CRITICAL | NoData (−9999) no filtrado en la carga de datos | Filtrado + interpolación con mediana por banda |
| 2 | CRITICAL | Doble normalización: z-score + Rescaling(1/255) | Eliminada capa Rescaling; solo z-score |
| 3 | CRITICAL | Fuga de datos: augmentaciones de la misma imagen en train y test | GroupShuffleSplit agrupando por imagen original |
| 4 | CRITICAL | Doble regularización L2: kernel_regularizer + AdamW weight_decay | Eliminado kernel_regularizer; solo AdamW weight_decay |
| 5 | HIGH | RandomContrast aplicado sobre datos z-score (espera [0,1]) | Eliminado RandomContrast del pipeline online |
| 6 | HIGH | Doble aumento de datos: offline (30×) + online (Keras layers) | Desactivado aumento online para datos ya aumentados |
| 7 | HIGH | ReduceLROnPlateau en conflicto con AdamW | Reemplazado por CosineDecay integrado |
| 8 | HIGH | Excepciones silenciosas en predicción y carga de modelo | Logging completo con traceback |
| 9–18 | MEDIUM | Incluye: hardcoded project IDs, augmentaciones float64, carga completa en RAM, distancia euclidiana en lat/lon, curvas ROC sintéticas, métricas hardcodeadas, keywords incompletas, BatchNorm faltante en shortcut, training=False hardcodeado | Correcciones individuales documentadas en CHANGELOG_V2.md |
| 19–28 | LOW | Incluye: documentación incorrecta, valores en UI desactualizados, rutas hardcodeadas, falta de retry en descargas, R² en clasificación binaria, Adam vs. AdamW, métricas None mostrando 0 | Correcciones individuales documentadas en CHANGELOG_V2.md |

### Anexo D. Configuración del hardware

**Entorno de desarrollo (Windows):**

| Componente | Detalle |
|-----------|---------|
| CPU | Intel Core i5-10300H |
| RAM | 12 GB |
| GPU | NVIDIA RTX 4070 12 GB VRAM |
| Almacenamiento | Disco externo Toshiba NTFS 931 GB (D:\geotermia_datos) |
| Sistema operativo | Windows 11 |
| IDE | Visual Studio Code |

**Entorno de entrenamiento v3 (WSL2):**

| Componente | Detalle |
|-----------|---------|
| Sistema operativo | Ubuntu 22.04 (WSL2) |
| GPU | NVIDIA RTX 4070 12 GB VRAM (CUDA 12.x, cuDNN) |
| Python | 3.12.12 |
| TensorFlow | 2.20.0 (con soporte GPU) |
| Keras | 3.12.1 |
| Precisión mixta | float16 (política mixed_float16) |
| Streamlit | 1.54.0 |

**Entorno de entrenamiento v2 (CPU):**

| Componente | Detalle |
|-----------|---------|
| CPU | Intel Core i5-10300H |
| RAM | 12 GB |
| GPU | No disponible (TensorFlow sin CUDA en Windows) |
| Python | 3.10.11 |
| TensorFlow | 2.20.0 |

### Anexo E. Campos geotérmicos de la Región Andina incluidos en el conjunto v3

| País | Zona representativa | Tipo | Observaciones |
|------|-------------------|------|---------------|
| Colombia | Nevado del Ruiz, Chiles-Cerro Negro, Azufral, Paipa-Iza | Volcánico/Hidrotermal | Base del conjunto v1/v2 |
| Ecuador | Chachimbiro, Tufiño-Chiles, Chalupas, Chimborazo | Volcánico/Hidrotermal | Zona binacional con Colombia |
| Perú | Calientes (Tacna), Tutupaca, región Misti-Ubinas | Volcánico/Fuentes termales | >500 fuentes termales (INGEMMET, 2014) |
| Chile | El Tatio, Cerro Pabellón, Tolhuaca, Apacheta | Volcánico/Operativo | Cerro Pabellón: 48 MW operativo |

El catálogo completo con coordenadas, fuentes y clasificación se encuentra en `docs/CAMPOS_GEOTERMICOS_REGION_ANDINA.md`.

---

*Documento generado: febrero de 2026*
*Proyecto de Grado — Universidad de San Buenaventura, Bogotá*
*Facultad de Ingeniería — Programa de Ingeniería de Sistemas*
*Autores: Daniel Santiago Arévalo Rubiano, Cristian Camilo Vega Sánchez, Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín*
*Asesor: Prof. Yeison Eduardo Conejo Sandoval*
