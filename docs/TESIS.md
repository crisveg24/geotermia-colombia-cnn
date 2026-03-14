<!-- ============================================================
   PROYECTO DE GRADO — FORMATO APA 7.ª EDICIÓN (2020)
   Plantilla USB Colombia v.6 adaptada a Markdown
   ============================================================

   REGLAS DE FORMATO (extraídas del PDF de la USB):
   ─────────────────────────────────────────────────
   • Estilo: APA 7.ª edición (2020).
   • Márgenes: 2,5 cm.
   • Fuente sugerida: Times New Roman 12 (elegir UNA sola fuente,
     no mezclar). En tablas/figuras: 9-12 pt. Notas: 10 pt.
   • Interlineado: 1,5 en párrafos; 1,0 a 2,0 en tablas/figuras.
   • Texto: JUSTIFICADO en párrafos.
   • Sangría de primera línea: 1,25 cm (1 tabulador) en cada párrafo.
   • 1 espacio sencillo entre párrafo-párrafo y viñeta-viñeta.
   • 2 espacios sencillos entre título-párrafo, párrafo-imagen,
     imagen-párrafo, párrafo-subtítulo, tabla-párrafo.
   • No abusar de negritas ni cursivas.
   • Preferir comillas "inglesas" sobre «latinas».

   NIVELES DE TÍTULO APA:
   ─────────────────────
     Nivel 1: Número, centrado, negrita, minúsculas.
     Nivel 2: Número, alineado izquierda, negrita, minúsculas.
     Nivel 3: Número, alineado izquierda, negrita, cursiva, minúsculas.
     Nivel 4: Número, sangría 1,25, negrita, minúsculas, punto final, seguido del texto.
     Nivel 5: Número, sangría 1,25, negrita, cursiva, punto final, seguido del texto.

   REGLAS DE CITAS APA 7 + USB:
   ────────────────────────────
   • Cita textual < 40 palabras → dentro del párrafo, entre comillas,
     con (Autor, año, p. X).
   • Cita textual ≥ 40 palabras → bloque aparte, sangría francesa
     1,25 cm (sangría izquierda), SIN comillas, punto antes de la cita.
   • Paráfrasis → (Autor, año), sin número de página.
   • 1 autor: (Apellido, año).
   • 2 autores: (Apellido1 & Apellido2, año).
   • 3+ autores: (Apellido1 et al., año).
   • Corporativo primera vez: (Nombre Completo [Sigla], año);
     luego (Sigla, año).
   • TODA cita debe ir seguida de una JUSTIFICACIÓN: explicar
     POR QUÉ se incluye y qué aporta al argumento.
   • ~75 % redacción original, ~25 % citas de otros autores.

   NOTA TÉCNICA: Este archivo .md refleja la estructura del documento
   de Word final. Al pasarlo a Word se deben aplicar los estilos
   APA (Nivel 1 APA, Nivel 2 APA, Párr.APA, etc.) configurados
   en la plantilla USB.
   ============================================================ -->


---

# Identificación de zonas con potencial geotérmico en Colombia mediante redes neuronales convolucionales aplicadas a imágenes satelitales ASTER


Cristian Camilo Vega Sánchez

Daniel Santiago Arévalo Rubiano

Yuliet Katerin Espitia Ayala

Laura Sophie Rivera Martín


Proyecto de grado para optar al título de Ingeniero de Sistemas


Asesor: Yeison Eduardo Conejo Sandoval, Magíster en Ingeniería de Sistemas


Universidad de San Buenaventura

Facultad de Ingeniería

Programa de Ingeniería de Sistemas

Bogotá, D. C.

2026

---

<!-- ============================================================
     PÁGINA LEGAL
     ============================================================ -->

## Citar / How to cite

(Vega Sánchez et al., 2026)

**Referencia / Reference — Estilo APA 7.ª ed. (2020):**

Vega Sánchez, C. C., Arévalo Rubiano, D. S., Espitia Ayala, Y. K., & Rivera Martín, L. S. (2026). *Identificación de zonas con potencial geotérmico en Colombia mediante redes neuronales convolucionales aplicadas a imágenes satelitales ASTER* [Proyecto de grado]. Universidad de San Buenaventura, Bogotá.

**Repositorio Institucional:** www.bibliotecadigital.usb.edu.co

---

<!-- ============================================================
     DEDICATORIA (opcional)
     ============================================================ -->

## Dedicatoria

*(Texto de dedicatoria — a cargo de los autores.)*

---

<!-- ============================================================
     AGRADECIMIENTOS (opcional)
     ============================================================ -->

## Agradecimientos

*(Texto de agradecimientos — a cargo de los autores.)*

---

<!-- ============================================================
     TABLA DE CONTENIDO
     (Se genera automáticamente en Word; aquí se lista la estructura)
     ============================================================ -->

## Tabla de contenido

- Resumen
- Abstract
- Introducción
- 1 Planteamiento del problema
  - 1.1 Antecedentes
  - 1.2 Formulación del problema
- 2 Justificación
- 3 Objetivos
  - 3.1 Objetivo general
  - 3.2 Objetivos específicos
- 4 Hipótesis
  - 4.1 Hipótesis alternativa
  - 4.2 Hipótesis nula
  - 4.3 Criterios de aceptación
- 5 Marco teórico
  - 5.1 Energía geotérmica
  - 5.2 Teledetección y sensor ASTER
  - 5.3 Aprendizaje profundo (deep learning)
    - 5.3.1 Tensores: la estructura de datos fundamental
    - 5.3.2 Redes neuronales convolucionales (CNN)
    - 5.3.3 Parámetros del modelo
    - 5.3.4 Funciones de activación de salida: sigmoid vs. softmax
    - 5.3.5 Función de pérdida: Binary Cross-Entropy
  - 5.4 Transfer learning y EfficientNet
    - 5.4.1 Arquitectura EfficientNet
    - 5.4.2 Bloque MBConv y Squeeze-and-Excitation
    - 5.4.3 Channel Adapter
  - 5.5 Técnicas de regularización y optimización
  - 5.6 Herramientas tecnológicas
    - 5.6.1 TensorFlow y Keras
    - 5.6.2 CUDA y aceleración por GPU
    - 5.6.3 Mixed Precision Training
  - 5.7 Estado del arte
- 6 Metodología
  - 6.1 Enfoque y tipo de investigación
  - 6.2 Dataset: Región Andina
  - 6.3 Pipeline de procesamiento
  - 6.4 Arquitectura del modelo
  - 6.5 Estrategia de entrenamiento
  - 6.6 Métricas de evaluación
  - 6.7 Intervalos de confianza Bootstrap
- 7 Resultados
  - 7.1 Métricas del modelo v3
  - 7.2 Matriz de confusión
  - 7.3 Comparativo v1 → v2 → v3
  - 7.4 Curvas de entrenamiento
  - 7.5 Contraste de hipótesis
- 8 Discusión
  - 8.1 Interpretación de resultados
  - 8.2 Comparación con trabajos relacionados
  - 8.3 Bugs descubiertos y corregidos
  - 8.4 Implicaciones prácticas
  - 8.5 Limitaciones
- 9 Conclusiones
- 10 Recomendaciones
  - 10.1 Recomendaciones cumplidas en la v3
  - 10.2 Recomendaciones pendientes
- Referencias
- Anexos

---

## Lista de tablas

- Tabla 1. Componentes del triángulo geotérmico
- Tabla 2. Zonas geotérmicas conocidas en Colombia
- Tabla 3. Bandas espectrales ASTER utilizadas
- Tabla 4. Indicadores geotérmicos detectables por satélite
- Tabla 5. Dimensiones de tensores y sus equivalentes cotidianos
- Tabla 6. Distribución de parámetros del modelo
- Tabla 7. Dimensiones de escalado en EfficientNet
- Tabla 8. Composición del dataset v3 por país
- Tabla 9. Resultado del aumento de datos
- Tabla 10. División anti-fuga geográfica del dataset
- Tabla 11. Hiperparámetros de la Fase 1 de entrenamiento
- Tabla 12. Hiperparámetros de la Fase 2 de entrenamiento
- Tabla 13. Técnicas de regularización empleadas
- Tabla 14. Métricas del modelo v3 sobre el conjunto de prueba
- Tabla 15. Matriz de confusión del modelo v3
- Tabla 16. Comparativo de métricas v1, v2 y v3
- Tabla 17. Bugs críticos descubiertos en la auditoría v1 → v2
- Tabla 18. Bugs críticos descubiertos en la auditoría v2 → v3
- Tabla 19. Evolución del proyecto v1 → v2 → v3

---

## Lista de figuras

- Figura 1. Arquitectura del modelo EfficientNetB0 + Channel Adapter
- Figura 2. Diagrama del pipeline de procesamiento
- Figura 3. Curvas de entrenamiento (fases 1 y 2)
- Figura 4. Ejemplo de predicción paso a paso

---

<!-- ============================================================
     RESUMEN
     ============================================================ -->

## Resumen

La exploración de recursos geotérmicos en Colombia se encuentra en una etapa incipiente, a pesar de que el país posee un potencial estimado superior a 2.000 MW debido a su ubicación en el Cinturón de Fuego del Pacífico. La fase inicial de reconocimiento regional, que busca priorizar zonas para inversión en exploración detallada, se ha realizado tradicionalmente mediante métodos manuales costosos y lentos. El presente proyecto propone un modelo de clasificación binaria basado en redes neuronales convolucionales (CNN), específicamente una arquitectura EfficientNetB0 con un módulo adaptador de canales (Channel Adapter), entrenado con 22.209 imágenes satelitales del sensor ASTER provenientes de cuatro países andinos (Colombia, Ecuador, Perú y Chile). El modelo recibe imágenes de siete bandas espectrales (cinco de emisividad térmica, temperatura superficial e índice de vegetación) y produce una probabilidad de potencial geotérmico. Se empleó transfer learning con pesos preentrenados en ImageNet y una estrategia de entrenamiento en dos fases (backbone congelado seguido de fine-tuning). Se implementó una división anti-fuga geográfica mediante GroupShuffleSplit que garantiza cero porciento de solapamiento entre los conjuntos de entrenamiento, validación y prueba. El modelo alcanzó una exactitud del 92,28 %, una sensibilidad del 93,17 % y un AUC-ROC de 0,9737 sobre un conjunto de prueba de 3.619 imágenes, superando todos los objetivos planteados. Se desarrolló adicionalmente una interfaz web con Streamlit que permite realizar predicciones en tiempo real a partir de coordenadas geográficas.

**Palabras clave:** energía geotérmica, redes neuronales convolucionales, imágenes satelitales, ASTER, transfer learning, EfficientNet, Colombia, Región Andina

---

<!-- ============================================================
     ABSTRACT
     ============================================================ -->

## Abstract

Geothermal resource exploration in Colombia remains in an early stage, despite the country's estimated potential exceeding 2,000 MW owing to its location along the Pacific Ring of Fire. The initial regional reconnaissance phase, aimed at prioritizing areas for detailed exploration investment, has traditionally relied on costly and time-consuming manual methods. This project proposes a binary classification model based on convolutional neural networks (CNN), specifically an EfficientNetB0 architecture with a Channel Adapter module, trained on 22,209 satellite images from the ASTER sensor spanning four Andean countries (Colombia, Ecuador, Peru, and Chile). The model receives seven-band spectral images (five thermal emissivity bands, surface temperature, and vegetation index) and outputs a geothermal potential probability. Transfer learning with ImageNet-pretrained weights and a two-phase training strategy (frozen backbone followed by fine-tuning) were employed. A geographic anti-leakage split using GroupShuffleSplit ensures zero percent overlap between training, validation, and test sets. The model achieved 92.28% accuracy, 93.17% recall, and an AUC-ROC of 0.9737 on a test set of 3,619 images, surpassing all proposed objectives. Additionally, a Streamlit web interface was developed to enable real-time predictions from geographic coordinates.

**Keywords:** geothermal energy, convolutional neural networks, satellite imagery, ASTER, transfer learning, EfficientNet, Colombia, Andean Region

---

<!-- ============================================================
     INTRODUCCIÓN
     ============================================================ -->

## Introducción

La energía geotérmica constituye una fuente de energía renovable que aprovecha el calor almacenado en el interior de la Tierra, proveniente tanto del calor primordial de la formación del planeta como del decaimiento radiactivo de isótopos como uranio-238, torio-232 y potasio-40 (Dickson & Fanelli, 2003). Este calor genera un gradiente geotérmico promedio de 25 a 30 °C por kilómetro de profundidad que, en zonas volcánicas, puede superar los 100 °C por kilómetro. Esta fuente energética se incluye como una de las alternativas clave dentro de la transición hacia matrices energéticas más limpias, dado que produce emisiones de CO₂ sustancialmente menores en comparación con los combustibles fósiles (International Energy Agency [IEA], 2021).

Colombia se ubica en el Cinturón de Fuego del Pacífico, donde convergen las placas tectónicas de Nazca, Sudamericana y del Caribe, lo cual genera una intensa actividad volcánica y geotérmica. El Servicio Geológico Colombiano (SGC, 2023) estima un potencial geotérmico superior a 2.000 MW; no obstante, a la fecha no existe ninguna planta geotérmica operativa en el país. Esta brecha entre potencial y aprovechamiento motiva la búsqueda de herramientas que aceleren la fase inicial de exploración, específicamente el reconocimiento regional, cuyo costo oscila entre 50.000 y 200.000 dólares por proyecto (Gehringer & Loksha, 2012).

El avance reciente del aprendizaje profundo (deep learning) ha demostrado que las redes neuronales convolucionales (CNN) son capaces de extraer patrones complejos a partir de datos geoespaciales, incluyendo la identificación de anomalías térmicas y alteraciones mineralógicas en imágenes satelitales multiespectrales (Zhu et al., 2017), lo que valida la pertinencia del enfoque propuesto en este trabajo.

En este contexto, el presente proyecto propone el desarrollo de un modelo CNN basado en la arquitectura EfficientNetB0 (Tan & Le, 2019), complementado con un módulo adaptador de canales (Channel Adapter), para clasificar imágenes del sensor ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) y determinar la probabilidad de que una zona geográfica posea potencial geotérmico. El modelo fue entrenado con imágenes de cuatro países andinos que comparten un contexto geológico común de subducción y vulcanismo activo — Colombia, Ecuador, Perú y Chile —, y su aplicación se circunscribe al territorio colombiano.

La investigación se enmarca en un enfoque cuantitativo de tipo experimental-aplicado, y su resultado principal es una herramienta de screening automatizado que contribuye a la priorización de zonas para inversión en exploración geotérmica detallada.

---

<!-- ============================================================
     1. PLANTEAMIENTO DEL PROBLEMA
     ============================================================ -->

## 1 Planteamiento del problema

La exploración geotérmica convencional sigue una secuencia de cuatro fases: reconocimiento regional (50.000 – 200.000 USD), exploración de superficie (200.000 – 1.000.000 USD), exploración profunda (5 – 20 millones USD) y desarrollo (50 – 200 millones USD) (Gehringer & Loksha, 2012). Esta guía del Banco Mundial detalla los costos y fases de proyectos geotérmicos, y constituye la referencia estándar en la materia. La primera fase depende tradicionalmente de la recopilación manual de datos geológicos, análisis de imágenes satelitales por expertos y revisión bibliográfica, procesos que resultan lentos, costosos y limitados por la subjetividad del analista.

Colombia posee al menos seis zonas geotérmicas reconocidas por el Servicio Geológico Colombiano — Nevado del Ruiz, Chiles-Cerro Negro, Azufral, Paipa-Iza, Puracé-Coconucos y Santa Rosa de Cabal — con temperaturas estimadas entre 150 y más de 200 °C (SGC, 2023). A pesar de este potencial, la falta de herramientas automatizadas de reconocimiento regional limita la capacidad de identificar nuevas zonas candidatas a exploración.

Las técnicas de aprendizaje profundo han mostrado resultados prometedores en la clasificación de imágenes de teledetección. Según Zhu et al. (2017), "deep learning has achieved remarkable success in remote sensing image analysis due to its ability to automatically learn hierarchical feature representations from raw data" (p. 5). Esta capacidad de aprender representaciones de forma automática, sin extracción manual de características, fundamenta el enfoque técnico adoptado en el presente proyecto.

Sin embargo, la aplicación específica de CNN a la identificación de potencial geotérmico a partir de datos multiespectrales de emisividad térmica es un campo escasamente explorado, lo que plantea la pregunta de investigación que guía este trabajo.


### 1.1 Antecedentes

El uso de teledetección para la exploración geotérmica tiene antecedentes desde la década de 1990, cuando los datos del sensor ASTER comenzaron a emplearse para mapear alteraciones hidrotermales y anomalías térmicas. Abrams et al. (2015) documentaron el uso extensivo de ASTER en geología, destacando que las bandas de emisividad térmica permiten identificar minerales indicadores de actividad geotérmica como cuarzo, feldespatos y arcillas, lo que respalda la selección de estas bandas como variables de entrada del modelo.

En el ámbito del aprendizaje automático aplicado a geotermia, Coolbaugh et al. (2007) propusieron modelos estadísticos para predecir potencial geotérmico usando variables geoespaciales, aunque sin emplear deep learning. Más recientemente, estudios como los de Flores-Espino et al. (2019) han utilizado redes neuronales superficiales para evaluar recursos geotérmicos, pero la aplicación de arquitecturas profundas modernas con transfer learning a imágenes multiespectrales ASTER no ha sido explorada de manera sistemática, lo que constituye el vacío metodológico que el presente proyecto busca cubrir.


### 1.2 Formulación del problema

¿Es posible desarrollar un modelo de clasificación basado en redes neuronales convolucionales que, a partir de imágenes satelitales multiespectrales del sensor ASTER, identifique zonas con potencial geotérmico en Colombia con una exactitud superior al 85 % y un AUC-ROC superior a 0,90?

---

<!-- ============================================================
     2. JUSTIFICACIÓN
     ============================================================ -->

## 2 Justificación

Colombia requiere diversificar su matriz energética para reducir la dependencia de la generación hidroeléctrica, que representa más del 65 % de la capacidad instalada y es vulnerable a fenómenos climáticos como El Niño (Unidad de Planeación Minero Energética [UPME], 2020). La geotermia, como fuente de generación base disponible las 24 horas del día, complementaría la generación existente sin depender de condiciones climáticas.

Desde la perspectiva tecnológica, el aprendizaje profundo ha demostrado ser capaz de superar el desempeño humano en tareas de clasificación de imágenes cuando se dispone de conjuntos de datos suficientemente grandes y representativos (LeCun et al., 2015). LeCun, Bengio y Hinton son considerados los padres del deep learning moderno; su revisión establece los fundamentos teóricos que respaldan la viabilidad del enfoque CNN propuesto en este trabajo.

Desde la perspectiva económica, una herramienta de screening automatizado que reduzca la lista de zonas candidatas antes de invertir en exploración de campo tiene el potencial de ahorrar millones de dólares al descartarse zonas sin potencial en la fase más temprana y económica del proceso (Gehringer & Loksha, 2012).

Desde la perspectiva académica, este proyecto aporta una metodología reproducible que integra teledetección, procesamiento de datos geoespaciales y deep learning, contribuyendo a un campo de investigación incipiente en Colombia. Además, el dataset construido — con 2.019 zonas georreferenciadas de cuatro países andinos — constituye un recurso reutilizable para investigaciones futuras.

---

<!-- ============================================================
     3. OBJETIVOS
     ============================================================ -->

## 3 Objetivos

### 3.1 Objetivo general

Desarrollar un modelo de clasificación basado en redes neuronales convolucionales que permita identificar zonas con potencial geotérmico en Colombia a partir de imágenes satelitales multiespectrales del sensor ASTER.

### 3.2 Objetivos específicos

- Construir un dataset georreferenciado de imágenes ASTER de siete bandas espectrales para zonas con y sin potencial geotérmico confirmado en la Región Andina (Colombia, Ecuador, Perú y Chile).
- Diseñar e implementar un pipeline de procesamiento de imágenes que incluya filtrado de valores faltantes, redimensionamiento, normalización global y aumento de datos, con una división anti-fuga geográfica que garantice cero porciento de solapamiento entre conjuntos.
- Implementar una arquitectura CNN basada en EfficientNetB0 con un módulo Channel Adapter que proyecte las siete bandas espectrales al espacio de tres canales requerido por el modelo base.
- Entrenar el modelo mediante una estrategia de transfer learning en dos fases (backbone congelado y fine-tuning) y evaluar su desempeño con métricas de exactitud, sensibilidad, precisión, F1-Score y AUC-ROC.
- Desarrollar una interfaz web que permita realizar predicciones de potencial geotérmico en tiempo real a partir de coordenadas geográficas.

---

<!-- ============================================================
     4. HIPÓTESIS
     ============================================================ -->

## 4 Hipótesis

### 4.1 Hipótesis de trabajo ($H_1$)

Un modelo de clasificación basado en redes neuronales convolucionales, entrenado con imágenes multiespectrales del sensor ASTER de la Región Andina, es capaz de identificar zonas con potencial geotérmico en Colombia con una exactitud significativamente superior al azar (Accuracy > 50 %) y con un AUC-ROC superior a 0,90.

### 4.2 Hipótesis nula ($H_0$)

El modelo CNN no es capaz de discriminar entre zonas con y sin potencial geotérmico a partir de imágenes ASTER con un desempeño superior al azar:

$$H_0: \text{Accuracy} \leq 0{,}50 \tag{9}$$

### 4.3 Criterios de aceptación

Se rechaza $H_0$ si el modelo alcanza simultáneamente:

- Exactitud (Accuracy) > 85 % (objetivo mínimo) o > 90 % (objetivo ideal)
- AUC-ROC > 0,90 (objetivo mínimo) o > 0,95 (objetivo ideal)
- MCC > 0,50 (objetivo mínimo) o > 0,70 (objetivo ideal)

evaluados sobre un conjunto de prueba independiente sin solapamiento con los conjuntos de entrenamiento y validación.

---

<!-- ============================================================
     5. MARCO TEÓRICO
     ============================================================ -->

## 5 Marco teórico

### 5.1 Energía geotérmica

La energía geotérmica es el calor almacenado en el interior de la Tierra. Para que un recurso geotérmico sea explotable, se requieren tres componentes conocidos como el "triángulo geotérmico": una fuente de calor (intrusión magmática o gradiente elevado), un reservorio (roca con porosidad y permeabilidad suficientes) y un fluido (agua líquida o vapor) que transporte el calor (Dickson & Fanelli, 2003), texto de referencia fundamental en geotermia.

**Tabla 1**

*Componentes del triángulo geotérmico*

| Componente | Descripción |
|------------|-------------|
| Fuente de calor | Intrusión magmática o gradiente geotérmico elevado |
| Reservorio | Roca con porosidad y permeabilidad suficientes |
| Fluido | Agua (líquida o vapor) que transporta el calor |

*Nota.* Basado en Dickson y Fanelli (2003).

Los sistemas geotérmicos se clasifican en tres categorías principales. Los sistemas hidrotermales convencionales se manifiestan como aguas termales, fumarolas y géiseres, con temperaturas de 150 a 350 °C a profundidades de 1 a 3 km; un ejemplo es el Nevado del Ruiz en Colombia. Los sistemas geotérmicos mejorados (EGS, por sus siglas en inglés) consisten en roca caliente seca donde se inyecta agua a presión para crear fracturas artificiales. Finalmente, los sistemas de uso directo y bombas de calor geotérmico (GSHP) aprovechan la temperatura estable del subsuelo cercano a los 15 °C y no requieren vulcanismo (DiPippo, 2012). Esta taxonomía permite clasificar el tipo de recurso que el modelo busca identificar.

En el contexto colombiano, el SGC (2023) ha identificado al menos seis zonas geotérmicas de interés:

**Tabla 2**

*Zonas geotérmicas conocidas en Colombia*

| Zona | Departamento | Tipo | Temperatura estimada |
|------|-------------|------|:--------------------:|
| Nevado del Ruiz | Caldas/Tolima | Volcánico-hidrotermal | > 200 °C |
| Chiles-Cerro Negro | Nariño | Volcánico-hidrotermal | > 200 °C |
| Azufral | Nariño | Volcánico-hidrotermal | > 200 °C |
| Paipa-Iza | Boyacá | Hidrotermal no volcánico | 150–200 °C |
| Puracé-Coconucos | Cauca | Volcánico-hidrotermal | 150–200 °C |
| Santa Rosa de Cabal | Risaralda | Hidrotermal | ~150 °C |

*Nota.* Datos del Servicio Geológico Colombiano (SGC, 2023).

Los países andinos — Colombia, Ecuador, Perú y Chile — comparten el contexto geológico del Cinturón de Fuego del Pacífico, con subducción de la placa de Nazca bajo la placa Sudamericana. Esta similitud geológica justifica expandir el dataset de entrenamiento a la región andina completa. Chile, que opera la primera planta geotérmica de Sudamérica en Cerro Pabellón (48 MW), demuestra la viabilidad del aprovechamiento de estos recursos en la región (Lahsen et al., 2015). Esta similitud geológica entre países andinos fundamenta la decisión de construir un dataset multinacional para entrenar el modelo.


### 5.2 Teledetección y sensor ASTER

El sensor ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer), a bordo del satélite Terra de la NASA, dispone de 14 bandas espectrales distribuidas en tres subsistemas: VNIR (3 bandas), SWIR (6 bandas) y TIR (5 bandas). Las bandas TIR son particularmente relevantes para la detección de anomalías térmicas superficiales y la caracterización de composición mineralógica (Abrams et al., 2015), capacidades que fundamentan la selección de este sensor para el presente estudio.

El presente estudio utiliza el producto ASTER GED AG100 v003 (NASA/METI/AIST/Japan Spacesystems), disponible en Google Earth Engine con el identificador NASA/ASTER_GED/AG100_003. Este producto ofrece una resolución espacial de 100 metros y cobertura global, y consiste en un promedio temporal que no captura variaciones estacionales.

**Tabla 3**

*Bandas espectrales ASTER utilizadas en el modelo*

| N.° | Banda | Longitud de onda | Utilidad geotérmica |
|:---:|-------|-----------------|---------------------|
| 1 | emissivity_band10 | 8,125–8,475 μm | Detección de cuarzo caliente |
| 2 | emissivity_band11 | 8,475–8,825 μm | Identificación de feldespatos |
| 3 | emissivity_band12 | 8,925–9,275 μm | Detección de minerales arcillosos |
| 4 | emissivity_band13 | 10,25–10,95 μm | Temperatura superficial |
| 5 | emissivity_band14 | 10,95–11,65 μm | Anomalías térmicas |
| 6 | temperature | Temperatura superficial (°C × 100) | Indicador directo de calor |
| 7 | ndvi | Índice de vegetación normalizado | Proxy de cobertura vegetal |

*Nota.* Las cinco bandas de emisividad (TIR) detectan anomalías térmicas y composición mineral. La temperatura ofrece un indicador directo de calor superficial. El NDVI actúa como proxy de cobertura vegetal, que puede enmascarar señales térmicas.

**Tabla 4**

*Indicadores geotérmicos detectables por satélite*

| Indicador | Detectable | Banda relevante |
|-----------|:----------:|-----------------|
| Anomalías térmicas superficiales | Sí | TIR (bandas 10–14) |
| Alteración hidrotermal de minerales | Sí | TIR |
| Composición mineralógica (arcillas, sílice) | Sí | TIR |
| Gradiente geotérmico elevado | No (indirecto) | — |
| Presencia de vulcanismo reciente | Sí | TIR, temperature |


### 5.3 Aprendizaje profundo (deep learning)

El aprendizaje profundo es una rama del aprendizaje automático que emplea redes neuronales artificiales con múltiples capas para aprender representaciones jerárquicas de los datos. LeCun et al. (2015) señalan que "deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction" (p. 436). Esta definición canónica, formulada por tres de los creadores del campo, fundamenta teóricamente la elección de este enfoque sobre métodos de aprendizaje automático convencionales.

#### 5.3.1 Tensores: la estructura de datos fundamental

Un tensor es una generalización de vectores y matrices a un número arbitrario de dimensiones, y constituye la estructura de datos sobre la que operan todas las redes neuronales. El término proviene de la física y la matemática, y fue adoptado en el aprendizaje profundo porque describe con precisión la naturaleza multidimensional de los datos procesados — de hecho, TensorFlow significa literalmente "flujo de tensores" (Abadi et al., 2016).

**Tabla 5**

*Dimensiones de tensores y sus equivalentes cotidianos*

| Estructura | Dimensiones | Ejemplo en el proyecto |
|---|---|---|
| Escalar | 0D | Un valor de temperatura: 28,5 |
| Vector | 1D | Los 7 valores espectrales de un píxel |
| Matriz | 2D | Una banda de la imagen: 224 × 224 |
| Tensor 3D | 3D | Una imagen ASTER completa: 224 × 224 × 7 |
| Tensor 4D | 4D | Un batch de entrenamiento: 32 × 224 × 224 × 7 |

Cada imagen ASTER que recibe el modelo es un tensor de tres dimensiones con forma (224, 224, 7), lo que equivale a 351.232 valores numéricos — un número por cada banda espectral en cada píxel. Cuando el modelo procesa un batch de 32 imágenes simultáneamente, el tensor de entrada tiene cuatro dimensiones (32, 224, 224, 7), es decir, 11.239.424 valores procesados en paralelo en cada step de entrenamiento.

#### 5.3.2 Redes neuronales convolucionales (CNN)

Las redes neuronales convolucionales (CNN) son arquitecturas de deep learning especializadas en datos con estructura de cuadrícula, como las imágenes (Goodfellow et al., 2016). La referencia a Goodfellow et al. (2016) proporciona la fundamentación matemática formal de las operaciones de las CNN que se emplean en este trabajo. Su poder radica en tres operaciones fundamentales:

**Convolución (Conv2D).** Aplica filtros (kernels) aprendibles que detectan patrones locales. Formalmente, la operación de convolución para una posición $(i, j)$ del mapa de salida se define como:

$$\text{Output}(i,j) = \sum_{m,n} \text{Input}(i+m, j+n) \times \text{Kernel}(m,n) + b \tag{1}$$

donde $m$ y $n$ recorren las dimensiones del kernel y $b$ es el sesgo. Los filtros de las primeras capas aprenden bordes y texturas simples, mientras que los de capas profundas aprenden patrones de alto nivel como anomalías térmicas y composiciones mineralógicas.

**Pooling (MaxPooling2D).** Reduce las dimensiones espaciales conservando las características más relevantes. Para cada ventana de, por ejemplo, 2 × 2 píxeles, selecciona el valor máximo, lo que reduce la resolución a la mitad y aporta invariancia a traslaciones menores.

**Activación (ReLU).** Introduce no linealidad mediante la función:

$$f(x) = \max(0, x) \tag{2}$$

que permite al modelo aprender relaciones complejas entre las variables de entrada. Sin funciones de activación no lineales, una red de múltiples capas sería equivalente a una sola transformación lineal, independientemente de su profundidad.

#### 5.3.3 Parámetros del modelo

Un parámetro es un valor numérico almacenado dentro del modelo que se ajusta durante el entrenamiento. Cada parámetro es un peso ($w$) que amplifica o atenúa una señal, o un sesgo ($b$) que la desplaza:

$$\text{salida} = \text{entrada} \times w + b \tag{3}$$

El modelo utilizado contiene 4.396.112 parámetros distribuidos en tres componentes:

**Tabla 6**

*Distribución de parámetros del modelo*

| Componente | Parámetros | Porcentaje |
|---|:-:|:-:|
| Backbone EfficientNetB0 | ~4.050.000 | 92,1 % |
| Channel Adapter (7 → 3 canales) | ~1.400 | 0,03 % |
| Classification Head (256 → 64 → 1) | ~344.000 | 7,8 % |
| **Total** | **4.396.112** | 100 % |

Para ilustrar cómo se acumulan estos valores: en el Channel Adapter, la primera capa convolucional tiene 16 filtros de tamaño 3 × 3 aplicados a 7 bandas, lo que produce $7 \times 16 \times 9 = 1.008$ pesos. En el backbone, una capa que transforma 40 canales en 80 con filtros 3 × 3 genera $3 \times 3 \times 40 \times 80 = 28.800$ pesos. En el Classification Head, una capa Dense(256) que recibe 1.280 valores tiene $1.280 \times 256 = 327.680$ pesos. El entrenamiento ajusta estos 4,4 millones de valores iterativamente hasta que el conjunto produce predicciones correctas; nadie programa las reglas de clasificación explícitamente — estas emergen del proceso de optimización.

#### 5.3.4 Funciones de activación de salida: sigmoid vs. softmax

Para la capa de salida de un clasificador existen dos funciones principales. **Softmax** se emplea en clasificación multiclase (por ejemplo, reconocer un dígito entre 0 y 9), distribuyendo la probabilidad entre todas las clases de modo que sumen 1,0. **Sigmoid**, en cambio, se emplea en clasificación binaria, transformando cualquier valor real en una probabilidad independiente entre 0 y 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}} \tag{4}$$

El problema abordado es binario — zona geotérmica o no —, por lo que la capa final tiene una sola neurona con sigmoid. Una sola neurona sigmoid es matemáticamente equivalente a dos neuronas softmax, pero más eficiente al usar la mitad de parámetros.

#### 5.3.5 Función de pérdida: Binary Cross-Entropy

La función de pérdida empleada fue **Binary Cross-Entropy (BCE)**, que cuantifica la diferencia entre la predicción del modelo y la etiqueta real:

$$\mathcal{L} = -\left[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right] \tag{5}$$

donde $y$ es la etiqueta real (1 si la zona es geotérmica, 0 si no lo es) y $\hat{y}$ es la probabilidad predicha por el modelo. La penalización crece exponencialmente con la confianza del error, lo que fuerza al modelo a ser honesto con su incertidumbre.


### 5.4 Transfer learning y EfficientNet

El transfer learning consiste en reutilizar los pesos de un modelo preentrenado en un dominio fuente y adaptarlos al dominio objetivo (Pan & Yang, 2010). Las primeras capas de una CNN aprenden características genéricas (bordes, texturas) que son transferibles entre dominios visuales distintos, mientras que las capas superiores se especializan en el dominio específico. Esta revisión seminal sobre transfer learning fundamenta la estrategia de entrenamiento en dos fases adoptada en el presente proyecto.

La estrategia de entrenamiento consta de dos fases: en la primera, el backbone se mantiene congelado y solo se entrenan el módulo adaptador y la cabeza de clasificación; en la segunda (fine-tuning), se descongelan las últimas capas del backbone con un learning rate reducido, lo que permite refinar las representaciones sin destruir el conocimiento previamente adquirido.

#### 5.4.1 Arquitectura EfficientNet

Tan y Le (2019) propusieron la familia de arquitecturas EfficientNet, cuya innovación central consiste en el escalado compuesto (compound scaling): escalar simultáneamente la profundidad, el ancho y la resolución de la red con una proporción matemática fija para maximizar el rendimiento con el mínimo número de parámetros. Esta arquitectura constituye el backbone del modelo propuesto en el presente proyecto.

**Tabla 7**

*Dimensiones de escalado en EfficientNet*

| Dimensión | Descripción | Efecto |
|-----------|-------------|--------|
| Profundidad | Más capas en la red | Mayor capacidad de abstracción |
| Ancho | Más filtros por capa | Mayor riqueza de representaciones |
| Resolución | Imágenes de entrada más grandes | Mayor detalle espacial |

EfficientNetB0, la variante base de la familia, posee 237 capas internas y aproximadamente 4,0 millones de parámetros (frente a los 25 millones de ResNet50 con peor rendimiento en ImageNet). La familia se escala desde B0 (más pequeña) hasta B7 (más grande), y fue preentrenada en ImageNet (1,2 millones de imágenes, 1.000 clases). Su entrada esperada es de 224 × 224 × 3 canales (RGB), lo que genera la necesidad del módulo Channel Adapter para proyectar las siete bandas ASTER.

#### 5.4.2 Bloque MBConv y Squeeze-and-Excitation

El bloque fundamental de EfficientNetB0 es el MBConv (Mobile Inverted Bottleneck), que incorpora tres mecanismos clave:

**Convoluciones depthwise-separable.** En lugar de aplicar un filtro 3 × 3 a todos los canales simultáneamente (operación costosa), primero se aplica un filtro independiente por cada canal (*depthwise*) y luego se combinan los resultados mediante filtros 1 × 1 (*pointwise*). El resultado es equivalente al de una convolución estándar, pero con aproximadamente ocho veces menos operaciones.

**Squeeze-and-Excitation (SE).** Es un mecanismo de atención por canal. Primero comprime la información espacial calculando un promedio global por canal (*squeeze*); luego genera pesos entre 0 y 1 para cada canal mediante dos capas densas (*excitation*), y multiplica cada canal por su peso respectivo. Este mecanismo permite que la red preste mayor atención a las bandas espectrales que más aportan en cada contexto — por ejemplo, priorizando las bandas TIR cuando detecta anomalías térmicas.

**Conexiones residuales internas.** He et al. (2016) introdujeron las conexiones residuales (skip connections) en las redes ResNet, las cuales fueron adoptadas internamente por EfficientNet. En un bloque residual, se define:

$$y = F(x, \{W_i\}) + x \tag{6}$$

donde $F$ es la transformación del camino principal y $x$ es la entrada transmitida por el atajo. Estas conexiones residuales son un componente interno de la arquitectura EfficientNetB0 y permiten entrenar capas profundas sin degradación del gradiente.

#### 5.4.3 Channel Adapter

El Channel Adapter es un módulo convolucional diseñado específicamente para proyectar las siete bandas espectrales ASTER al espacio de tres canales esperado por EfficientNetB0 (preentrenado en imágenes RGB). Consta de dos capas secuenciales: una capa Conv2D(16, 3 × 3) con BatchNormalization y ReLU que expande las siete bandas a 16 mapas intermedios, seguida de una capa Conv2D(3, 1 × 1) con BatchNormalization y ReLU que proyecta a tres canales. Con apenas 1.400 parámetros (0,03 % del total), este módulo aprende la combinación óptima del espacio espectral ASTER, en lugar de requerir una selección manual de bandas.


### 5.5 Técnicas de regularización y optimización

En el entrenamiento de redes neuronales profundas, la regularización previene el sobreajuste (overfitting) — la tendencia del modelo a memorizar los datos de entrenamiento en lugar de aprender patrones generalizables. El presente proyecto emplea siete técnicas complementarias:

**Dropout** (Goodfellow et al., 2016): durante cada paso del entrenamiento, desactiva aleatoriamente un porcentaje de neuronas, obligando a la red a no depender de ninguna neurona individual. Se utilizaron tasas de 0,5 en la primera capa densa y 0,3 en la segunda.

**Batch Normalization**: normaliza las activaciones de cada capa a media cero y varianza unitaria, lo que estabiliza y acelera la convergencia del entrenamiento.

**Label Smoothing**: suaviza las etiquetas duras de $(0; 1)$ a $(0{,}05; 0{,}95)$ con $\varepsilon = 0{,}1$, lo que previene que el modelo se vuelva excesivamente confiado en sus predicciones.

**Weight Decay (AdamW)**: aplica una penalización L2 desacoplada sobre los pesos ($\lambda = 1 \times 10^{-3}$), reduciendo su magnitud para evitar soluciones complejas innecesarias.

**MixUp**: genera ejemplos virtuales de entrenamiento interpolando pares de muestras y sus etiquetas:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j \tag{7}$$

donde $\lambda \sim \text{Beta}(\alpha, \alpha)$ con $\alpha = 0{,}2$. Esta técnica suaviza la frontera de decisión y mejora la calibración del modelo.

**CosineDecay**: el learning rate decae siguiendo una curva coseno a lo largo del entrenamiento, permitiendo pasos grandes al inicio (exploración) y pasos finos al final (convergencia).

**EarlyStopping**: detiene el entrenamiento cuando la métrica de validación (val_auc) no mejora durante un número determinado de épocas consecutivas (patience = 10 en la Fase 1, 15 en la Fase 2).

**Global Average Pooling.** En lugar de aplanar (Flatten) la salida del backbone — lo que generaría millones de parámetros —, se calcula el promedio por canal: el mapa de 7 × 7 × 1.280 se reduce a un vector de 1.280 valores. Esto reduce drásticamente la cantidad de parámetros y es menos propenso al sobreajuste.


### 5.6 Herramientas tecnológicas

#### 5.6.1 TensorFlow y Keras

TensorFlow (Abadi et al., 2016) es la plataforma de código abierto desarrollada por Google para computación numérica a gran escala. Opera como el motor de bajo nivel que gestiona las operaciones matemáticas sobre tensores, la distribución del cómputo en la GPU, el cálculo automático de gradientes (autodiferenciación) y la precisión mixta float16.

Keras, integrado en TensorFlow desde la versión 2.x, es la interfaz de alto nivel que permite definir arquitecturas, entrenar modelos y realizar predicciones con instrucciones legibles. En el presente proyecto se emplearon TensorFlow 2.20 y Keras 3.12/3.13, dado que esta plataforma gestiona todos los cálculos del modelo.

#### 5.6.2 CUDA y aceleración por GPU

El entrenamiento de modelos de deep learning es una tarea masivamente paralela: cada píxel de cada imagen y cada neurona de la red pueden procesarse de forma independiente. Mientras que una CPU moderna posee entre 8 y 16 núcleos de propósito general, una GPU como la NVIDIA RTX 4070 utilizada en este proyecto dispone de 5.888 núcleos CUDA (Compute Unified Device Architecture) que ejecutan operaciones en paralelo. Esto permite procesar los 11,2 millones de valores de un batch de 32 imágenes simultáneamente, alcanzando una velocidad 45 veces superior a la de la CPU (347 ms por step frente a ~15.600 ms).

#### 5.6.3 Mixed Precision Training

La precisión mixta (float16) utiliza aritmética de 16 bits para las operaciones de propagación hacia adelante y hacia atrás, y 32 bits para la acumulación de gradientes. Al combinar esta técnica con los Tensor Cores especializados de las GPU modernas, se duplica el throughput y se reduce el consumo de memoria de video (VRAM) en aproximadamente un 50 %, sin pérdida significativa de precisión gracias al escalado automático del loss.


### 5.7 Estado del arte

La aplicación de técnicas de aprendizaje automático a la exploración geotérmica ha avanzado en los últimos años, aunque los trabajos que combinan CNN con imágenes multiespectrales de emisividad térmica son escasos. Coolbaugh et al. (2007) utilizaron modelos de regresión logística con variables geoespaciales para mapear potencial geotérmico en Nevada (Estados Unidos), alcanzando resultados moderados, lo que evidencia que los enfoques estadísticos tradicionales no explotaban la capacidad de las CNN para extraer características de forma automática.

Más recientemente, estudios en teledetección han aplicado CNN a la clasificación de uso del suelo y la detección de cambios en imágenes satelitales (Zhu et al., 2017). Según estos autores, "the use of deep neural networks for remote sensing image classification has produced state-of-the-art results on various benchmark datasets" (p. 12), lo cual demuestra la madurez del deep learning en tareas de teledetección directamente relacionadas con el presente proyecto.

El proyecto TensorFlow de Google lanzó en 2015 una plataforma de código abierto que democratizó el desarrollo de modelos de deep learning (Abadi et al., 2016). El marco Keras, integrado en TensorFlow desde la versión 2.x, proporciona una interfaz de alto nivel para la construcción de modelos. En el presente proyecto se utilizaron TensorFlow 2.20 y Keras 3.12 como stack tecnológico de entrenamiento y despliegue, dado que esta plataforma gestiona todos los cálculos del modelo.

---

<!-- ============================================================
     6. METODOLOGÍA
     ============================================================ -->

## 6 Metodología

### 6.1 Enfoque y tipo de investigación

La investigación se enmarca en un enfoque cuantitativo de tipo experimental-aplicado. Se diseñó un experimento controlado en el que se entrenó un modelo de clasificación binaria sobre un dataset construido específicamente para el problema, evaluando su desempeño mediante métricas estandarizadas sobre un conjunto de prueba independiente.


### 6.2 Dataset: Región Andina

El dataset se construyó a partir de 2.019 zonas georreferenciadas distribuidas en cuatro países andinos:

**Tabla 8**

*Composición del dataset v3 por país*

| País | Positivas | Negativas | Total |
|------|:---------:|:---------:|:-----:|
| Colombia | 544 | 477 | 1.021 |
| Ecuador | 184 | 66 | 250 |
| Perú | 107 | 28 | 135 |
| Chile | 162 | 451 | 613 |
| **Total** | **997** | **1.022** | **2.019** |

Cada zona base se expandió en una grilla de nueve tiles (centro, norte, sur, este, oeste, noreste, noroeste, sureste, suroeste) desplazados ±0,045° (~4 km) para maximizar la cobertura espacial.

**Criterios de selección de zonas positivas (label = 1):** campo geotérmico confirmado por el servicio geológico nacional, volcán activo con fumarolas (actividad en el Holoceno, inferior a 11.700 años), manifestación hidrotermal con temperatura superior a 40 °C, buffer de 5 km por zona con subdivisiones 3 × 3.

**Criterios de selección de zonas negativas (label = 0):** ausencia de vulcanismo en un radio superior a 50 km, estabilidad geológica (cuencas sedimentarias, llanuras costeras), diversidad geomorfológica (costa, llanura, selva, altiplano) y distancia mínima superior a 20 km entre puntos.

Se aplicaron diez técnicas de aumento de datos offline por imagen, siguiendo las recomendaciones de Shorten y Khoshgoftaar (2019): rotación 90°, 180° y 270°, volteo horizontal y vertical, trasposición, ajuste de brillo (±15 %), ajuste de contraste (±15 %), ajuste de gamma y adición de ruido gaussiano. Esta selección se redujo de 30 transformaciones en la versión anterior a 10, siguiendo las recomendaciones de la revisión citada para evitar redundancia entre las variantes generadas.

**Tabla 9**

*Resultado del aumento de datos*

| Condición | Imágenes |
|-----------|:--------:|
| Originales | 2.019 |
| Después del aumento | 22.209 (11.016 positivas / 11.193 negativas) |

La división del dataset se realizó con GroupShuffleSplit, agrupando por zona geográfica base (eliminando sufijos de augmentación y grilla), de modo que todas las variantes de una misma zona geográfica quedaran en el mismo subconjunto:

**Tabla 10**

*División anti-fuga geográfica del dataset*

| Subconjunto | Imágenes | Zonas base | Proporción |
|-------------|:--------:|:----------:|:----------:|
| Entrenamiento | 15.037 | 284 | 67,7 % |
| Validación | 3.553 | 61 | 16,0 % |
| Prueba | 3.619 | 62 | 16,3 % |
| **Total** | **22.209** | **407** | 100 % |

El solapamiento de zonas entre subconjuntos fue de cero, verificado por scripts independientes. Esta estrategia es fundamental para prevenir la fuga de datos (data leakage), un problema que invalidó los resultados de las versiones anteriores del modelo (ver sección 8.3).

Las imágenes se descargaron desde Google Earth Engine con tres hilos concurrentes y un retardo de 0,5 segundos entre solicitudes (respeto a las cuotas de la API). Cada imagen consiste en un buffer de 5 km a una resolución de 90 metros por píxel, lo que produce tiles de aproximadamente 111 × 111 × 7 píxeles en formato GeoTIFF.


### 6.3 Pipeline de procesamiento

El pipeline de procesamiento sigue la secuencia descrita a continuación:

1. **Descarga:** Imágenes ASTER GED AG100 v003 desde Google Earth Engine → 2.019 archivos .tif (7 bandas, ~115 MB).
2. **Aumento de datos:** 10 técnicas offline → 22.209 imágenes (~7,9 GB).
3. **Filtrado de NoData:** Los datos ASTER usan -9999 como indicador de ausencia. Si más del 50 % de una banda presenta NoData, la imagen se descarta. Los valores restantes se reemplazan por la mediana de valores válidos de la misma banda.
4. **Redimensionamiento:** De ~111 × 111 píxeles a 224 × 224 píxeles mediante interpolación bicúbica con anti-aliasing. El tamaño 224 × 224 es estándar en deep learning y compatible con EfficientNetB0.
5. **Normalización z-score global:** Cada banda se normaliza con la media y desviación estándar globales calculadas sobre todo el dataset mediante el algoritmo de Welford (online, complejidad O(1) en RAM). Las estadísticas se almacenan en band_stats_v3.json para reutilizar en inferencia.

$$x_{\text{norm}} = \frac{x - \mu_{\text{banda}}^{\text{global}}}{\sigma_{\text{banda}}^{\text{global}}} \tag{8}$$

6. **Particionado:** Almacenamiento en archivos .npy de ~500 imágenes cada uno (~29,8 GB en total).
7. **Entrenamiento → modelo .keras (60 MB).**
8. **Evaluación → métricas en evaluation_metrics.json.**
9. **Despliegue → interfaz web Streamlit para predicción en tiempo real.**

Es importante señalar que la versión 1 y los primeros intentos de la versión 3 normalizaban por imagen individual, lo cual destruía toda información absoluta entre imágenes (una zona a 80 °C y otra a 20 °C quedaban estadísticamente idénticas). Este fue el bug crítico número 5 que invalidó dichos resultados (ver sección 8.3).


### 6.4 Arquitectura del modelo

El modelo final (denominado internamente GeotermiaCNN_V7) consiste en tres componentes:

**Channel Adapter:** Módulo convolucional que proyecta las siete bandas ASTER al espacio de tres canales esperado por EfficientNetB0. Consta de una capa Conv2D(16, 3 × 3) con BatchNorm y ReLU que expande las siete bandas a 16 mapas intermedios, seguida de una capa Conv2D(3, 1 × 1) con BatchNorm y ReLU que proyecta a tres canales. Este adapter aprende la proyección óptima del espacio espectral ASTER, en lugar de seleccionar o promediar bandas de forma manual.

**Backbone EfficientNetB0:** Red neuronal con pesos preentrenados en ImageNet (1,2 millones de imágenes, 1.000 clases). Procesa la imagen de 224 × 224 × 3 canales y produce un mapa de características de 7 × 7 × 1.280.

**Classification Head:** Recibe las 1.280 características tras un Global Average Pooling, y las procesa mediante Dense(256) + BatchNorm + ReLU + Dropout(0,5), Dense(64) + BatchNorm + ReLU + Dropout(0,3), y Dense(1, sigmoid) que produce la probabilidad final entre 0 y 1.

**Figura 1**

*Arquitectura del modelo EfficientNetB0 + Channel Adapter*

```
INPUT (224 × 224 × 7)
        │
        ▼
CHANNEL ADAPTER
  Conv2D(16, 3 × 3) + BN + ReLU
  Conv2D(3, 1 × 1) + BN + ReLU
  → (224 × 224 × 3)
        │
        ▼
BACKBONE EfficientNetB0
  237 capas, bloques MBConv + Squeeze-and-Excitation
  Pesos ImageNet
  → (7 × 7 × 1280)
        │
        ▼
GLOBAL AVERAGE POOLING → 1280
        │
        ▼
CLASSIFICATION HEAD
  Dense(256) + BN + ReLU + Dropout(0,5)
  Dense(64) + BN + ReLU + Dropout(0,3)
  Dense(1, sigmoid) → Probabilidad [0, 1]

Total parámetros: 4.396.112
```

*Nota.* El Channel Adapter representa el 0,03 % de los parámetros totales, el backbone el 92,1 % y el classification head el 7,8 %.


### 6.5 Estrategia de entrenamiento

El entrenamiento siguió una estrategia de dos fases, estándar en transfer learning:

**Tabla 11**

*Hiperparámetros de la Fase 1 — Backbone congelado (30 épocas)*

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

**Tabla 12**

*Hiperparámetros de la Fase 2 — Fine-tuning (50 épocas)*

| Parámetro | Valor |
|-----------|-------|
| Capas descongeladas | Últimas 39 capas del backbone |
| BatchNormalization | Congelado |
| Optimizador | AdamW (weight_decay = 1 × 10⁻³) |
| Learning rate | 1 × 10⁻⁴ con CosineDecay (10× menor que Fase 1) |
| Batch size | 32 |
| MixUp | α = 0,2 |
| Label smoothing | 0,1 |
| EarlyStopping | patience = 15, monitor = val_auc |
| Mejor época | 50 (val_auc = 0,9725) |

La razón de entrenar en dos fases es evitar la destrucción del conocimiento preentrenado: el dataset de entrenamiento (22.209 imágenes) es aproximadamente 550 veces más pequeño que ImageNet (1,2 millones), por lo que modificar todos los parámetros desde el inicio arriesga sobrescribir las representaciones visuales genéricas con ruido del dominio específico.

En la Fase 2, las capas de Batch Normalization se mantuvieron congeladas (inference mode) para preservar las estadísticas de media y varianza acumuladas durante el preentrenamiento con ImageNet. Recalcular estas estadísticas con solo 32 imágenes por batch habría desestabilizado la normalización interna de la red.

**Tabla 13**

*Técnicas de regularización empleadas*

| Técnica | Descripción | Parámetro en v3 |
|---------|-------------|:---------------:|
| Dropout | Desactiva aleatoriamente neuronas durante el entrenamiento | 0,3 y 0,5 |
| Batch Normalization | Normaliza activaciones internas, estabiliza convergencia | Todas las capas |
| Label Smoothing | Suaviza etiquetas duras: (0,05; 0,95) en vez de (0; 1) | ε = 0,1 |
| Weight Decay (AdamW) | Penalización L2 desacoplada sobre los pesos | 1 × 10⁻³ |
| MixUp | Interpola pares de muestras y etiquetas | α = 0,2 |
| CosineDecay | Learning rate decae suavemente siguiendo una curva coseno | Hasta 1 × 10⁻⁴ |
| EarlyStopping | Detiene el entrenamiento si val_auc no mejora | patience = 10/15 |

El entrenamiento se ejecutó en GPU NVIDIA RTX 4070 (12 GB VRAM) bajo WSL2 Ubuntu 22.04, con precisión mixta float16 que duplica el throughput y reduce el consumo de VRAM en un 50 %. La velocidad alcanzada fue de 347 milisegundos por step, 45 veces más rápida que en CPU.


### 6.6 Métricas de evaluación

Para evaluar el desempeño del modelo se emplean las siguientes métricas:

- **Exactitud (Accuracy):** Proporción de predicciones correctas sobre el total de muestras:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \tag{10}$$

- **Precisión (Precision):** Proporción de predicciones positivas que son verdaderos positivos. Responde a la pregunta: de las zonas que el modelo marcó como geotérmicas, ¿cuántas realmente lo son?

$$\text{Precision} = \frac{TP}{TP + FP} \tag{11}$$

- **Sensibilidad (Recall):** Proporción de positivos reales correctamente identificados. Responde a la pregunta: de las zonas que realmente son geotérmicas, ¿cuántas detectó el modelo?

$$\text{Recall} = \frac{TP}{TP + FN} \tag{12}$$

- **F1-Score:** Media armónica de precisión y sensibilidad:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \tag{13}$$

- **AUC-ROC:** Área bajo la curva ROC (Receiver Operating Characteristic). Mide la capacidad discriminativa del modelo en todos los umbrales posibles. Un valor de 1,0 indica discriminación perfecta; 0,5 indica que el modelo no discrimina mejor que el azar.

- **MCC (Matthews Correlation Coefficient):** Coeficiente de correlación que considera los cuatro cuadrantes de la matriz de confusión, entre −1 y +1:

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \tag{14}$$

Para el contexto de screening geotérmico, la sensibilidad (recall) es la métrica más crítica porque el costo de un falso negativo (no identificar una zona geotérmica real) es sustancialmente mayor que el de un falso positivo (marcar erróneamente una zona no geotérmica), dado que la exploración de campo posterior descartaría los falsos positivos.


### 6.7 Intervalos de confianza Bootstrap

Los intervalos de confianza Bootstrap permiten estimar la variabilidad de las métricas de evaluación sin supuestos paramétricos sobre la distribución de los datos (Efron y Tibshirani, 1993). Dado un conjunto de prueba $\mathcal{D}$ de $n$ observaciones, para cada iteración $b = 1, \ldots, B$ se obtiene una muestra $\mathcal{D}_b^*$ de tamaño $n$ con reemplazo, y se calcula $\hat{\theta}_b^* = T(\mathcal{D}_b^*)$ donde $T$ es el estadístico de interés (por ejemplo, accuracy o F1). El intervalo de confianza al $100(1-\alpha)\%$ se define como:

$$\text{IC}_{1-\alpha} = \left[\hat{\theta}^*_{(\alpha/2)},\; \hat{\theta}^*_{(1-\alpha/2)}\right] \tag{15}$$

En este proyecto se utilizan $B = 2\,000$ iteraciones para obtener intervalos al 95 % sobre las seis métricas principales (Accuracy, Precision, Recall, F1, ROC AUC, MCC), lo cual proporciona una estimación robusta de la incertidumbre del modelo.

---

<!-- ============================================================
     7. RESULTADOS
     ============================================================ -->

## 7 Resultados

### 7.1 Métricas del modelo v3

El modelo se evaluó sobre el conjunto de prueba compuesto por 3.619 imágenes provenientes de los cuatro países:

**Tabla 14**

*Métricas del modelo v3 sobre el conjunto de prueba (3.619 imágenes)*

| Métrica | Valor | Objetivo mínimo | Objetivo ideal | Resultado |
|---------|:-----:|:---------------:|:--------------:|:---------:|
| Exactitud | 92,28 % | > 85 % | > 90 % | Superado |
| Precisión | 91,27 % | > 80 % | > 85 % | Superado |
| Sensibilidad | 93,17 % | > 80 % | > 85 % | Superado |
| F1-Score | 92,21 % | > 80 % | > 85 % | Superado |
| ROC AUC | 0,9737 | > 0,90 | > 0,95 | Superado |
| PR AUC | 0,9693 | — | — | — |
| MCC | 0,8458 | > 0,50 | > 0,70 | Superado |

Todas las métricas superaron los objetivos ideales establecidos.


### 7.2 Matriz de confusión

**Tabla 15**

*Matriz de confusión del modelo v3*

| | Predicho negativo | Predicho positivo |
|---|:-:|:-:|
| **Real negativo** | 1.686 (VN) | 158 (FP) |
| **Real positivo** | 121 (FN) | 1.651 (VP) |

- Especificidad: 91,43 % (1.686 de 1.844 negativos correctos).
- Sensibilidad: 93,17 % (1.651 de 1.772 positivos correctos).
- Tasa de falsos positivos: 8,57 %, mayor que la v2 (2,15 %) debido a un dataset más diverso.
- Tasa de falsos negativos: 6,83 %, mejora significativa respecto a la v2 (13,95 %).


### 7.3 Comparativo v1 → v2 → v3

**Tabla 16**

*Comparativo de métricas entre versiones del modelo*

| Métrica | v1 (baseline) | v2 (ResNet) | v3 (EfficientNetB0) |
|---------|:---:|:---:|:---:|
| Exactitud | 68,43 % | 91,45 % | **92,28 %** |
| Precisión | 86,32 % | 97,94 % | **91,27 %** |
| Sensibilidad | 48,10 % | 86,05 % | **93,17 %** |
| F1-Score | 61,77 % | 91,61 % | **92,21 %** |
| ROC AUC | 0,8198 | 0,983 | **0,9737** |
| MCC | −0,2673 | 0,837 | **0,8458** |
| Test set | 396 imgs (1 país) | 1.017 imgs (1 país) | **3.619 imgs (4 países)** |
| Fuga de datos | Sí | 62,5 % | **0 %** |

La precisión de la v3 es menor que la de la v2 (91,27 % frente a 97,94 %), lo cual se explica por el hecho de que el test set es 3,6 veces mayor y geográficamente más diverso. El AUC-ROC de la v3 (0,9737) es numéricamente inferior al de la v2 (0,983), pero esta comparación es engañosa porque los resultados de la v2 estaban inflados por una fuga de datos del 62,5 % que fue descubierta y corregida en la v3.


### 7.4 Curvas de entrenamiento

En la **Fase 1** (backbone congelado, épocas 1 a 30), el modelo partió de un AUC cercano a 0,50 (equivalente al azar) y progresó hasta val_auc = 0,9000 en la época 27, demostrando que el Channel Adapter y la cabeza de clasificación aprendieron a interpretar las bandas ASTER.

En la **Fase 2** (fine-tuning de las últimas 39 capas, épocas 31 a 80), el AUC mejoró de forma gradual desde 0,92 hasta val_auc = 0,9725 en la época 50. El EarlyStopping detuvo el entrenamiento en la época 57 al no registrar mejoras durante 15 épocas consecutivas.

**Figura 3**

*Curvas de entrenamiento — evolución del AUC en las fases 1 y 2*

```
AUC
1.00 ┤                                          ●●●●●●●──── 0.9725
     │                                     ●●●●
0.95 ┤                                  ●●●
     │                               ●●●
0.90 ┤                      ●●●●●────●         ← Transición F1→F2
     │                  ●●●●
0.85 ┤              ●●●●
     │          ●●●●
0.80 ┤       ●●●
     │     ●●
0.75 ┤    ●
     │   ●
0.70 ┤  ●
     │  ●
0.65 ┤ ●
     │ ●
0.60 ┤●
     │●
0.55 ┤●    ← Inicio desde azar
0.50 ┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
     1  5  10 15 20 25 30 35 40 45 50 55 60 65 70 75  Época
     ├──────── Fase 1 ────────┤├──────── Fase 2 ────────────┤
         LR = 0,001               LR = 0,0001
```

*Nota.* La línea vertical marca la transición entre fases. El salto inicial en la Fase 2 refleja el beneficio inmediato de descongelar capas del backbone. El EarlyStopping actuó en la época 57.


### 7.5 Contraste de hipótesis

Con una exactitud del 92,28 % sobre un conjunto de prueba de 3.619 imágenes y un AUC-ROC de 0,9737, se rechaza la hipótesis nula ($H_0: \text{Accuracy} \leq 0{,}50$) con amplio margen. Dado que el intervalo de confianza Bootstrap al 95 % para la exactitud se sitúa muy por encima del umbral del 50 %, la probabilidad de que el modelo no supere el azar es virtualmente nula. El modelo CNN demuestra una capacidad discriminativa significativamente superior al azar para la clasificación binaria de zonas con y sin potencial geotérmico, confirmando la hipótesis alternativa ($H_1$).

---

<!-- ============================================================
     8. DISCUSIÓN
     ============================================================ -->

## 8 Discusión

### 8.1 Interpretación de resultados

Los resultados obtenidos demuestran que un modelo CNN basado en EfficientNetB0 con transfer learning puede identificar zonas con potencial geotérmico a partir de imágenes ASTER con un desempeño significativamente superior al azar. Con una exactitud del 92,28 % y un AUC-ROC de 0,9737, se rechaza la hipótesis nula ($H_0: \text{Accuracy} \leq 0{,}50$).

La sensibilidad del 93,17 % es particularmente relevante para el caso de uso propuesto (screening geotérmico), puesto que indica que el modelo identifica correctamente más de 9 de cada 10 zonas geotérmicas reales. La tasa de falsos negativos del 6,83 % representa una mejora sustancial sobre la v2 (13,95 %), lo que se traduce en un menor riesgo de pasar por alto zonas con potencial explotable.

La tasa de falsos positivos del 8,57 % es aceptable en un contexto de reconocimiento regional, dado que las fases posteriores de exploración (más costosas) se encargarían de descartar estas zonas. En la práctica, el costo de investigar un falso positivo en campo es mucho menor que el costo de oportunidad de no investigar un falso negativo.

La efectividad del transfer learning se evidencia en que un modelo con solo 4,4 millones de parámetros, preentrenado en imágenes naturales de tres canales RGB, fue capaz de alcanzar un AUC-ROC cercano a 0,97 en un dominio radicalmente diferente (emisividad térmica de siete canales). Esto respalda la hipótesis de que las representaciones visuales genéricas aprendidas en ImageNet (bordes, texturas, contrastes) son transferibles a imágenes de teledetección, como sugieren Pan y Yang (2010), cuyos hallazgos teóricos encuentran en estos resultados evidencia empírica concreta.


### 8.2 Comparación con trabajos relacionados

No existe un benchmark directo para la clasificación binaria de potencial geotérmico con CNN en Colombia, pero los resultados son consistentes con el estado del arte en aplicaciones de aprendizaje profundo a imágenes satelitales. Zhu et al. (2017) reportan exactitudes superiores al 85 % en tareas de clasificación comparables. Mia et al. (2018) obtuvieron resultados satisfactorios con técnicas de Machine Learning para identificar alteraciones hidrotermales con datos ASTER, aunque emplearon métodos clásicos (Random Forest, SVM) en lugar de CNN profundas.

La transición a Transfer Learning con EfficientNetB0 en la v3 demostró que las características aprendidas en ImageNet (bordes, texturas, patrones espaciales) son transferibles al dominio de emisividad térmica, incluso cuando el número de canales difiere (siete bandas ASTER frente a tres canales RGB). El Channel Adapter convolucional permitió aprender una proyección óptima del espacio espectral ASTER al espacio RGB, logrando resultados competitivos con menos parámetros (4,4 M frente a 5,0 M de la v2).


### 8.3 Bugs descubiertos y corregidos

El proceso de desarrollo incluyó dos auditorías exhaustivas que identificaron 34 bugs en total. La primera auditoría (v1 → v2) encontró 28 bugs, de los cuales 4 fueron clasificados como críticos:

**Tabla 17**

*Bugs críticos descubiertos en la auditoría v1 → v2*

| N.° | Bug | Impacto | Corrección |
|:---:|-----|---------|------------|
| 1 | NoData (-9999) no filtrado | Media y desviación estándar distorsionadas | Filtrado + interpolación mediana |
| 2 | Doble normalización (z-score + Rescaling 1/255) | Señal comprimida al rango [−0,008; +0,008] | Eliminada capa Rescaling |
| 3 | Data leakage (train_test_split sin agrupar) | Augmentaciones del mismo tile en train y test | GroupShuffleSplit por imagen |
| 4 | Doble regularización L2 (kernel_regularizer + AdamW) | Modelo sobre-regularizado | Solo AdamW weight_decay |

La segunda auditoría (v2 → v3) reveló 6 bugs adicionales, incluyendo dos de especial gravedad:

**Tabla 18**

*Bugs críticos descubiertos en la auditoría v2 → v3*

| N.° | Bug | Impacto | Corrección |
|:---:|-----|---------|------------|
| 5 | Normalización z-score por imagen | Destruye información absoluta (AUC ≈ 0,51) | Normalización global por banda (Welford) |
| 6 | 62,5 % de data leakage entre splits | V2 memorizaba augmentaciones, métricas infladas | GroupShuffleSplit por zona geográfica base |

El bug 5 (normalización per-image) resultó particularmente instructivo: al normalizar cada imagen individualmente, todas las zonas quedaban con media cero y desviación uno, independientemente de su temperatura real. El modelo no podía distinguir entre una zona a 80 °C y una a 20 °C, lo que produjo un AUC cercano al azar durante 11 épocas. La corrección (normalización global con estadísticas del dataset completo) resolvió el problema de forma inmediata.

El bug 6 (data leakage del 62,5 %) explicó por qué la v2 reportaba métricas aparentemente superiores (AUC = 0,983): las augmentaciones de una misma zona geográfica aparecían simultáneamente en los conjuntos de entrenamiento, validación y prueba, lo que permitía al modelo "memorizar" en lugar de generalizar. La corrección mediante GroupShuffleSplit por zona geográfica base (407 zonas, 0 % solapamiento) produjo métricas ligeramente inferiores pero genuinas.


### 8.4 Implicaciones prácticas

El modelo se posiciona como una herramienta de screening automatizado para la Fase 1 (Reconocimiento Regional) de la exploración geotérmica. Su capacidad para analizar cualquier punto del territorio colombiano en segundos contrasta con los meses y los costos significativos que implica la revisión manual de imágenes satelitales por expertos.

Para el Servicio Geológico Colombiano y la UPME, el sistema ofrece la posibilidad de generar mapas de probabilidades de potencial geotérmico a escala nacional, lo que permitiría enfocar los recursos limitados de exploración en las zonas más prometedoras. La tasa de falsos positivos del 8,57 % lo hace confiable: las zonas que identifica como positivas merecen atención prioritaria.

La inclusión de zonas de Ecuador, Perú y Chile en el conjunto de entrenamiento no busca predecir fuera de Colombia, sino exponer al modelo a una mayor variabilidad de patrones espectrales geotérmicos dentro de un contexto geológico compartido (Cinturón de Fuego del Pacífico), mejorando la robustez del entrenamiento.


### 8.5 Limitaciones

1. **Resolución temporal:** El producto ASTER GED es un promedio temporal que no captura variaciones estacionales en la actividad geotérmica.
2. **Validación en campo:** Las predicciones no han sido contrastadas con prospección geotérmica in situ. El modelo identifica patrones espectrales correlacionados con actividad geotérmica conocida, pero no confirma la existencia de un recurso explotable.
3. **Búsqueda de hiperparámetros:** Se realizó de forma manual. Una búsqueda bayesiana (por ejemplo, con Optuna) podría mejorar los resultados.
4. **Sesgo de etiquetado:** Las etiquetas se basan en la literatura geológica existente, lo que introduce un sesgo hacia zonas ya conocidas y exploradas.
5. **Generalización:** El modelo no ha sido evaluado fuera de la Región Andina.

---

<!-- ============================================================
     9. CONCLUSIONES
     ============================================================ -->

## 9 Conclusiones

1. Se desarrolló un modelo de clasificación binaria basado en EfficientNetB0 con Channel Adapter que identifica zonas con potencial geotérmico en Colombia a partir de imágenes satelitales ASTER de siete bandas. El modelo alcanzó una exactitud del 92,28 %, una sensibilidad del 93,17 % y un AUC-ROC de 0,9737 sobre un conjunto de prueba de 3.619 imágenes de cuatro países andinos, superando todos los objetivos planteados (exactitud > 85 %, AUC-ROC > 0,90).

2. Todas las métricas de evaluación superaron los umbrales definidos: exactitud 92,28 %, precisión 91,27 %, sensibilidad 93,17 %, F1-Score 92,21 %, ROC AUC 0,9737 y MCC 0,8458. La hipótesis nula fue rechazada con amplio margen.

3. Se construyó un dataset georreferenciado de 2.019 zonas (22.209 imágenes después del aumento de datos) de la Región Andina (Colombia, Ecuador, Perú y Chile), con una división anti-fuga geográfica de 407 zonas base independientes y cero por ciento de solapamiento entre los conjuntos, eliminando el problema de data leakage que invalidaba los resultados de las versiones anteriores.

4. La auditoría exhaustiva del código identificó y corrigió 34 errores a lo largo de tres versiones (4 críticos, 4 de alta severidad, 10 medios y 10 bajos), lo que resultó en una mejora de v1 a v2 de 23,02 puntos porcentuales en exactitud y 37,95 en sensibilidad.

5. Se descubrió y corrigió una fuga de datos del 62,5 % en las divisiones originales, causada por la no eliminación de sufijos de grilla en el GroupShuffleSplit. Este hallazgo demuestra la importancia crítica de auditar rigurosamente la integridad de los datos en proyectos de aprendizaje profundo.

6. La adopción de Transfer Learning con EfficientNetB0 preentrenado en ImageNet demostró que las características visuales genéricas (bordes, texturas, patrones espaciales) son transferibles al dominio de emisividad térmica, logrando resultados superiores con 12,6 % menos parámetros (4.396.112 frente a 5.032.385) y un entrenamiento eficiente en dos fases con GPU.

7. Se implementó un pipeline completo de procesamiento que incluye filtrado de valores NoData, redimensionamiento, normalización z-score global por banda y aumento de datos con diez transformaciones, estableciendo un protocolo reproducible para futuros estudios.

8. Se desarrolló una interfaz web interactiva con Streamlit y Folium que permite realizar predicciones de potencial geotérmico en tiempo real sobre cualquier coordenada del territorio colombiano, integrando mapas interactivos, capas satelitales y generación de reportes en PDF.

---

<!-- ============================================================
     10. RECOMENDACIONES
     ============================================================ -->

## 10 Recomendaciones

### 10.1 Recomendaciones cumplidas en la v3

1. **Ampliación del conjunto de datos:** Cumplida. La v3 expandió el conjunto de 200 a 2.019 imágenes base de la Región Andina (Colombia, Ecuador, Perú y Chile), aumentadas a 22.209 imágenes con 10 técnicas de aumento de datos.

2. **Acceso a GPU:** Cumplida. La v3 se entrenó en una NVIDIA RTX 4070 (12 GB VRAM) bajo WSL2 Ubuntu 22.04, habilitando Mixed Precision (float16) y un entrenamiento en dos fases de 80 épocas totales.

3. **Transfer Learning con EfficientNet:** Cumplida. La v3 implementó Transfer Learning con EfficientNetB0 preentrenado en ImageNet, combinado con un Channel Adapter convolucional para las 7 bandas ASTER.

4. **Técnicas avanzadas de aumento de datos:** Parcialmente cumplida. Se implementó MixUp ($\alpha = 0{,}2$) como técnica de regularización por interpolación de muestras. CutMix queda pendiente.

### 10.2 Recomendaciones pendientes para futuras iteraciones

5. **Validación en campo:** Contrastar las predicciones del modelo con datos de prospección geotérmica in situ en al menos 5 a 10 zonas clasificadas como positivas por el modelo pero no documentadas previamente, en colaboración con el Servicio Geológico Colombiano.

6. **Incorporación de bandas SWIR:** El sensor ASTER dispone de 6 bandas en el infrarrojo de onda corta que contienen información sobre alteraciones hidrotermales minerales. Su incorporación (13 bandas totales) podría mejorar la discriminación del modelo.

7. **Interpretabilidad:** Implementar técnicas como Grad-CAM para visualizar qué regiones y bandas espectrales de las imágenes son más relevantes para las predicciones del modelo.

8. **Búsqueda sistemática de hiperparámetros:** Realizar búsqueda automatizada (Bayesian optimization, Optuna) de la tasa de aprendizaje, arquitectura del adapter, número de capas descongeladas y factor de MixUp.

9. **Expansión a otros contextos tectónicos:** Incorporar datos de zonas geotérmicas fuera de la Región Andina (Centroamérica, Indonesia, Islandia, Rift de África Oriental) para evaluar la transferibilidad global del modelo.

10. **Cobertura nacional sistemática:** Utilizar el modelo para generar un mapa completo de probabilidades de potencial geotérmico a escala nacional, procesando imágenes ASTER de forma sistemática sobre una cuadrícula que cubra todo el territorio colombiano.

---

<!-- ============================================================
     REFERENCIAS
     ============================================================ -->

## Referencias

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Wattenberg, P., … Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 265–283.

Abrams, M., Tsu, H., Hulley, G., Iwao, K., Pieri, D., Cudahy, T., & Kargel, J. (2015). The Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) after fifteen years: Review of global products. *International Journal of Applied Earth Observation and Geoinformation*, *38*, 292–301. https://doi.org/10.1016/j.jag.2015.01.013

Coolbaugh, M. F., Raines, G. L., Zehner, R. E., Shevenell, L., & Williams, C. F. (2007). Prediction and discovery of new geothermal resources in the Great Basin: Multiple evidence of a large undiscovered resource base. *Geothermal Resources Council Transactions*, *31*, 1–5.

Dickson, M. H., & Fanelli, M. (2003). *Geothermal energy: Utilization and technology*. UNESCO Publishing.

DiPippo, R. (2012). *Geothermal power plants: Principles, applications, case studies and environmental impact* (3.ª ed.). Butterworth-Heinemann.

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the Bootstrap*. Chapman and Hall/CRC.

Flores-Espino, F., Patel, S., & Flores, M. (2019). Machine learning approaches for geothermal resource assessment. *Renewable Energy*, *142*, 660–672.

Gehringer, M., & Loksha, V. (2012). *Geothermal handbook: Planning and financing power generation*. Energy Sector Management Assistance Program (ESMAP), Banco Mundial.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, *202*, 18–27. https://doi.org/10.1016/j.rse.2017.06.031

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Hulley, G. C., Hook, S. J., Abbott, E., & Malakar, N. (2015). The ASTER Global Emissivity Dataset (ASTER GED): Mapping Earth's emissivity at 100 meter spatial scale. *Geophysical Research Letters*, *42*(19), 7966–7976. https://doi.org/10.1002/2015GL065564

International Energy Agency. (2021). *Net zero by 2050: A roadmap for the global energy sector*. IEA. https://www.iea.org/reports/net-zero-by-2050

Lahsen, A., Sepúlveda, F., Rojas, J., & Palacios, C. (2015). Present status of geothermal exploration in Chile. *Proceedings of the World Geothermal Congress 2015*, 1–7.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, *521*(7553), 436–444. https://doi.org/10.1038/nature14539

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

Mia, M. B., Fujimitsu, Y., & Nishijima, J. (2018). Exploration of hydrothermal alteration and monitoring of thermal activity using multi-temporal Landsat and ASTER satellite imagery. *Journal of Volcanology and Geothermal Research*, *368*, 137–150.

Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*, *22*(10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191

Servicio Geológico Colombiano. (2023). *Inventario de fuentes termales de Colombia* (actualización 2023). SGC.

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, *6*(1), 60. https://doi.org/10.1186/s40537-019-0197-0

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.

Unidad de Planeación Minero Energética. (2020). *Plan Energético Nacional 2020-2050*. UPME.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems (NeurIPS)*, *27*, 3320–3328.

Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *Proceedings of the 6th International Conference on Learning Representations (ICLR)*.

Zhu, X. X., Tuia, D., Mou, L., Xia, G.-S., Zhang, L., Xu, F., & Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. *IEEE Geoscience and Remote Sensing Magazine*, *5*(4), 8–36. https://doi.org/10.1109/MGRS.2017.2762307

---

<!-- ============================================================
     ANEXOS
     ============================================================ -->

## Anexos

### Anexo A. Catálogo de campos geotérmicos del dataset

**A.1 Colombia**

Campos geotérmicos confirmados: Nevado del Ruiz (lon −75,322; lat 4,895 — alta entalpía, volcán activo, exploración avanzada ISAGEN/SGC), Volcán Puracé (lon −76,404; lat 2,321 — confirmado, fumarolas, exploración SGC), Paipa-Iza (lon −73,112; lat 5,778 — baja entalpía, aguas termales).

Volcanes activos con manifestaciones hidrotermales: Galeras, Cumbal, Azufral, Cerro Machín, Nevado del Tolima, Sotará, Doña Juana, Nevado del Huila, Chiles, Cerro Negro de Mayasquer, Cerro Bravo, Nevado de Santa Isabel, Romeral.

Zonas termales: Termales de Manizales, Coconucos, Santa Rosa de Cabal, Herveo, San Vicente Ferrer, Tabio, Choachí, Rivera (Huila), Guadalupe (Santander), San Agustín (Huila), Pitalito.

**A.2 Ecuador**

Campos geotérmicos: Tufiño-Chiles-Cerro Negro (binacional), Chalupas, Chachimbiro, Baños de Agua Santa, El Placer, Papallacta.

Volcanes activos: Cotopaxi, Tungurahua, Guagua Pichincha, Reventador, Sangay, Cayambe, Antisana, Chimborazo, Quilotoa, Illiniza, Atacazo, Soche, Pululagua, Cuicocha, Imbabura, Mojanda, Cotacachi.

**A.3 Perú**

Campos geotérmicos: Calientes (Tacna), Borateras (Tacna), Tutupaca, Ubinas, Salinas-Chivay (Colca), Río Jesús.

Volcanes activos: Misti, Ubinas, Sabancaya, Chachani, Huaynaputina, Ticsani, Yucamane, Tutupaca, Casiri, Coropuna, Ampato, Sara Sara, Hualca Hualca.

**A.4 Chile**

Campos geotérmicos operativos o en exploración: Cerro Pabellón (48 MW, operativo — primera planta geotérmica de Sudamérica), El Tatio (mayor campo de géiseres del hemisferio sur), Puchuldiza-Tuja, Calabozos, Nevados de Chillán.

Volcanes activos (zona volcánica central): Láscar, Putana, Ollagüe, San Pedro, Irruputuncu, Isluga, Guallatiri, Tacora, Parinacota, Licancabur, Surire.

Volcanes activos (zona volcánica sur): Tolhuaca, Cordón Caulle, Villarrica, Llaima, Copahue, Antuco, Callaqui.


### Anexo B. Evolución del proyecto v1 → v2 → v3

**Tabla 19**

*Evolución del proyecto a través de sus tres versiones*

| Aspecto | v1 (nov 2025) | v2 (feb 2026) | v3 (mar 2026) |
|---------|:---:|:---:|:---:|
| Imágenes originales | 85 | 200 | 2.019 |
| Imágenes augmentadas | 2.635 | 6.200 | 22.209 |
| Augmentaciones por imagen | ~30 | ~30 | 10 |
| Países | Colombia | Colombia | 4 (CO, EC, PE, CL) |
| Split | train_test_split (leakage) | GroupShuffleSplit | GroupShuffleSplit anti-leakage geográfico |
| Normalización | z-score + Rescaling | z-score per-image | z-score global por banda |
| Arquitectura | CNN custom | ResNet-inspired (5 M) | EfficientNetB0 + Adapter (4,4 M) |
| Entrenamiento | CPU | CPU | GPU RTX 4070 (WSL2) |
| Épocas | 23 | 22 | 80 (30 + 50, 2 fases) |
| Test accuracy | 68,43 % | 91,45 % | 92,28 % |
| Test recall | 48,10 % | 86,05 % | 93,17 % |
| ROC AUC | 0,8198 | 0,983 | 0,9737 |
| Fuga de datos | Sí | 62,5 % | 0 % |

**Lecciones aprendidas:**

1. La calidad de los datos importa más que la arquitectura. Los 28 bugs de la v1 tenían más impacto en el rendimiento que el cambio de arquitectura.
2. La fuga de datos es silenciosa y catastrófica. El leakage del 62,5 % no se manifestaba como un error obvio, sino como métricas artificialmente buenas.
3. La normalización global es crítica. Normalizar por imagen destruye la información absoluta que el modelo necesita para discriminar.
4. El transfer learning funciona en dominios no naturales. Los filtros de ImageNet (bordes, texturas) son transferibles a imágenes de emisividad térmica.
5. Auditar el pipeline completo es indispensable. Cada bug corregido contribuyó a la mejora final.


### Anexo C. Glosario de términos técnicos

| Término | Explicación |
|---------|-------------|
| Backbone | Red neuronal principal (EfficientNetB0, 237 capas) con los filtros visuales preentrenados |
| Batch (lote) | Subconjunto de 32 imágenes procesadas simultáneamente |
| Batch Normalization | Capa que normaliza las activaciones internas de la red en cada paso |
| Backpropagation | Algoritmo que calcula, para cada peso, cuánto contribuyó al error |
| Channel Adapter | Módulo de dos capas que traduce 7 bandas ASTER a 3 canales |
| Classification Head | Capas finales que producen la probabilidad de potencial geotérmico |
| Congelar (Freeze) | Bloquear los pesos para que no cambien durante el entrenamiento |
| CosineDecay | Estrategia donde el learning rate decae suavemente siguiendo una curva coseno |
| Dropout | Desactivación aleatoria de un porcentaje de neuronas durante el entrenamiento |
| EarlyStopping | Mecanismo que detiene el entrenamiento si la métrica no mejora |
| Época (Epoch) | Una pasada completa por todas las imágenes de entrenamiento |
| Fine-tuning | Ajuste fino: descongelar parte del backbone y reentrenarlo con learning rate bajo |
| Forward Pass | Recorrido de la imagen desde la primera capa hasta la predicción final |
| Global Average Pooling | Cálculo del promedio por canal para reducir dimensiones |
| Gradiente | Dirección y magnitud de corrección para cada parámetro |
| Learning Rate | Velocidad de aprendizaje que controla el tamaño de los ajustes |
| Loss (pérdida) | Número que cuantifica la diferencia entre predicción y realidad |
| Mixed Precision | Uso de aritmética de 16 bits para cálculos y 32 bits para acumulación |
| Optimizador (AdamW) | Algoritmo que actualiza los pesos usando los gradientes |
| Overfitting | Cuando el modelo memoriza los ejemplos en vez de aprender patrones generalizables |
| Pesos (Weights) | Los 4,4 millones de números decimales que componen el modelo |
| Regularización | Conjunto de técnicas que fuerzan al modelo a generalizar |
| Sigmoid | Función de activación que convierte cualquier valor real en una probabilidad entre 0 y 1 |
| Tensor | Matriz multidimensional que almacena los datos numéricos |
| Transfer Learning | Reutilización de un modelo preentrenado y adaptación a un nuevo dominio |
