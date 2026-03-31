<!-- ============================================================
   PROYECTO DE GRADO – FORMATO APA 7.ª EDICIÓN (2020)
   Plantilla USB Colombia v.6 adaptada a Markdown
   ============================================================ -->

---

# Implementación de una interfaz web segura para el sistema de clasificación de zonas con potencial geotérmico: protección de datos de usuario y cumplimiento del estándar OWASP 2025


Cristian Camilo Vega Sánchez

Yuliet Katerin Espitia Ayala

Laura Sophie Rivera Martín


Asesor: Yeison Eduardo Conejo Sandoval


Universidad de San Buenaventura, Sede Bogotá.

Facultad de Ingeniería.

Programa de Ingeniería de Sistemas

Bogotá, Colombia

2026

---

<!-- ============================================================
     PÁGINA LEGAL
     ============================================================ -->

## Citar / How to cite

(Vega Sánchez et al., 2026)

**Referencia / Reference — Estilo APA 7.ª ed. (2020):**

Vega Sánchez, C. C., Espitia Ayala, Y. K., & Rivera Martín, L. S. (2026). *Implementación de una interfaz web segura para el sistema de clasificación de zonas con potencial geotérmico: protección de datos de usuario y cumplimiento del estándar OWASP 2025* [Proyecto de grado]. Universidad de San Buenaventura, Bogotá.

---

## DEDICATORIA

*(Esta página es opcional.)*

---

## Tabla de Contenido

- [INTRODUCCIÓN](#introducción)
- [Capítulo 1](#capítulo-1)
  - [Antecedentes](#antecedentes)
  - [Planteamiento del problema](#planteamiento-del-problema)
  - [Justificación y pregunta de Investigación](#justificación-y-pregunta-de-investigación)
  - [Objetivo General](#objetivo-general)
  - [Objetivos Específicos](#objetivos-específicos)
  - [Alcances y Limitaciones](#alcances-y-limitaciones)
  - [Estado del arte](#estado-del-arte)
  - [Metodología](#metodología)
- [Capítulo 2 — Desarrollo de Ingeniería](#capítulo-2)
- [Capítulo 3 — Análisis de Resultados](#capítulo-3)
- [CONCLUSIONES](#conclusiones)
- [RECOMENDACIONES](#recomendaciones)
- [REFERENCIAS](#referencias)
- [Anexo I](#anexo-i)

---

## Lista de tablas

Tabla 1. Título de Tabla .....................................................3

---

## Lista de Figuras

Figura 1. Descripción de la Figura. .........................................3

---

## INTRODUCCIÓN

La energía geotérmica constituye una fuente de energía renovable que aprovecha el calor almacenado en el interior de la Tierra, proveniente tanto del calor primordial de la formación del planeta como del decaimiento radiactivo de isótopos como uranio-238, torio-232 y potasio-40 (Dickson & Fanelli, 2003). Este calor genera un gradiente geotérmico promedio de 25 a 30 °C por kilómetro de profundidad que, en zonas volcánicas, puede superar los 100 °C por kilómetro. Esta fuente energética se incluye como una de las alternativas clave dentro de la transición hacia matrices energéticas más limpias, dado que produce emisiones de CO₂ sustancialmente menores en comparación con los combustibles fósiles (International Energy Agency [IEA], 2021).

Colombia se ubica en el Cinturón de Fuego del Pacífico, donde convergen las placas tectónicas de Nazca, Sudamericana y del Caribe, lo cual genera una intensa actividad volcánica y geotérmica. El Servicio Geológico Colombiano (SGC, 2023) estima un potencial geotérmico superior a 2.000 MW; no obstante, a la fecha no existe ninguna planta geotérmica operativa en el país. Esta brecha entre potencial y aprovechamiento motiva la búsqueda de herramientas que aceleren la fase inicial de exploración, específicamente el reconocimiento regional, cuyo costo oscila entre 50.000 y 200.000 dólares por proyecto (Gehringer & Loksha, 2012).

El avance reciente del aprendizaje profundo (deep learning) ha demostrado que las redes neuronales convolucionales (CNN) son capaces de extraer patrones complejos a partir de datos geoespaciales, incluyendo la identificación de anomalías térmicas y alteraciones mineralógicas en imágenes satelitales multiespectrales (Zhu et al., 2017), lo que valida la pertinencia del enfoque propuesto en este trabajo.

En este contexto, el presente proyecto propone el desarrollo de un modelo CNN basado en la arquitectura EfficientNetB0 (Tan & Le, 2019), complementado con un módulo adaptador de canales (Channel Adapter), para clasificar imágenes del sensor ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) y determinar la probabilidad de que una zona geográfica posea potencial geotérmico. El modelo fue entrenado con imágenes de cuatro países andinos que comparten un contexto geológico común de subducción y vulcanismo activo (Colombia, Ecuador, Perú y Chile) y su aplicación se circunscribe al territorio colombiano.

La investigación se enmarca en un enfoque cuantitativo de tipo experimental-aplicado, y su resultado principal es una herramienta de screening automatizado que contribuye a la priorización de zonas para inversión en exploración geotérmica detallada.

---

## Capítulo 1

### Antecedentes

#### 1.1.1. Exploración geotérmica en Colombia

La exploración geotérmica en Colombia tiene antecedentes de estudios realizados por entidades como ISAGEN y el Servicio Geológico Colombiano. El proyecto más avanzado es el del Nevado del Ruiz en los departamentos de Caldas y Tolima, donde ISAGEN realizó estudios de factibilidad que identificaron temperaturas superiores a 200 °C (Alfaro, 2015). Otras zonas evaluadas incluyen Chiles–Cerro Negro y Azufral en Nariño, Paipa–Iza en Boyacá, y Coconucos en Cauca.

No obstante, la cobertura de estos estudios se ha concentrado en zonas con manifestaciones superficiales evidentes (volcanes activos, fuentes termales), dejando sin evaluar regiones donde la actividad geotérmica podría no manifestarse en superficie de manera evidente. Los sistemas geotérmicos mejorados (EGS, por sus siglas en inglés) demuestran que la ausencia de manifestaciones superficiales no descarta la presencia de recursos geotérmicos en profundidad (Tester et al., 2006).

#### 1.1.2. Teledetección aplicada a la exploración geotérmica

El uso de sensores remotos para la identificación de recursos geotérmicos ha sido documentado extensamente en la literatura científica. Coolbaugh et al. (2007) demostraron la efectividad del sensor ASTER para detectar anomalías térmicas asociadas a actividad geotérmica en Nevada, Estados Unidos. Vaughan et al. (2005) utilizaron imágenes multiespectrales del infrarrojo térmico para mapear minerales de alteración hidrotermal en Steamboat Springs.

El producto ASTER Global Emissivity Dataset (AG100 v003) proporciona datos de emisividad y temperatura superficial con resolución de 100 metros y cobertura global, lo que lo convierte en una fuente de datos idónea para estudios de reconocimiento a escala regional (Hulley et al., 2015).

#### 1.1.3. Aprendizaje profundo en teledetección

La aplicación de técnicas de aprendizaje profundo a la clasificación de imágenes satelitales ha experimentado un crecimiento exponencial en la última década. Zhu et al. (2017) presentaron una revisión exhaustiva de las aplicaciones de Deep Learning en teledetección, destacando las CNN como la arquitectura dominante para tareas de clasificación y segmentación. He et al. (2016) introdujeron las redes residuales (ResNet), cuyas conexiones de atajo permiten entrenar redes más profundas sin degradación del rendimiento, alcanzando resultados sobresalientes en competiciones de clasificación de imágenes como ImageNet.

En el contexto específico de la exploración geotérmica, estudios como los de Mia et al. (2018) han aplicado técnicas de Machine Learning a datos ASTER para la identificación de alteraciones hidrotermales, aunque el uso de CNN profundas con arquitectura residual para esta tarea particular es aún incipiente y constituye un área de oportunidad investigativa.

#### 1.1.4. Estado del desarrollo geotérmico en Colombia

Mejía et al. (2014) documentaron el estado de desarrollo geotérmico del país hasta ese momento, identificando un potencial estimado en 2.210 MW y señalando como zonas prioritarias el Macizo Volcánico del Ruiz y la región binacional Tufiño-Chiles-Cerro Negro (en cooperación con Ecuador). Los autores señalan que, pese a que los estudios de exploración se iniciaron en la década de 1970 con el apoyo de entidades como el Servicio Geológico Colombiano (SGC) e ISAGEN, el desarrollo geotérmico colombiano continuaba siendo incipiente y sin capacidad instalada para generación eléctrica. Este contexto pone de manifiesto la necesidad de avanzar tanto en la exploración del recurso como en el desarrollo de metodologías que permitan caracterizar el potencial geotérmico del territorio nacional de forma más eficiente.

#### 1.1.5. Predicción del gradiente geotérmico en Colombia mediante aprendizaje automático

Mejía-Fragoso et al. (2024) propusieron un modelo de aprendizaje automático para estimar el gradiente geotérmico en regiones de Colombia donde no existen mediciones directas de pozo, empleando únicamente conjuntos de datos geofísicos de escala global y conocimiento geológico general. Tras evaluar múltiples algoritmos de regresión, encontraron que los árboles de regresión con refuerzo de gradiente (Gradient-Boosted Regression Trees) producían las predicciones óptimas, con un error inferior al 12 %. Como resultado principal presentaron un mapa nacional del gradiente geotérmico con valores entre 16,75 y 41,20 °C/km, cuya distribución es coherente con indicadores geológicos conocidos como fallas activas y manifestaciones termales. Este trabajo representa uno de los primeros antecedentes directos del uso de ML aplicado a la caracterización geotérmica del territorio colombiano y constituye un referente metodológico para el presente estudio.

---

### Planteamiento del problema

La exploración geotérmica convencional sigue una secuencia de cuatro fases: reconocimiento regional (50.000 – 200.000 USD), exploración de superficie (200.000 – 1.000.000 USD), exploración profunda (5 – 20 millones USD) y desarrollo (50 – 200 millones USD) (Gehringer & Loksha, 2012). Esta guía del Banco Mundial detalla los costos y fases de proyectos geotérmicos, y constituye la referencia estándar en la materia. La primera fase depende tradicionalmente de la recopilación manual de datos geológicos, análisis de imágenes satelitales por expertos y revisión bibliográfica, procesos que resultan lentos, costosos y limitados por la subjetividad del analista.

Colombia posee al menos seis zonas geotérmicas reconocidas por el Servicio Geológico Colombiano (Nevado del Ruiz, Chiles-Cerro Negro, Azufral, Paipa-Iza, Puracé-Coconucos y Santa Rosa de Cabal) con temperaturas estimadas entre 150 y más de 200 °C (SGC, 2023). A pesar de este potencial, la falta de herramientas automatizadas de reconocimiento regional limita la capacidad de identificar nuevas zonas candidatas a exploración.

Las técnicas de aprendizaje profundo han mostrado resultados prometedores en la clasificación de imágenes de teledetección. Según Zhu et al. (2017), "deep learning has achieved remarkable success in remote sensing image analysis due to its ability to automatically learn hierarchical feature representations from raw data" (p. 5). Esta capacidad de aprender representaciones de forma automática, sin extracción manual de características, fundamenta el enfoque técnico adoptado en el presente proyecto.

Sin embargo, la aplicación específica de CNN a la identificación de potencial geotérmico a partir de datos multiespectrales de emisividad térmica es un campo escasamente explorado, lo que plantea la pregunta de investigación que guía este trabajo.

---

### Justificación y pregunta de Investigación

¿Es posible desarrollar un modelo de clasificación basado en redes neuronales convolucionales que, a partir de imágenes satelitales multiespectrales del sensor ASTER, identifique zonas con potencial geotérmico en Colombia con una exactitud superior al 85 % y un AUC-ROC superior a 0,90?

Colombia requiere diversificar su matriz energética para reducir la dependencia de la generación hidroeléctrica, que representa más del 65 % de la capacidad instalada y es vulnerable a fenómenos climáticos como El Niño (Unidad de Planeación Minero-Energética [UPME], 2020). La geotermia, como fuente de generación base disponible las 24 horas del día, complementaría la generación existente sin depender de condiciones climáticas.

Desde la perspectiva tecnológica, el aprendizaje profundo ha demostrado ser capaz de superar el desempeño humano en tareas de clasificación de imágenes cuando se dispone de conjuntos de datos suficientemente grandes y representativos (LeCun et al., 2015). LeCun, Bengio y Hinton son considerados los padres del deep learning moderno; su revisión establece los fundamentos teóricos que respaldan la viabilidad del enfoque CNN propuesto en este trabajo.

Desde la perspectiva económica, una herramienta de screening automatizado que reduzca la lista de zonas candidatas antes de invertir en exploración de campo tiene el potencial de ahorrar millones de dólares al descartarse zonas sin potencial en la fase más temprana y económica del proceso (Gehringer & Loksha, 2012).

Desde la perspectiva académica, este proyecto aporta una metodología reproducible que integra teledetección, procesamiento de datos geoespaciales y deep learning, contribuyendo a un campo de investigación incipiente en Colombia. Además, el dataset construido — con 2.019 zonas georreferenciadas de cuatro países andinos — constituye un recurso reutilizable para investigaciones futuras.

---

### Objetivo General

- Desarrollar un modelo de clasificación basado en redes neuronales convolucionales que permita identificar zonas con potencial geotérmico en Colombia a partir de imágenes satelitales multiespectrales del sensor ASTER.

---

### Objetivos Específicos

- Construir un dataset georreferenciado de imágenes ASTER de siete bandas espectrales para zonas con y sin potencial geotérmico confirmado en la Región Andina (Colombia, Ecuador, Perú y Chile).
- Diseñar e implementar un pipeline de procesamiento de imágenes que incluya filtrado de valores faltantes, redimensionamiento, normalización global y aumento de datos, con una división anti-fuga geográfica que garantice cero porciento de solapamiento entre conjuntos.
- Implementar una arquitectura CNN basada en EfficientNetB0 con un módulo Channel Adapter que proyecte las siete bandas espectrales al espacio de tres canales requerido por el modelo base.
- Entrenar el modelo mediante una estrategia de transfer learning en dos fases (backbone congelado y fine-tuning) y evaluar su desempeño con métricas de exactitud, sensibilidad, precisión, F1-Score y AUC-ROC.
- Desarrollar una interfaz web que permita realizar predicciones de potencial geotérmico en tiempo real a partir de coordenadas geográficas.

---

### Alcances y Limitaciones

El alcance del presente proyecto comprende:

- La implementación de una versión web funcional del sistema de clasificación de zonas con potencial geotérmico, accesible desde navegador web mediante una interfaz Streamlit desplegable en servidor.
- La aplicación de los controles de seguridad correspondientes a los diez riesgos del estándar **OWASP Top 10 2025**, incluyendo gestión de autenticación y autorización, protección contra inyección, cifrado de comunicaciones y control de acceso.
- La protección de los datos personales de los usuarios registrados en el sistema, mediante cifrado en tránsito (HTTPS/TLS), gestión segura de sesiones, almacenamiento con hashing de credenciales y política de tratamiento de datos conforme a la Ley 1581 de 2012.
- La evaluación de seguridad del sistema mediante análisis estático (SAST) y dinámico (DAST), con reporte de hallazgos y remediaciones implementadas para cada riesgo OWASP catalogado.
- La documentación completa de las decisiones de diseño de seguridad, controles implementados y resultados de las pruebas realizadas.

No forma parte del alcance:

- La detección de intrusiones a nivel de red (NIDS) o a nivel de sistema operativo (HIDS); el sistema se limita a la capa de aplicación web.
- La estimación de temperaturas de reservorio o profundidades geotérmicas.
- La evaluación de viabilidad económica de proyectos geotérmicos.
- La certificación formal bajo normas ISO/IEC 27001 u otros estándares de seguridad.

---

### Estado del arte

En los últimos años, la exploración de recursos geotérmicos ha cobrado relevancia debido a la necesidad de diversificar la matriz energética mediante fuentes renovables. En este contexto, el uso de técnicas de inteligencia artificial, particularmente el aprendizaje profundo, ha permitido mejorar significativamente los procesos de análisis y predicción en diferentes dominios, incluyendo la geociencia y el análisis de imágenes satelitales.

Las redes neuronales convolucionales (CNN) han demostrado ser muy efectivas en tareas de procesamiento de imágenes, en especial en la teledetección. Diversos estudios han evidenciado su capacidad para clasificar imágenes satelitales con altos niveles de precisión. Por ejemplo, Hasan et al. (2025) proponen un modelo híbrido CNN-SVM que alcanza una precisión del 98 %, mientras que Adda et al. (2025) emplean MobileNetV2 logrando resultados similares en clasificación multiclase. Asimismo, Gill et al. (2023) utilizan modelos CNN secuenciales para la clasificación de coberturas terrestres, destacando la importancia del uso de datos multiespectrales para mejorar la representación de características del terreno.

En relación con el uso de datos provenientes de sensores remotos, estudios como los de Bai y Zhao (2020) y Appiah-Twum et al. (2023) resaltan la relevancia de integrar imágenes satelitales como ASTER y Landsat con modelos de aprendizaje profundo, permitiendo mejorar la resolución espacial y la interpretación geológica. De igual manera, Chen et al. (2023) abordan el problema de consistencia en imágenes satelitales mediante técnicas basadas en CNN, facilitando la generación de mosaicos de alta calidad que sirven como base para análisis posteriores.

Por otra parte, en el ámbito de las geociencias y la exploración del subsuelo, se han desarrollado múltiples enfoques basados en aprendizaje automático y profundo. Anandhi et al. (2025) combinan Random Forest y CNN para la evaluación de idoneidad de sitios geotérmicos utilizando datos geoespaciales y térmicos, mientras que Chen et al. (2026) proponen un modelo basado en CNN para la predicción de zonas de riesgo térmico a partir de datos multifuente. Asimismo, Zhou et al. (2025) emplean un modelo híbrido CNN-SVM para la predicción de temperatura en túneles geotérmicos, logrando altos niveles de precisión. Estos estudios evidencian el potencial de las técnicas de aprendizaje profundo en la caracterización de fenómenos geológicos complejos.

Adicionalmente, investigaciones como las de Vu et al. (2021) exploran el uso de arquitecturas como SegNet para la reconstrucción tridimensional de resistividad eléctrica del subsuelo, mientras que Esmaeilzadeh et al. (2023) emplean técnicas de procesamiento de imágenes sobre datos ASTER para la detección de zonas de alteración mineral. Estos enfoques destacan la importancia de combinar datos geofísicos y de teledetección para mejorar la precisión en la exploración de recursos naturales.

En el ámbito energético, diversos trabajos han aplicado modelos híbridos como CNN-LSTM y CNN-MLP para la predicción de consumo y comportamiento energético (Yoon et al., 2024; Mousa et al., 2026), evidenciando mejoras significativas frente a métodos tradicionales. Sin embargo, revisiones recientes como la de AlQemlas et al. (2026) señalan limitaciones importantes, como la falta de conjuntos de datos públicos, problemas de generalización y la ausencia de estándares de evaluación.

A pesar de los avances descritos, aún persisten diversas limitaciones en la literatura. En particular, se observa una escasa integración de modelos de deep learning avanzados con datos multiespectrales específicamente orientados a la exploración geotérmica. Asimismo, muchos estudios presentan limitaciones relacionadas con el tamaño de los datasets, la falta de validación geográfica adecuada y la baja generalización de los modelos en diferentes regiones. Adicionalmente, se evidencia una limitada aplicación de estas técnicas en contextos latinoamericanos, especialmente en Colombia, donde el potencial geotérmico aún no ha sido ampliamente explorado mediante herramientas basadas en inteligencia artificial.

Con el fin de sintetizar los principales aportes y limitaciones de los trabajos revisados, en la Tabla 1 se presenta un resumen comparativo de los estudios más relevantes.

**Tabla 1. Síntesis de estudios relacionados**

| Autor / Año | Problema | Objetivo | Datos | Metodología | Resultados | Limitaciones |
|---|---|---|---|---|---|---|
| Anandhi et al., 2025 | Dificultad en identificar sitios geotérmicos | Evaluar idoneidad de sitios | Datos geoespaciales y térmicos | Random Forest + CNN | Mejora en precisión | No valida fuga geográfica |
| Visaya et al., 2024 | Fallas en sistemas eléctricos | Detectar cortocircuitos | Señales eléctricas | Wavelet + CNN | 94% accuracy | No aplica a imágenes satelitales |
| Chen et al., 2023 | Inconsistencia de color en imágenes satelitales | Mejorar mosaicos | Imágenes satelitales | CNN + fusión de imágenes | Mejor calidad visual | No enfocado en clasificación |
| Hasan et al., 2025 | Alta complejidad de CNN | Reducir parámetros | Imágenes satelitales | CNN + SVM | 98% accuracy | No aplicado a geotermia |
| Adda et al., 2025 | Clasificación de imágenes satelitales | Clasificar imágenes | Imágenes RGB | MobileNetV2 | 98% accuracy | No usa datos multiespectrales |
| Gill et al., 2023 | Clasificación de cobertura terrestre | Clasificar áreas | Imágenes satelitales | CNN secuencial | Buen desempeño | Modelo básico |
| Vu et al., 2021 | Reconstrucción 3D del subsuelo | Tomografía eléctrica | Datos geofísicos | CNN (SegNet) | Resultados precisos | Datos sintéticos |
| Yamada et al., 2022 | Segmentación de partículas rocosas | Segmentar partículas en imágenes | Imágenes industriales | Mask R-CNN | Mejora significativa | No geológico |
| Murugesan et al., 2025 | Predicción energética compleja | Predecir consumo energético | Datos energéticos | ANN-CNN | 98.78% accuracy | No usa imágenes |
| Pavlov et al., 2024 | Uso limitado de geotermia | Analizar tecnologías geotérmicas | Datos teóricos | Revisión | Identifica tecnologías | No ML aplicado |
| Häfner et al., 2025 | Falta de herramientas en geotermia | Proponer toolkit VR | Datos geológicos | Simulación VR para geotermia | Mejor visualización | No predicción |
| Yoon et al., 2024 | Baja precisión en predicción energética | Mejorar predicción de consumo | Series temporales | CNN + LSTM | Alta precisión | No espacial |
| Esmaeilzadeh et al., 2023 | Detección de minerales en zonas de alteración | Identificar zonas de alteración mineral | ASTER | Procesamiento espectral (SAM, SID, MTMF) | Resultados consistentes con geología | No deep learning |
| Bai et al., 2020 | Baja resolución de datos geoquímicos | Integrar datos geoqu ímicos con teledetección | ASTER + datos geoquímicos | CNN (AlexNet) | Mejora de resolución espacial | Modelo antiguo |
| Appiah-Twum et al., 2023 | Clasificación geológica compleja | Clasificar litología en semiarid environment | Landsat-9 + ASTER | DenseNet | 83.89% accuracy | Precisión menor que enfoques individuales |
| AlQemlas et al., 2026 | Fallas en sistemas de energías renovables | Revisar IA en mantenimiento predictivo | SCADA, imágenes térmicas | Revisión sistemática | Identifica tendencias y arquitecturas | Falta de datasets públicos |
| Chen et al., 2026 | Riesgo térmico en túneles geológicos | Predecir zonas de riesgo térmico | Datos multifuente (SDGSAT-1) | CNN (LI-CNN) | AUC 0.81 | Menor precisión en zonas sin datos |
| Zhou et al., 2025 | Temperatura en túneles geotérmicos | Predecir temperatura en túneles | Datos geológicos | CNN + SVM | R² 92% | Dataset pequeño |
| Gomez et al., 2024 | Variabilidad en modelos de nivel freático | Analizar desempeño de CNN 1D | Series temporales | CNN 1D | Buen desempeño | Problemas de generalización |
| Mousa et al., 2026 | Predicción de áreas quemadas por incendios | Mejorar predicción de incendios forestales | Datos meteorológicos | CNN + MLP (PSO-WOA) | R² 99.89% | No geológico |
| Mejía-Fragoso et al., 2024 | Falta de mediciones del gradiente geotérmico | Estimar gradiente geotérmico con ML | Datos geofísicos globales (gravedad, SRTM, geológicos) | Gradient-Boosted Regression Tree | Precisión ≤12%, mapa 16.75–41.20 °C/km | No usa imágenes satelitales multiespectrales |
| Mejía et al., 2014 | Desarrollo incipiente de geotermia en Colombia | Documentar estado del arte geotérmico | Datos históricos de exploración, ISAGEN | Revisión + 2 estudios de campo | 2 proyectos viables identificados (190 MW combinados) | Estudio de 2014, sin modelos ML |

Finalmente, a partir del análisis realizado, se identifica un vacío en la literatura en cuanto a la aplicación de modelos avanzados de redes neuronales convolucionales, como EfficientNet, integrados con datos multiespectrales provenientes de sensores como ASTER, y validados mediante estrategias que eviten la fuga de información geográfica. En este sentido, el presente trabajo propone un modelo de clasificación binaria basado en CNN para la identificación de zonas con potencial geotérmico en la región andina, con énfasis en Colombia, contribuyendo así al avance en el uso de inteligencia artificial aplicada a la exploración de energías renovables.

---

### Metodología

#### Enfoque y tipo de investigación

La investigación se enmarca en un enfoque cuantitativo de tipo experimental-aplicado y empírico-analítico. Se diseñó e implementó una aplicación web sobre un modelo CNN preentrenado (GeotermiaCNN_V7), sometiendo la capa de aplicación a evaluaciones de seguridad controladas y verificando el cumplimiento de los controles del estándar OWASP Top 10 2025 mediante métricas objetivas.

#### Metodología específica: CRISP-DM

Como marco metodológico para el desarrollo del proyecto se adoptó **CRISP-DM** (Cross-Industry Standard Process for Data Mining), propuesto por Chapman et al. (2000). CRISP-DM estructura los proyectos de minería de datos y aprendizaje automático en seis fases iterativas y es la metodología más utilizada en la industria y la academia para este tipo de proyectos. Su carácter cíclico permite revisitar fases anteriores cuando los hallazgos de una fase posterior lo requieren.

A continuación se describe cada fase y su correspondencia con las actividades realizadas en el presente trabajo:

**Fase 1. Comprensión del negocio (Business Understanding).** Se identificó la necesidad de exponer el modelo CNN de clasificación geotérmica a través de una interfaz web segura, accesible por múltiples usuarios con diferentes roles. Se definieron los objetivos del proyecto: implementar una interfaz Streamlit desplegable en servidor, aplicar los diez controles del estándar OWASP Top 10 2025, proteger los datos personales de los usuarios conforme a la Ley 1581 de 2012, y evaluar la seguridad de la aplicación mediante SAST y DAST. Esta fase se documenta en las secciones de Planteamiento del problema, Justificación y Objetivos.

**Fase 2. Comprensión de los datos (Data Understanding).** Se analizó el modelo preentrenado (GeotermiaCNN_V7, 4.396.112 parámetros, 60 MB, formato `.keras`) y sus requerimientos de entrada: imágenes ASTER de 224 × 224 × 7 bandas espectrales normalizadas con las estadísticas globales almacenadas en `band_stats_v3.json`. Se caracterizaron también los datos de usuario que la aplicación web gestionará: credenciales (correo y contraseña), historial de predicciones (coordenadas, resultados, fecha) y datos de sesión.

**Fase 3. Preparación de los datos (Data Preparation).** Se configuró el entorno de despliegue seguro: generación de certificados TLS para HTTPS, configuración de la política CORS, definición del esquema de base de datos para usuarios y predicciones, y reutilización del pipeline de normalización existente (`band_stats_v3.json`) para garantizar que las imágenes descargadas desde Google Earth Engine en tiempo de inferencia reciban el mismo preprocesamiento que las imágenes de entrenamiento.

**Fase 4. Modelado (Modeling).** Se diseñó e implementó la arquitectura de la aplicación web: módulo de autenticación con hashing bcrypt, control de acceso basado en roles (RBAC), validación y sanitización de entradas de usuario, integración segura con la API de Google Earth Engine para descarga de imágenes ASTER, inferencia sobre el modelo CNN, y visualización de resultados mediante mapas interactivos Folium y generación de reportes en PDF. Para cada uno de los diez riesgos del OWASP Top 10 2025 se diseñó e implementó el control de seguridad correspondiente.

**Fase 5. Evaluación (Evaluation).** Se verificó el cumplimiento de los controles OWASP mediante análisis estático del código (SAST) con Bandit y Semgrep, y análisis dinámico de la aplicación en ejecución (DAST) con OWASP ZAP. Se documentaron los hallazgos encontrados en cada herramienta y las remediaciones implementadas. Se verificó que la aplicación no presentara ningún hallazgo de severidad alta o crítica al finalizar el ciclo de evaluación.

**Fase 6. Despliegue (Deployment).** La aplicación se desplegó en un servidor con HTTPS/TLS, cabeceras de seguridad HTTP (CSP, HSTS, X-Frame-Options), gestión segura de variables de entorno (sin credenciales en código fuente), política de tratamiento de datos conforme a la Ley 1581 de 2012, y monitoreo de eventos de seguridad mediante registro estructurado de logs.

---

## Capítulo 2

### Desarrollo de Ingeniería

En este capítulo se describen los procedimientos realizados para implementar la interfaz web segura sobre el modelo GeotermiaCNN_V7, aplicar los controles del estándar OWASP Top 10 2025 y garantizar la protección de los datos personales de los usuarios conforme a la Ley 1581 de 2012.

---

#### 2.1 Arquitectura de la interfaz web

La interfaz web se implementó con **Streamlit**, un framework Python de código abierto orientado al despliegue rápido de aplicaciones de datos. Se eligió Streamlit porque el modelo ya estaba desarrollado en Python (TensorFlow/Keras), lo que permite integrar directamente la lógica de inferencia sin necesidad de serializar el modelo a una API REST separada, reduciendo la superficie de ataque.

La arquitectura de la aplicación se organiza en tres capas:

**Tabla 2**

*Capas de la arquitectura de la aplicación web*

| Capa | Componente | Responsabilidad |
|---|---|---|
| Presentación | Streamlit + Folium | Formularios, mapas interactivos, reportes PDF |
| Lógica de negocio | Python (TensorFlow/Keras + GEE) | Autenticación, autorización, inferencia CNN |
| Datos | SQLite / PostgreSQL + archivos .npy | Usuarios, historial de predicciones, estadísticas |

El flujo de una predicción autenticada sigue los pasos descritos a continuación:

1. El usuario introduce sus credenciales en el formulario de inicio de sesión.
2. La aplicación verifica la contraseña contra el hash bcrypt almacenado en la base de datos.
3. Se genera un token de sesión seguro (UUID v4, almacenado en `st.session_state`) con tiempo de expiración de 30 minutos.
4. El usuario ingresa coordenadas geográficas (latitud, longitud) para la zona a evaluar.
5. La aplicación solicita la imagen ASTER AG100 v003 a Google Earth Engine mediante la clave de servicio almacenada como variable de entorno (nunca en código fuente).
6. La imagen descargada se procesa a través del pipeline de normalización (`band_stats_v3.json`) y se redimensiona a 224 × 224 × 7.
7. El modelo GeotermiaCNN_V7 produce una probabilidad de potencial geotérmico entre 0 y 1.
8. El resultado se visualiza en un mapa Folium con marcador codificado por color (verde ≥ 0,50 / rojo < 0,50) y se registra en el historial del usuario.
9. El usuario puede descargar un reporte en PDF con los metadatos de la predicción.

**Figura 2**

*Diagrama de flujo de la aplicación web*

```
Usuario (navegador)
        │  HTTPS/TLS
        ▼
┌─────────────────────────────────┐
│  Streamlit Web Server           │
│  ┌──────────┐  ┌─────────────┐  │
│  │  Auth    │  │  RBAC       │  │
│  │  bcrypt  │  │  (roles)    │  │
│  └────┬─────┘  └──────┬──────┘  │
│       │               │         │
│       ▼               ▼         │
│  ┌─────────────────────────┐    │
│  │  Lógica de Inferencia   │    │
│  │  GEE → pipeline → CNN   │    │
│  └──────────┬──────────────┘    │
│             │                   │
│             ▼                   │
│  ┌──────────────────────────┐   │
│  │  Base de datos (cifrada) │   │
│  │  Usuarios + historial    │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

*Nota.* Todas las comunicaciones entre el navegador y el servidor se realizan sobre HTTPS con TLS 1.2 o superior. Las variables de entorno se gestionan con `python-dotenv`; ninguna credencial se almacena en el repositorio de código.

---

#### 2.2 Implementación del estándar OWASP Top 10 2025

Para cada uno de los diez riesgos del estándar OWASP Top 10 2025 se diseñó e implementó un control de seguridad específico. La Tabla 3 resume los controles aplicados.

**Tabla 3**

*Controles de seguridad implementados por riesgo OWASP Top 10 2025*

| # | Riesgo OWASP | Control implementado |
|---|---|---|
| A01 | Broken Access Control | RBAC con roles `admin` / `user`; restricción de rutas por sesión activa; verificación de propiedad del recurso antes de consultar historial |
| A02 | Cryptographic Failures | HTTPS/TLS 1.2+; credenciales hasheadas con bcrypt (cost factor 12); sin datos sensibles en logs ni en URL |
| A03 | Injection | Consultas parametrizadas (SQLAlchemy ORM); validación y sanitización de coordenadas de entrada con `pydantic`; sin construcción de SQL dinámico |
| A04 | Insecure Design | Modelado de amenazas (STRIDE) previo al desarrollo; principio de privilegio mínimo; separación de responsabilidades entre capas |
| A05 | Security Misconfiguration | Cabeceras HTTP de seguridad: `Content-Security-Policy`, `Strict-Transport-Security`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`; modo debug deshabilitado en producción |
| A06 | Vulnerable and Outdated Components | Auditoría de dependencias con `pip-audit` y `safety`; `requirements.txt` con versiones fijas; actualización periódica documentada |
| A07 | Identification and Authentication Failures | Límite de intentos de inicio de sesión (5 intentos / 15 min con bloqueo temporal); tokens de sesión con expiración; cierre de sesión explícito que invalida el token |
| A08 | Software and Data Integrity Failures | Verificación de integridad del modelo `.keras` mediante hash SHA-256 al arranque; política de revisión de dependencias antes de actualizar |
| A09 | Security Logging and Monitoring Failures | Registro estructurado de eventos de seguridad (inicio de sesión, fallos de autenticación, predicciones) con timestamp, IP y usuario; alertas ante N fallos consecutivos |
| A10 | Server-Side Request Forgery (SSRF) | Validación del dominio de salida en la integración con GEE (allowlist de dominios `earthengine.googleapis.com`); sin retransmisión de URLs arbitrarias proporcionadas por el usuario |

A continuación se describen en detalle los tres controles de mayor complejidad de implementación.

**A01 — Control de acceso (RBAC).** Se definieron dos roles: `admin` (acceso a panel de gestión de usuarios y monitoreo de logs) y `user` (acceso a predicciones e historial propio). Cada ruta de la aplicación verifica la existencia y validez del token de sesión antes de ejecutar cualquier lógica de negocio. El historial de predicciones se filtra por `user_id`, impidiendo que un usuario acceda a los registros de otro.

**A03 — Prevención de inyección.** Todas las interacciones con la base de datos se realizan a través del ORM SQLAlchemy con consultas parametrizadas. Las coordenadas geográficas ingresadas por el usuario se validan con `pydantic` (latitud ∈ [−90, 90], longitud ∈ [−180, 180]) antes de ser procesadas o almacenadas. Los nombres de usuario se normalizan y se recorta cualquier carácter de control.

**A07 — Gestión de autenticación.** El mecanismo de bloqueo ante intentos fallidos utiliza un contador almacenado en sesión y en base de datos. Tras cinco intentos fallidos en una ventana de 15 minutos, la cuenta queda bloqueada temporalmente y se registra el evento. Las contraseñas se almacenan únicamente como hash bcrypt con salt aleatorio; no existe ningún mecanismo de recuperación que exponga la contraseña original.

---

#### 2.3 Protección de datos de usuarios — Ley 1581 de 2012

La Ley 1581 de 2012 (Ley de Protección de Datos Personales de Colombia) establece los principios de legalidad, finalidad, libertad, veracidad, transparencia, acceso y circulación restringida, seguridad y confidencialidad para el tratamiento de datos personales. Para dar cumplimiento a esta norma se implementaron las siguientes medidas:

**Tabla 4**

*Medidas de protección de datos por principio*

| Principio | Medida implementada |
|---|---|
| Finalidad | Los datos de usuario (correo, historial) se recopilan únicamente para autenticación y registro de predicciones; no se comparten con terceros |
| Libertad | El usuario puede solicitar la eliminación de su cuenta y todos sus datos desde el panel de configuración |
| Seguridad | Credenciales con bcrypt; base de datos con cifrado en reposo (SQLCipher); comunicación HTTPS/TLS |
| Confidencialidad | Acceso a datos restringido por RBAC; los administradores no tienen acceso a las contraseñas (solo al hash) |
| Transparencia | Política de tratamiento de datos visible antes del registro; consentimiento explícito requerido |

Datos recopilados y su finalidad:

- **Correo electrónico:** identificación del usuario; generación de notificaciones del sistema.
- **Contraseña (hash bcrypt):** autenticación; nunca se almacena en texto plano.
- **Historial de predicciones** (coordenadas, probabilidad, fecha): funcionalidad de consulta personal; no se comparte ni se vende.
- **Logs de seguridad** (IP, timestamp, tipo de evento): monitoreo de seguridad; se retienen por 90 días.

La política de tratamiento de datos se presenta al usuario en el formulario de registro y debe ser aceptada explícitamente (checkbox requerido) antes de crear la cuenta.

---

#### 2.4 Evaluación de seguridad: SAST y DAST

Se realizaron dos tipos de evaluación de seguridad sobre la aplicación web:

**Análisis estático (SAST — Static Application Security Testing).** Se ejecutaron dos herramientas de análisis estático sobre el código fuente de la aplicación:

- **Bandit:** Herramienta específica para Python que detecta patrones de código inseguros (uso de `eval`, `exec`, `subprocess` con entrada de usuario, contraseñas hardcodeadas, uso de algoritmos criptográficos débiles, etc.). Se configuró con perfil de severidad media–alta.
- **Semgrep:** Analizador semántico basado en reglas que detecta vulnerabilidades a nivel de lógica de aplicación. Se utilizó el ruleset `python.flask.security` y `python.django.security` adaptado para Streamlit.

Los hallazgos encontrados en el análisis inicial y sus remediaciones se documentan en la Tabla 5.

**Tabla 5**

*Hallazgos SAST y remediaciones aplicadas*

| Herramienta | Hallazgo inicial | Severidad | Remediación aplicada |
|---|---|---|---|
| Bandit | `subprocess.run` con `shell=True` en script auxiliar | Media | Reemplazado por llamada con lista de argumentos y `shell=False` |
| Bandit | Uso de `assert` para validación de seguridad | Baja | Reemplazado por `if not ... raise ValueError` |
| Semgrep | Consulta SQL construida con concatenación de strings | Alta | Migrado a ORM SQLAlchemy con parámetros vinculados |
| Semgrep | Token de sesión con longitud insuficiente (8 caracteres) | Alta | Cambiado a UUID v4 (128 bits de entropía) |

Tras aplicar las remediaciones, ambas herramientas reportaron cero hallazgos de severidad media o alta.

**Análisis dinámico (DAST — Dynamic Application Security Testing).** Se empleó **OWASP ZAP** (Zed Attack Proxy) en modo de escaneo automatizado sobre la aplicación en ejecución en entorno de staging (no producción). El escaneo activo realizó las siguientes pruebas:

- Inyección SQL y XSS en todos los campos de entrada.
- Detección de cabeceras de seguridad HTTP faltantes.
- Enumeración de rutas no protegidas.
- Pruebas de CSRF en formularios.
- Verificación de configuración TLS (versiones, cifrados, certificado).

**Tabla 6**

*Resultados del escaneo DAST con OWASP ZAP (post-remediación)*

| Categoría | Alertas altas | Alertas medias | Alertas bajas | Informativas |
|---|:---:|:---:|:---:|:---:|
| Inyección (SQL, XSS) | 0 | 0 | 0 | 0 |
| Cabeceras de seguridad | 0 | 0 | 0 | 2 |
| Autenticación | 0 | 0 | 0 | 1 |
| Configuración TLS | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **3** |

Las tres alertas informativas corresponden a: (i) ausencia de cabecera `Permissions-Policy` (no requerida por OWASP Top 10), (ii) ausencia de página `robots.txt` y (iii) cabecera `Server` que revela la versión de Python (mitigada mediante proxy inverso Nginx que sobreescribe dicha cabecera en producción).

---

## Capítulo 3

### Análisis de Resultados

Mostrar de forma concisa, organizada y siguiendo un orden lógico la información (figuras, tablas, etc.) relacionada con cada una de las pruebas definidas. Se sugiere la siguiente estructura:

(i) Analiza los resultados obtenidos a la luz de lo que predice la teoría y de lo que se esperaba del experimento y describiendo explícitamente si el resultado obtenido permite o no verificar el correcto funcionamiento del sistema y el cumplimiento de los objetivos específicos asociados.

(ii) Compara con otros trabajos realizados previamente (si aplica) a nivel local, nacional o internacional; esto resulta de gran utilidad pues permite evidenciar el aspecto innovador del trabajo (y permitiría además identificar su potencial para ser publicado posteriormente).

---

## CONCLUSIONES

Las conclusiones deben ser la respuesta a los objetivos o propósitos planteados. Deben contemplar las perspectivas de la investigación, las cuales son sugerencias, proyecciones o alternativas que se presentan para modificar, cambiar o incidir sobre una situación específica o una problemática encontrada. Pueden presentarse como un texto con características argumentativas, resultado de una reflexión acerca del trabajo de investigación.

---

## RECOMENDACIONES

*(Esta sección es opcional.)*

Se presentan como una serie de aspectos que se podrían realizar en un futuro para emprender investigaciones similares o fortalecer la investigación realizada.

---

## REFERENCIAS

Adda, S., Valeti, H., Enduri, M. K., Salla, G., & Tejaswi, A. (2025). Deep learning for aerial and satellite image analysis: A CNN-based approach. *2025 IEEE 14th International Conference on Communication Systems and Network Technologies (CSNT)*. https://ieeexplore.ieee.org/document/10967713

AlQemlas, T., Saber, A., Abdelfattah, A., Mokhiamar, O., Amine, S., & Gazo-Hanna, E. (2026). Artificial intelligence-driven fault detection and predictive maintenance in renewable energy systems: A review. *Engineered Science Publisher*. https://www.scopus.com/pages/publications/105032226962

Anandhi, R. J., Arun, V., Singh, N., Kumar, P. S., Rangaiah, Y. P., Sharma, S., & Jabbar, F. D. (2025). Geospatial analysis and machine learning for site suitability of geothermal energy plants. *2025 International Conference on Cognitive Computing in Engineering, Communications, Sciences and Biomedical Health Informatics (IC3ECSBHI)*. https://ieeexplore.ieee.org/document/10991171

Appiah-Twum, M., Wenbo, X., Mawuli, C. B., & Atandoh, P. (2023). Lithological classification using densely connected convolution network on Landsat-9 and ASTER datasets in a semi-arid environment. *2023 20th International Computer Conference on Wavelet Active Media Technology and Information Processing (ICCWAMTIP)*. https://ieeexplore.ieee.org/document/10387076

Bai, S., & Zhao, J. (2020). Interpolation of geochemical data with ASTER images based on AlexNet convolution neural network. *IGARSS 2020 - IEEE International Geoscience and Remote Sensing Symposium*. https://ieeexplore.ieee.org/document/9324116

Chen, S., Lv, Q., Cui, P., & Guo, S. (2023). CNN-based image color consistency for multi-source satellite images: A case study in China's water diversion project application. *2023 9th Annual International Conference on Network and Information Systems for Computers (ICNISC)*. https://ieeexplore.ieee.org/document/10473449

Chen, Z., Guo, S., Guo, H., Yu, Z., Lei, X., Ji, Q., He, Z., Chang, R., Sun, Z., Pei, X., Zhou, Z., & Picco, L. (2026). Prediction and interpretation of high-temperature heat damage risk zones based on a location-information-fusion convolution neural network. *Elsevier*. https://www.scopus.com/pages/publications/105024012037

Esmaeilzadeh, N., Barak, S., Gani, N. D., Imamalipour, A., Abedi, M., & Pour, A. B. (2023). Alteration zones detection using image-based and spectrum-based image processing techniques to ASTER data. https://ieeexplore.ieee.org/document/10281515

Gill, K. S., Anand, V., Chauhan, R., Garg, A., & Gupta, R. (2023). Classification of satellite images and predicting field areas after fine-tuning using sequential CNN model. *2023 2nd International Conference on Futuristic Technologies (INCOFT)*. https://ieeexplore.ieee.org/document/10425312

Gomez, M., Nölscher, M., Hartmann, A., & Broda, S. (2024). Assessing groundwater level modelling using a 1-D convolutional neural network (CNN): Linking model performances to geospatial and time series features. https://www.scopus.com/pages/publications/85206458889

Häfner, V., Häfner, P., Michels, F. L., Bauer, F., & Grethler, M. (2025). PolyVRGeo: Immersive engineering toolkit for deep geothermal energy. *2025 IEEE Conference on Virtual Reality and 3D User Interfaces Workshops (VRW)*. https://ieeexplore.ieee.org/document/10972724

Hasan, N., Islam, M. S., & Hayat, M. T. (2025). Satellite image classification using a lightweight CNN-SVM hybrid model. *2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)*. https://ieeexplore.ieee.org/document/11013863

Mejía, E., Rayo, L., Méndez, J., & Echeverri, J. (2014). Geothermal development in Colombia. *Short Course VI on Utilization of Low- and Medium-Enthalpy Geothermal Resources and Financial Aspects of Utilization*. UNU-GTP and LaGeo, Santa Tecla, El Salvador. ISAGEN S.A. ESP.

Mejía-Fragoso, J. C., Flórez, M. A., & Bernal-Olaya, R. (2024). Predicting the geothermal gradient in Colombia: A machine learning approach. *Geothermics, 122*, 103074. https://doi.org/10.1016/j.geothermics.2024.103074

Mousa, M. H., Algamdi, A. M., Fouad, Y., & Elshewey, A. M. (2026). CNN-MLP framework for forest burned areas prediction using PSO-WOA algorithm. https://www.scopus.com/pages/publications/105029499758

Murugesan, R., Kumar, A., Kumar, H., Sasidhar, U., & Kumar, S. (2025). Analyzing the impact of meteorological factors on new energy power prediction using capsule networks with hybrid embedding layer. *2025 3rd International Conference on Data Science and Information System (ICDSIS)*. https://ieeexplore.ieee.org/document/11070717

Pavlov, A. A., & Lavrenov, A. A. (2024). Prospects for the implementation of innovative technologies in geothermal energy production. *2024 XXVII International Conference on Soft Computing and Measurements (SCM)*. https://ieeexplore.ieee.org/document/10554136

Visaya, R. L. G., & Sangalang, R. G. B. (2024). Design and simulation of short circuit fault detection in DC power supply using Daubechies wavelet and convolutional neural networks. *2024 9th International Conference on Mechatronics Engineering (ICOM)*. https://ieeexplore.ieee.org/document/10652523

Vu, M. T., & Jardani, A. (2021). Convolutional neural networks with SegNet architecture applied to three-dimensional tomography of subsurface electrical resistivity. *Geophysical Journal International*. https://academic.oup.com/gji/article/225/2/1319/6105326

Yamada, T., & Di Santo, S. (2022). Instance segmentation of piled rock particles based on Mask R-CNN. *IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium*. https://ieeexplore.ieee.org/document/9883183

Yoon, N., Lee, S., Kim, S. K., Park, C., & Kim, T. (2024). Energy consumption prediction using CNN-LSTM models: A time series big data analysis. *2024 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)*. https://ieeexplore.ieee.org/document/10774020

Zhou, L., Yan, P., Li, X., Liu, T., Liu, Z., & Jia, W. (2025). Research on prediction model of high geothermal tunnels temperature based on CNN-SVM. *Elsevier*. https://www.scopus.com/pages/publications/105012946980

> **Nota:** Verificar que todas las referencias se ajusten a la normatividad APA y sean utilizadas dentro del documento.

---

## Anexo I

Utilice el anexo para incluir datos, instrumentos de investigación y material adicional que aporte a la consecución de los objetivos y alcances del proyecto de grado. Se debe incluir el cronograma de actividades y presupuesto (si aplica).
