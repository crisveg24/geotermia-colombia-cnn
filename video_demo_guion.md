# Guion de video demo - Geotermia CNN (máx. 5:00)

Sitio: https://geotermia-colombia-cnn.vercel.app
Narración: 4 voces. Total aprox. 700 palabras (~4:30 a ritmo normal).

| Voz | Rol | Voz provisional (edge-tts) |
|---|---|---|
| V1 | Presentador | es-CO-GonzaloNeural |
| V2 | Científica de datos | es-CO-SalomeNeural |
| V3 | Ingeniero de software | es-MX-JorgeNeural |
| V4 | Analista de resultados | es-MX-DaliaNeural |

---

## Escena 1 - Problema (0:00 - 0:35) - V1
**Pantalla:** Inicio de la web, scroll suave por el hero.

> Colombia está sobre el Cinturón de Fuego del Pacífico y guarda un enorme potencial geotérmico.
> Pero explorarlo es caro: un solo pozo exploratorio cuesta millones de dólares y meses de trabajo de campo.
> Por eso creamos Geotermia CNN, un proyecto de grado de Ingeniería de Sistemas de la Universidad de San Buenaventura, Bogotá.
> Su idea es simple: usar imágenes satelitales de la NASA e inteligencia artificial para decir, en segundos, qué tan prometedor es cualquier punto del país.

## Escena 2 - Los datos (0:35 - 1:20) - V2
**Pantalla:** Página Proyecto (objetivos, tecnologías), luego Arquitectura (pipeline).

> Los datos vienen del sensor ASTER de la NASA, consultados a través de Google Earth Engine.
> Para cada punto extraemos siete bandas: cinco de emisividad térmica infrarroja, la temperatura de la superficie y el índice de vegetación NDVI, que ayuda a separar el calor del suelo de la cobertura vegetal.
> Con ellas construimos un dataset de más de veintidós mil imágenes de cuatrocientas siete zonas únicas, y lo dividimos con cuidado para que ninguna zona aparezca a la vez en entrenamiento y en prueba.
> Así evitamos fugas de información y la evaluación es honesta.

## Escena 3 - El modelo (1:20 - 2:05) - V3
**Pantalla:** Página Arquitectura (diagrama CNN, Channel Adapter, fases de entrenamiento).

> El modelo es una red convolucional EfficientNetB0.
> Esta red nació para imágenes de tres canales, así que añadimos un Channel Adapter de uno por uno que convierte las siete bandas a un formato que la red entiende, sin perder información espectral.
> Entrenamos en dos fases: primero con el backbone congelado, y luego con ajuste fino.
> El resultado es un clasificador que entrega la probabilidad de que una zona tenga potencial geotérmico alto, medio o bajo.

## Escena 4 - Prueba real en vivo (2:05 - 3:50) - V1 y V3 (alternan)
**Pantalla:** Página Predicción. Modo Coordenadas. Prueba 1: Nevado del Ruiz (4.8824, -75.3411).
Prueba 2: llanura de los Llanos Orientales (4.1500, -72.0000) como contraste. Se muestra el resultado, la distancia a la fuente termal y la tabla de siete bandas.

> **V1:** Ahora lo probamos de verdad, con datos reales y sin trucos. Entramos al escáner térmico y escribimos las coordenadas del complejo volcánico Ruiz-Tolima.
> **V3:** Al pulsar Computar Modelo, el frontend llama a nuestra API en Flask. El backend consulta Google Earth Engine, arma el tensor de siete bandas y ejecuta la red neuronal en tiempo real.
> **V1:** Aquí está el resultado: potencial alto, con su porcentaje, la distancia a la fuente termal más cercana y el valor medio, mínimo, máximo y desviación de cada banda.
> **V3:** Ahora un punto de contraste, en la llanura de los Llanos Orientales, sin actividad volcánica.
> **V1:** El modelo lo clasifica con un potencial claramente menor. Dos puntos distintos, dos respuestas coherentes con la geología.

## Escena 5 - Métricas (3:50 - 4:20) - V4
**Pantalla:** Página Métricas (accuracy, AUC, matriz de confusión).

> En el conjunto de prueba, con zonas que el modelo nunca vio, alcanzamos un noventa y dos por ciento de exactitud, un noventa y siete por ciento de AUC-ROC y un recall superior al noventa y tres por ciento.
> Esto importa porque en exploración es peor ignorar una zona buena que revisar una dudosa.

## Escena 6 - En qué trabajamos ahora y cierre (4:20 - 4:55) - V2 y V4
**Pantalla:** Vuelta al inicio, con la URL.

> **V2:** Seguimos mejorando la plataforma. Redujimos el consumo de memoria para correr en un servidor gratuito, añadimos caché de predicciones, tolerancia a fallos de Earth Engine y manejo del arranque en frío.
> **V4:** Lo próximo es ampliar el catálogo de zonas, validar con expertos geólogos y comparar contra datos de pozos reales.
> **V2:** Geotermia CNN es abierta y reproducible. Pruébala en geotermia-colombia-cnn punto vercel punto app.
> **V4:** Gracias por ver.

---

## Notas de producción
- Arranque en frío: el backend en Render puede tardar hasta 90 s. Se graba la espera completa y se acelera o recorta en la edición, sin falsear el resultado final.
- Créditos de ElevenLabs: se aprueba el guion y el video con voces gratuitas (edge-tts) y solo se genera la versión final en ElevenLabs una vez, ~700 palabras (~4.000 caracteres).
