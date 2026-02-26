# Contexto Geotérmico: Fundamentos y Rol del Modelo CNN

## 1. ¿Qué es la Energía Geotérmica?

La energía geotérmica es el **calor almacenado en el interior de la Tierra**. Este calor proviene de dos fuentes principales:

- **Calor primordial**: remanente de la formación del planeta hace ~4,500 millones de años.
- **Decaimiento radiactivo**: desintegración de isótopos como uranio-238, torio-232 y potasio-40 en la corteza y el manto.

Este calor se manifiesta en un **gradiente geotérmico** que, en promedio, aumenta ~25–30 °C por cada kilómetro de profundidad, aunque en zonas volcánicas o tectónicamente activas puede superar los 100 °C/km.

## 2. El "Triángulo Geotérmico"

Para que un recurso geotérmico sea **explotable** se necesitan tres componentes:

```
        🔥 Fuente de Calor
           /        \
          /          \
         /   RECURSO  \
        /  GEOTÉRMICO  \
       /________________\
   🪨 Reservorio      💧 Fluido
   (roca permeable)   (agua/vapor)
```

| Componente | Descripción |
|------------|-------------|
| **Fuente de calor** | Intrusión magmática, gradiente geotérmico elevado o roca caliente profunda |
| **Reservorio** | Formación rocosa con suficiente porosidad y permeabilidad para almacenar y transmitir fluido caliente |
| **Fluido** | Agua (líquida o en fase vapor) que actúa como medio de transporte del calor hacia la superficie |

### ¿Se necesitan aguas termales para explotar geotermia?

**No necesariamente.** Existen tres tipos principales de sistemas geotérmicos:

### 2.1 Sistemas Hidrotermales (convencionales)

Son los más conocidos y los primeros explotados históricamente. En estos sistemas **sí existe agua subterránea** que se calienta naturalmente al entrar en contacto con rocas calientes. El fluido caliente sube por convección natural y puede manifestarse en superficie como:

- Aguas termales y fuentes calientes
- Fumarolas y géiseres
- Suelos con temperaturas anómalas

**Ejemplos:** The Geysers (California), Cerro Machín y zona Paipa-Iza (Colombia), Islandia.

**Temperaturas típicas:** 150–350 °C a profundidades de 1–3 km.

### 2.2 Sistemas Geotérmicos Mejorados (EGS – Enhanced Geothermal Systems)

En estos sistemas **hay roca caliente pero NO hay suficiente agua ni permeabilidad natural**. La solución:

1. Se perfora hasta la roca caliente seca (Hot Dry Rock).
2. Se inyecta agua a alta presión para crear fracturas artificiales (estimulación hidráulica).
3. El agua inyectada se calienta al circular por las fracturas.
4. Se extrae el agua caliente por un segundo pozo.

**No requieren aguas termales ni acuíferos previos.** Solo necesitan roca caliente a profundidad accesible.

**Ejemplos:** Proyecto FORGE (Utah, EE.UU.), Soultz-sous-Forêts (Francia).

**Temperaturas típicas:** 150–300 °C a profundidades de 3–6 km.

### 2.3 Uso Directo y Bombas de Calor Geotérmicas

Para aplicaciones de baja temperatura (calefacción, invernaderos, acuicultura, procesos industriales) se puede aprovechar:

- **Acuíferos tibios** (30–100 °C) sin necesidad de generar electricidad.
- **Bombas de calor geotérmicas (GSHP):** Aprovechan la temperatura estable del subsuelo (~15 °C a pocos metros de profundidad) en prácticamente cualquier lugar del mundo.

**Estas aplicaciones no requieren vulcanismo ni aguas termales.**

## 3. Indicadores de Potencial Geotérmico

Las características que indican un posible recurso geotérmico, y que pueden detectarse remotamente, incluyen:

| Indicador | Detectable por satélite | Banda ASTER relevante |
|-----------|:-----------------------:|----------------------|
| Anomalías térmicas superficiales | ✅ | TIR (bandas 10–14) |
| Alteración hidrotermal de minerales | ✅ | SWIR (bandas 4–9), VNIR (1–3) |
| Composición mineralógica (arcillas, sílice, óxidos) | ✅ | SWIR, TIR |
| Gradiente geotérmico elevado | ❌ (indirecto) | — |
| Presencia de vulcanismo reciente | ✅ | VNIR, TIR |
| Fallas y lineamientos tectónicos | ✅ (parcial) | VNIR |

### ¿Por qué ASTER?

El sensor **ASTER** (Advanced Spaceborne Thermal Emission and Reflection Radiometer) a bordo del satélite Terra de la NASA es especialmente adecuado porque:

- **14 bandas** que cubren VNIR (3), SWIR (6) y TIR (5)
- Las bandas SWIR son ideales para detectar **alteración hidrotermal** (minerales como alunita, caolinita, montmorillonita)
- Las bandas TIR permiten detectar **anomalías térmicas** y composición mineral de silicatos
- Cobertura global y datos gratuitos via Google Earth Engine
- El producto **ASTER GED (AG100)** provee emisividad y temperatura superficial a resolución de 100 m

## 4. Contexto Geotérmico de Colombia

Colombia tiene un alto potencial geotérmico debido a su ubicación en el **Cinturón de Fuego del Pacífico**, donde convergen las placas tectónicas de Nazca, Sudamericana y del Caribe.

### Zonas geotérmicas conocidas

| Zona | Departamento | Tipo | Temperatura estimada |
|------|-------------|------|---------------------|
| Nevado del Ruiz – Macizo Volcánico | Caldas/Tolima | Volcánico-hidrotermal | >200 °C |
| Chiles – Cerro Negro | Nariño | Volcánico-hidrotermal | >200 °C |
| Azufral | Nariño | Volcánico-hidrotermal | >200 °C |
| Paipa – Iza | Boyacá | Hidrotermal no volcánico | 150–200 °C |
| Chocontá – Machetá | Cundinamarca | Indicadores ASTER | En evaluación |
| Coconucos | Cauca | Volcánico-hidrotermal | 150–200 °C |
| Santa Rosa de Cabal | Risaralda | Hidrotermal | ~150 °C |

### Iniciativas nacionales

- **ISAGEN** realizó estudios de factibilidad en el Nevado del Ruiz (proyecto más avanzado del país).
- El **Servicio Geológico Colombiano (SGC)** mantiene un programa de vigilancia volcánica y estudios geotérmicos.
- La **UPME** (Unidad de Planeación Minero-Energética) incluye la geotermia en el plan de diversificación de la matriz energética.
- Colombia aún no tiene una planta geotérmica operativa, pero el potencial estimado supera los **2,000 MW** según estudios del SGC.

## 5. Rol del Modelo CNN en la Exploración Geotérmica

### 5.1 La exploración geotérmica tradicional

La exploración geotérmica convencional sigue una secuencia costosa y prolongada:

```
Fase 1: Reconocimiento regional    (~$50K–200K, meses)
  ├── Revisión bibliográfica
  ├── Mapeo geológico superficial
  └── Análisis de imágenes satelitales ← AQUÍ ACTÚA LA CNN

Fase 2: Exploración de superficie   (~$200K–1M, 1–2 años)
  ├── Geoquímica de aguas y gases
  ├── Geofísica (gravimetría, magnetometría, MT)
  └── Estudios estructurales detallados

Fase 3: Exploración profunda        (~$5M–20M, 2–3 años)
  ├── Pozos exploratorios (>1 km)
  ├── Pruebas de flujo y temperatura
  └── Modelamiento del reservorio

Fase 4: Desarrollo y producción     (~$50M–200M)
  ├── Pozos de producción e inyección
  ├── Planta de generación
  └── Conexión a red eléctrica
```

### 5.2 ¿Dónde encaja nuestra CNN?

Nuestro modelo CNN opera en la **Fase 1 (Reconocimiento regional)** como una herramienta de **screening automatizado**:

```
┌─────────────────────────────────────────────────┐
│            Territorio de Colombia               │
│          (~1.14 millones de km²)                │
│                                                 │
│   CNN + ASTER ──► Mapa de probabilidades        │
│                   de potencial geotérmico       │
│                                                 │
│   Resultado: Zonas priorizadas para             │
│   inversión en exploración detallada            │
└─────────────────────────────────────────────────┘
```

**Lo que hace la CNN:**
1. Recibe una imagen ASTER (14 bandas) de cualquier punto de Colombia.
2. Analiza patrones espectrales asociados a alteración hidrotermal y anomalías térmicas.
3. Produce una **probabilidad (0–100%)** de que la zona tenga potencial geotérmico.

**Lo que NO hace la CNN:**
- No confirma la existencia de un recurso explotable.
- No reemplaza la exploración geológica de campo.
- No estima temperaturas de reservorio ni profundidades.
- No evalúa viabilidad económica.

### 5.3 Valor para el gobierno y tomadores de decisiones

| Sin CNN (método tradicional) | Con CNN (herramienta de screening) |
|-----|------|
| Revisión manual de imágenes satelitales por expertos | Análisis automatizado de cualquier punto en segundos |
| Cobertura limitada por tiempo y presupuesto | Cobertura potencial de todo el territorio nacional |
| Sesgo hacia zonas ya conocidas (volcanes, termales) | Capacidad de descubrir zonas no obvias |
| Meses de trabajo para priorizar zonas | Priorización inmediata basada en datos |

**Escenario de uso:**
> El Servicio Geológico Colombiano quiere identificar nuevas zonas con potencial geotérmico fuera de las áreas volcánicas conocidas. Usando la CNN, puede escanear sistemáticamente regiones del territorio colombiano y obtener un mapa de probabilidades que le permita **enfocar los recursos limitados de exploración en las zonas más prometedoras**.

### 5.4 Logros del modelo v2 (versión actual) y limitaciones residuales

El modelo v1 era un **prototipo académico** con limitaciones significativas (accuracy 68.43%, recall 48.10%, solo 85 imágenes). Todas fueron abordadas en la versión v2:

| Aspecto | v1 (Baseline) | v2 (Actual) |
|---------|---------------|--------------|
| **Accuracy** | 68.43% | **91.45%** |
| **Recall** | 48.10% | **86.05%** |
| **F1-Score** | 61.77% | **91.61%** |
| **ROC AUC** | 0.8198 | **0.9830** |
| **Dataset** | 85 imágenes (5 bandas) | 200 imágenes (7 bandas) |
| **Filtrado NoData** | No | Sí |
| **Split** | Aleatorio (data leakage) | GroupShuffleSplit |

**Limitaciones residuales** (alcance académico del proyecto):

- **Generalización:** Entrenado solo en Colombia; puede no funcionar en otros contextos geológicos.
- **Resolución temporal:** ASTER GED es un producto promediado; no captura variaciones temporales.
- **Entrenamiento CPU:** Limitado a 22 épocas con CosineDecay; GPU podría mejorar más.
- **Validación por zona específica:** Predicciones individuales por zona pendientes de re-ejecutar con v2.

Estas limitaciones están documentadas en detalle en [PREDICCIONES_PRUEBA.md](PREDICCIONES_PRUEBA.md) y [CHANGELOG_V2.md](CHANGELOG_V2.md).

## 6. Resumen Visual

```
ENERGÍA GEOTÉRMICA
       │
       ├── ¿Qué necesita?
       │     └── Calor + Reservorio + Fluido (triángulo geotérmico)
       │
       ├── ¿Se necesitan aguas termales?
       │     ├── Hidrotermales: SÍ (agua natural caliente)
       │     ├── EGS: NO (se inyecta agua en roca caliente seca)
       │     └── Uso directo/GSHP: NO (temperatura estable del subsuelo)
       │
       ├── ¿Cómo se detecta desde satélite?
       │     ├── Anomalías térmicas (TIR)
       │     ├── Alteración hidrotermal (SWIR)
       │     └── Composición mineral (SWIR + TIR)
       │
       ├── ¿Qué hace nuestra CNN?
       │     ├── Analiza imágenes ASTER (7 bandas: 5 TIR emisividad + Temperatura + NDVI)
       │     ├── Identifica patrones térmicos y espectrales de potencial geotérmico
       │     └── Produce probabilidad (0–100%) por zona — v2: 91.45% accuracy
       │
       └── ¿Para qué sirve?
             ├── Screening automatizado (Fase 1 exploración)
             ├── Priorización de zonas para inversión
             └── Herramienta de apoyo a decisiones gubernamentales
```

---

**Referencias:**
- DiPippo, R. (2015). *Geothermal Power Plants: Principles, Applications, Case Studies and Environmental Impact*. 4th Ed. Elsevier.
- Servicio Geológico Colombiano. (2023). *Mapa de potencial geotérmico de Colombia*.
- Coolbaugh, M. et al. (2007). Detection of geothermal anomalies using ASTER thermal infrared data. *GRC Transactions*, 31.
- Vaughan, R.G. et al. (2005). Surface mineral mapping at Steamboat Springs, Nevada with multi-wavelength thermal infrared images. *Remote Sensing of Environment*, 99(1-2).
- NASA/METI/AIST/Japan Spacesystems. ASTER Global Emissivity Dataset (GED). DOI: 10.5067/COMMUNITY/ASTER_GED/AG100.003

---

*Documento creado: 18 de febrero de 2026*
*Proyecto de Grado — Universidad de San Buenaventura, Bogotá*
*Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano, Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín*
*Asesor: Prof. Yeison Eduardo Conejo Sandoval*
