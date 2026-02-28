# Campos Geotérmicos y Zonas Volcánicas de la Región Andina

> **Documento de referencia para el dataset de entrenamiento CNN**  
> Universidad de San Buenaventura — Bogotá, 2025-2026  
> Última actualización: Febrero 2026

---

## Justificación del uso de zonas extra-colombianas

El título de la tesis se centra en la **identificación de zonas con potencial geotérmico
en Colombia**. Sin embargo, el territorio colombiano por sí solo ofrece un número limitado
de zonas geotérmicas confirmadas (~111 positivas + ~89 de control = ~200 imágenes), lo
cual es **insuficiente para entrenar una red neuronal convolucional robusta**.

Para resolver esta limitación, se amplió el dataset de entrenamiento con zonas de la
**Región Andina** (Ecuador, Perú y Chile), ya que comparten el mismo contexto geológico:
vulcanismo del Cinturón de Fuego del Pacífico y similares firmas espectrales ASTER GED.
Esto permite alcanzar **~2000 imágenes** con un balance adecuado entre clases.

> **Nota**: La aplicación web (Streamlit) y las conclusiones del trabajo se limitan
> exclusivamente a Colombia. Las zonas andinas se utilizan únicamente para robustecer
> el entrenamiento del modelo.

---

## 1. Resumen

Este documento cataloga los **campos geotérmicos confirmados**, **zonas volcánicas activas**
y **manifestaciones hidrotermales** de la **Región Andina** (Colombia, Ecuador, Perú y Chile)
que se utilizan como fuentes de muestras positivas (label=1) para el entrenamiento de la
CNN de identificación de potencial geotérmico.

Cada zona incluye:
- Coordenadas [longitud, latitud]
- Tipo de manifestación geotérmica
- Estado de exploración/explotación
- Referencia bibliográfica

---

## 2. Colombia 🇨🇴

Colombia posee recursos geotérmicos asociados principalmente al vulcanismo de la
Cordillera Central de los Andes. El Servicio Geológico Colombiano (SGC) ha identificado
varias zonas con potencial.

### 2.1. Campos Geotérmicos Confirmados

| Zona | Coordenadas [lon, lat] | Tipo | Estado | Referencia |
|------|------------------------|------|--------|------------|
| **Nevado del Ruiz** | [-75.322, 4.895] | Volcán activo, campo geotérmico de alta entalpía | Exploración avanzada (ISAGEN/SGC) | SGC, 2019 |
| **Volcán Puracé** | [-76.404, 2.321] | Campo geotérmico confirmado, fumarolas | Exploración por SGC | SGC, 2018 |
| **Paipa-Iza** | [-73.112, 5.778] | Campo de baja entalpía, aguas termales | Exploración preliminar | Alfaro et al., 2017 |

### 2.2. Volcanes Activos con Manifestaciones Hidrotermales

| Zona | Coordenadas [lon, lat] | Tipo | Última erupción |
|------|------------------------|------|-----------------|
| **Volcán Galeras** | [-77.360, 1.220] | Estratovolcán, fumarolas permanentes | 2010 |
| **Volcán Cumbal** | [-77.880, 0.950] | Fumarolas activas | Histórica |
| **Volcán Azufral** | [-77.680, 1.085] | Lago cratérico sulfuroso, fumarolas | Holoceno |
| **Volcán Cerro Machín** | [-75.390, 4.490] | Actividad hidrotermal, alto riesgo | ~1200 d.C. |
| **Nevado del Tolima** | [-75.330, 4.660] | Nevado con actividad fumarólica | Holoceno |
| **Volcán Sotará** | [-76.590, 2.110] | Fumarolas | Holoceno |
| **Volcán Doña Juana** | [-76.940, 1.500] | Fumarolas activas | 1906 |
| **Nevado del Huila** | [-76.030, 2.940] | Volcán activo | 2012 |
| **Volcán Chiles** | [-77.940, 0.820] | Actividad sísmica y fumarólica | 1936 (compartido con Ecuador) |
| **Cerro Negro de Mayasquer** | [-77.960, 0.830] | Fumarolas | Holoceno |
| **Cerro Bravo** | [-75.300, 5.090] | Actividad fumarólica | ~1720 |
| **Nevado de Santa Isabel** | [-75.370, 4.815] | Retroceso glaciar, termalismo | N/A |
| **Romeral** | [-75.364, 4.960] | Actividad fumarólica | Holoceno |

### 2.3. Zonas Termales Confirmadas

| Zona | Coordenadas [lon, lat] | Tipo |
|------|------------------------|------|
| **Termales de Manizales** | [-75.520, 5.070] | Aguas termales asociadas al complejo Ruiz |
| **Coconucos** | [-76.385, 2.430] | Sistema termal complejo |
| **Santa Rosa de Cabal** | [-75.620, 4.870] | Termales |
| **Herveo** | [-75.130, 4.880] | Termales |
| **San Vicente Ferrer** | [-73.420, 6.880] | Termales de Santander |
| **Tabio** | [-74.090, 4.920] | Termales |
| **Choachí** | [-73.920, 4.530] | Termales |
| **Rivera (Huila)** | [-75.260, 2.780] | Termales |
| **Guadalupe (Santander)** | [-73.410, 6.250] | Termales |
| **San Agustín (Huila)** | [-76.270, 1.880] | Termales |
| **Pitalito** | [-76.040, 1.860] | Termales |

---

## 3. Ecuador 🇪🇨

Ecuador posee una intensa actividad volcánica a lo largo de la "Avenida de los Volcanes"
(Cordillera Occidental y Real). Tiene ~30 volcanes activos o potencialmente activos.

### 3.1. Campos Geotérmicos Identificados

| Zona | Coordenadas [lon, lat] | Tipo | Estado | Referencia |
|------|------------------------|------|--------|------------|
| **Tufiño-Chiles-Cerro Negro** | [-77.930, 0.825] | Campo binacional Colombia-Ecuador, alta entalpía | Exploración avanzada (CELEC EP / ISAGEN) | Rueda, 2015 |
| **Chalupas** | [-78.340, -0.780] | Caldera con potencial de alta entalpía | Prefactibilidad | CELEC EP, 2013 |
| **Chachimbiro** | [-78.290, 0.460] | Sistema hidrotermal, aguas termales | Exploración | INER, 2014 |
| **Baños de Agua Santa** (Tungurahua) | [-78.420, -1.400] | Aguas termales sulfurosas calientes | Conocido/turístico | --- |
| **El Placer** (Chimborazo) | [-78.600, -1.500] | Manifestaciones termales | Identificado | --- |
| **Papallacta** | [-78.140, -0.370] | Aguas termales de alta temperatura | Conocido/turístico | --- |

### 3.2. Volcanes Activos con Manifestaciones Hidrotermales

| Zona | Coordenadas [lon, lat] | Tipo | Última erupción |
|------|------------------------|------|-----------------|
| **Cotopaxi** | [-78.436, -0.677] | Estratovolcán activo, fumarolas | 2023 |
| **Tungurahua** | [-78.442, -1.467] | Estratovolcán activo | 2018 |
| **Guagua Pichincha** | [-78.598, -0.171] | Fumarolas activas | 2004 |
| **Reventador** | [-77.650, -0.007] | Volcán muy activo, fumarolas | 2021 (intermitente) |
| **Sangay** | [-78.340, -2.030] | Actividad permanente | 2020 (intermitente) |
| **Cayambe** | [-77.986, 0.029] | Fumarolas cumbres | 1786 |
| **Antisana** | [-78.140, -0.481] | Fumarolas | 1802 |
| **Chimborazo** | [-78.817, -1.469] | Volcán más alto de Ecuador, fumarolas | ~640 d.C. |
| **Quilotoa** | [-78.900, -0.850] | Caldera con lago, actividad termal | 1797 |
| **Illiniza** | [-78.714, -0.659] | Fumarolas | Holoceno |
| **Atacazo** | [-78.617, -0.353] | Sistema volcánico con termalismo | 320 a.C. |
| **Soche** | [-77.580, 0.552] | Volcán fronterizo | 6650 a.C. |
| **Pululagua** | [-78.463, 0.038] | Caldera con fumarolas | 467 a.C. |
| **Cuicocha** | [-78.364, 0.308] | Caldera con lago y fumarolas | 950 a.C. |
| **Imbabura** | [-78.180, 0.260] | Estratovolcán | Pleistoceno |
| **Mojanda** | [-78.270, 0.130] | Complejo de calderas | Holoceno |
| **Cotacachi** | [-78.349, 0.361] | Estratovolcán | Holoceno |

### 3.3. Zonas Termales Adicionales

| Zona | Coordenadas [lon, lat] | Tipo |
|------|------------------------|------|
| **Termas de Oyacachi** | [-78.070, -0.210] | Aguas termales volcánicas |
| **Termas de Nangulví** (Cotacachi) | [-78.500, 0.330] | Termales |
| **Termas de Guapán** | [-78.900, -2.730] | Termales |
| **Baños de San Vicente** | [-80.230, -2.220] | Termales costeras volcánicas |

---

## 4. Perú 🇵🇪

Perú tiene la Zona Volcánica Central (ZVC) en el sur, con ~16 volcanes activos o
potencialmente activos y múltiples campos geotérmicos de alta entalpía.

### 4.1. Campos Geotérmicos Confirmados/En Exploración

| Zona | Coordenadas [lon, lat] | Tipo | Estado | Referencia |
|------|------------------------|------|--------|------------|
| **Calientes** (Tacna) | [-69.900, -17.550] | Campo de alta entalpía (~150°C) | Exploración (INGEMMET) | Cruz & Vargas, 2016 |
| **Borateras** (Tacna) | [-69.830, -17.480] | Fumarolas, aguas termales con boro | Exploración | INGEMMET, 2014 |
| **Tutupaca** | [-70.358, -17.025] | Actividad fumarólica y termal | Exploración | JICA/INGEMMET |
| **Ubinas** | [-70.903, -16.355] | Volcán más activo de Perú, fumarolas | Monitoreo permanente | IGP/INGEMMET |
| **Salinas-Chivay** (Colca) | [-71.600, -15.630] | Aguas termales de alta temperatura | Identificado | --- |
| **Río Jesús** | [-70.200, -16.350] | Sistema termal volcánico | Identificado | --- |

### 4.2. Volcanes Activos con Manifestaciones Hidrotermales

| Zona | Coordenadas [lon, lat] | Tipo | Última erupción |
|------|------------------------|------|-----------------|
| **Misti** | [-71.409, -16.294] | Estratovolcán activo, fumarolas | 1985 |
| **Ubinas** | [-70.903, -16.355] | Volcán más activo del Perú | 2019 |
| **Sabancaya** | [-71.850, -15.780] | Erupción frecuente | 2017 (intermitente) |
| **Chachani** | [-71.530, -16.191] | Fumarolas de cumbre | Holoceno |
| **Huaynaputina** | [-70.850, -16.608] | Erupciones históricas catastróficas | 1600 (mayor del hemisferio sur) |
| **Ticsani** | [-70.595, -16.755] | Fumarolas y termalismo | Holoceno |
| **Yucamane** | [-70.200, -17.180] | Actividad fumarólica | 1902 |
| **Tutupaca** | [-70.358, -17.025] | Fumarolas activas | Holoceno |
| **Casiri** | [-69.813, -17.470] | Fumarolas | Holoceno |
| **Coropuna** | [-72.650, -15.520] | Volcán nevado con termalismo | Holoceno |
| **Ampato** | [-71.883, -15.817] | Fumarolas | Holoceno |
| **Sara Sara** | [-73.450, -15.330] | Fumarolas y termalismo | Holoceno |
| **Hualca Hualca** | [-71.883, -15.817] | Actividad fumarólica | Holoceno |

### 4.3. Zonas Termales Confirmadas

| Zona | Coordenadas [lon, lat] | Tipo |
|------|------------------------|------|
| **Baños del Inca** (Cajamarca) | [-78.470, -7.160] | Aguas termales históricas |
| **Churín** (Lima) | [-76.870, -10.820] | Aguas termales sulfurosas |
| **Calientes (Candarave)** | [-70.260, -17.290] | Aguas termales volcánicas |
| **La Calera** (Chivay, Colca) | [-71.600, -15.630] | Termales |
| **Yura** (Arequipa) | [-71.680, -16.250] | Termales |
| **Jesús** (Arequipa) | [-71.460, -16.480] | Termales |

---

## 5. Chile 🇨🇱

Chile posee el mayor potencial geotérmico de Sudamérica (estimado 6,000-112,000 MW).
Tiene la **primera y única planta geotérmica operativa de Sudamérica** (Cerro Pabellón,
48 MW, inaugurada 2017). Los campos se distribuyen en el norte (Zona Volcánica Central)
y el sur (Zona Volcánica Sur).

### 5.1. Campos Geotérmicos Confirmados/En Explotación

| Zona | Coordenadas [lon, lat] | Tipo | Estado | Referencia |
|------|------------------------|------|--------|------------|
| **Cerro Pabellón (Apacheta)** | [-68.150, -21.850] | Planta geotérmica operativa 48 MW | **En operación (Enel/ENAP)** — la más alta del mundo (4,500 msnm) | Enel Chile, 2017 |
| **El Tatio** | [-68.010, -22.332] | Mayor campo de géiseres del hemisferio sur, 120-170 MW térmicos | Exploración (concesión otorgada 2008) | Lahsen, 1982; Munoz-Saez et al., 2018 |
| **Puchuldiza-Tuja** | [-69.000, -19.417] | Géiseres, fumarolas, 33 MW térmicos | Exploración (120-180 MW potencial eléctrico) | Trujillo, 1978; Bona & Coviello, 2016 |
| **Calabozos** | [-70.500, -35.550] | Caldera con potencial de alta entalpía | Exploración | Lahsen, 2005 |
| **Nevados de Chillán** | [-71.370, -36.860] | Volcán activo, campo geotérmico | Exploración avanzada | Lahsen, 2005 |

### 5.2. Volcanes Activos y Campos Geotérmicos del Norte (Zona Volcánica Central)

| Zona | Coordenadas [lon, lat] | Tipo | Última erupción |
|------|------------------------|------|-----------------|
| **Volcán Láscar** | [-67.730, -23.370] | Volcán más activo de Chile norte | 1993 |
| **Volcán Putana** | [-67.850, -22.570] | Fumarolas permanentes | Holoceno |
| **Volcán Ollagüe** | [-68.180, -21.300] | Fumarolas activas | Holoceno |
| **Volcán San Pedro** | [-68.400, -21.880] | Fumarolas | 1960 |
| **Volcán Irruputuncu** | [-68.550, -20.730] | Fumarolas activas | 1995 |
| **Volcán Isluga** | [-68.830, -19.150] | Fumarolas | 1913 |
| **Volcán Guallatiri** | [-69.090, -18.420] | Fumarolas intensas | 1960 |
| **Volcán Tacora** | [-69.770, -17.720] | Fumarolas | Holoceno |
| **Volcán Parinacota** | [-69.140, -18.170] | Estratovolcán | Holoceno |
| **Volcán Licancabur** | [-67.883, -22.833] | Fumarolas | Holoceno |
| **Surire** | [-69.050, -18.850] | Campo geotermal (salar con fumarolas) | --- |

### 5.3. Volcanes Activos del Sur (Zona Volcánica Sur)

| Zona | Coordenadas [lon, lat] | Tipo | Última erupción |
|------|------------------------|------|-----------------|
| **Volcán Tolhuaca** | [-71.650, -38.320] | Campo geotérmico en exploración | 1956 |
| **Cordón Caulle** | [-71.750, -40.520] | Fisura volcánica con potencial geotérmico | 2011 |
| **Volcán Villarrica** | [-71.930, -39.420] | Volcán muy activo, lago de lava | 2015 |
| **Volcán Llaima** | [-71.730, -38.690] | Volcán activo | 2009 |
| **Volcán Copahue** | [-71.170, -37.850] | Fumarolas y aguas termales | 2016 |
| **Volcán Antuco** | [-71.349, -37.406] | Fumarolas | 1869 |
| **Volcán Callaqui** | [-71.449, -37.923] | Fumarolas activas | 1966 |

### 5.4. Zonas Termales Confirmadas

| Zona | Coordenadas [lon, lat] | Tipo |
|------|------------------------|------|
| **Termas de Polloquere** | [-69.050, -18.850] | Aguas termales del Salar de Surire |
| **Termas de Jurasi** (Putre) | [-69.510, -18.180] | Termales volcánicas |
| **Termas de Colina** (Santiago) | [-70.300, -33.300] | Termales andinas |
| **Termas de Chillán** | [-71.400, -36.900] | Termales volcánicas |
| **Termas de Huife** (Pucón) | [-71.750, -39.350] | Termales volcánicas |
| **Termas de Menetúe** (Villarrica) | [-71.850, -39.330] | Termales |
| **Termas Geométricas** (Coñaripe) | [-71.900, -39.600] | Termales volcánicas |
| **Termas del Flaco** (O'Higgins) | [-70.420, -34.970] | Termales andinas |

---

## 6. Zonas de Control (Negativas, label=0) — Nuevos Países

Para cada país se definen zonas **sin actividad volcánica ni geotermal conocida**,
en regiones geológicamente estables, llanuras costeras o cuencas sedimentarias.

### 6.1. Ecuador — Zonas de Control

| Zona | Coordenadas [lon, lat] | Justificación |
|------|------------------------|---------------|
| **Guayaquil** | [-79.897, -2.170] | Llanura costera, sin vulcanismo |
| **Machala** | [-79.960, -3.260] | Costa sur, cuenca sedimentaria |
| **Esmeraldas** | [-79.650, 0.960] | Costa norte, sin actividad volcánica |
| **Santo Domingo** | [-79.170, -0.250] | Llanura costera occidental |
| **Quevedo** | [-79.460, -1.020] | Cuenca del río Guayas |
| **Manta** | [-80.733, -0.950] | Costa Pacífico |
| **Portoviejo** | [-80.450, -1.050] | Costa Pacífica interior |
| **Lago Agrio** | [-76.878, 0.084] | Amazonía ecuatoriana |
| **Puyo** | [-77.990, -1.490] | Amazonia occidental |
| **Tena** | [-77.810, -1.000] | Amazonia interior |
| **Zamora** | [-78.950, -4.070] | Amazonía sur |
| **Loja sur** | [-79.220, -4.050] | Andes meridionales estables |
| **Santa Elena** | [-80.860, -2.230] | Península costera |
| **Durán** | [-79.830, -2.170] | Llanura costera |
| **Milagro** | [-79.590, -2.130] | Cuenca del Guayas |
| **Babahoyo** | [-79.530, -1.800] | Cuenca sedimentaria |
| **Vinces** | [-79.750, -1.560] | Cuenca del río |
| **Ambato control** | [-76.530, -1.250] | Zona oriental no volcánica |
| **Cuenca control** | [-79.020, -2.920] | Sur andino estable |
| **Riobamba control** | [-79.050, -1.680] | Control alejado de volcanes |

### 6.2. Perú — Zonas de Control

| Zona | Coordenadas [lon, lat] | Justificación |
|------|------------------------|---------------|
| **Lima** | [-77.028, -12.046] | Costa central, sin vulcanismo |
| **Piura** | [-80.630, -5.190] | Costa norte, desierto |
| **Chiclayo** | [-79.840, -6.770] | Llanura costera |
| **Trujillo** | [-79.030, -8.110] | Valle costero |
| **Ica** | [-75.730, -14.060] | Desierto costero |
| **Nasca** | [-75.120, -14.830] | Desierto |
| **Chimbote** | [-78.530, -9.070] | Puerto costero |
| **Huancayo** | [-75.210, -12.070] | Sierra central, sin vulcanismo |
| **Cusco** | [-71.970, -13.520] | Sierra sureste, sin vulcanismo activo |
| **Puno** | [-70.020, -15.840] | Altiplano del Titicaca |
| **Juliaca** | [-70.130, -15.500] | Altiplano |
| **Iquitos** | [-73.250, -3.750] | Amazonía peruana |
| **Pucallpa** | [-74.530, -8.380] | Amazonía central |
| **Tarapoto** | [-76.370, -6.490] | Selva alta |
| **Tumbes** | [-80.450, -3.570] | Costa extremo norte |
| **Tacna control** | [-70.250, -18.010] | Control costa sur |
| **Arequipa control** | [-71.540, -16.409] | Control urbano (lejos de volcanes) |
| **Ayacucho** | [-74.220, -13.160] | Sierra sin vulcanismo |
| **Huaraz** | [-77.528, -9.530] | Sierra norte (glaciar sin termalismo) |
| **Cajamarca control** | [-78.520, -7.160] | Sierra norte estable |

### 6.3. Chile — Zonas de Control

| Zona | Coordenadas [lon, lat] | Justificación |
|------|------------------------|---------------|
| **Santiago** | [-70.650, -33.448] | Valle central, zona urbana |
| **Valparaíso** | [-71.620, -33.047] | Costa central |
| **Concepción** | [-73.050, -36.830] | Costa centro-sur |
| **La Serena** | [-71.250, -29.900] | Costa norte (sin vulcanismo — gap volcánico) |
| **Copiapó** | [-70.330, -27.370] | Desierto de Atacama |
| **Antofagasta ciudad** | [-70.400, -23.650] | Costa desierto, lejos de volcanes |
| **Arica** | [-70.311, -18.475] | Costa extremo norte |
| **Temuco** | [-72.640, -38.740] | Valle central sur |
| **Osorno ciudad** | [-73.150, -40.573] | Valle central sur |
| **Valdivia** | [-73.240, -39.810] | Costa sur |
| **Rancagua** | [-70.740, -34.170] | Valle central |
| **Talca** | [-71.650, -35.430] | Valle central |
| **Chillán ciudad** | [-72.100, -36.600] | Valle central (lejos del volcán) |
| **Punta Arenas** | [-70.917, -53.150] | Patagonia (sin vulcanismo) |
| **Puerto Montt** | [-72.940, -41.470] | Sur |
| **Iquique** | [-70.130, -20.210] | Costa norte |
| **Calama control** | [-68.920, -22.450] | Desierto interior (lejos de géiseres) |
| **San Fernando** | [-71.000, -34.580] | Valle central |
| **Curicó** | [-71.240, -34.980] | Valle central |
| **Los Ángeles** | [-72.350, -37.470] | Valle del Biobío |

---

## 7. Resumen Cuantitativo

### Muestras Positivas (label=1)

| País | Campos geotérmicos | Volcanes activos | Zonas termales | **Total zonas** |
|------|-------------------|-----------------|----------------|----------------|
| Colombia | 3 | 13 | 11 | **~111** (con subdivisiones) |
| Ecuador | 6 | 17 | 4 | **~170** (con subdivisiones ×5-7 por zona) |
| Perú | 6 | 13 | 6 | **~160** (con subdivisiones ×5-7 por zona) |
| Chile Norte | 5 + 11 volcanes | 11 | 4 | **~200** (con subdivisiones ×5-7 por zona) |
| Chile Sur | 7 volcanes | 7 | 4 | **~80** (con subdivisiones) |
| **TOTAL** | | | | **~720 positivas** |

### Muestras Negativas (label=0)

| País | Zonas de control |
|------|-----------------|
| Colombia | 89 (existentes) |
| Ecuador | 20 |
| Perú | 20 |
| Chile | 20 |
| **TOTAL** | **~149** → expandir a **~720 con subdivisiones** |

### Dataset Final (Real v3)

| Métrica | Valor |
|---------|-------|
| **Imágenes originales** | **2,019** (997 positivas + 1,022 negativas) |
| **Aumentadas (×10)** | **22,209** (10,967 pos + 11,242 neg) |
| **Espacio en disco (raw)** | ~115 MB |
| **Espacio en disco (augmented)** | ~7.9 GB |
| **Espacio en disco (processed)** | ~29.8 GB |
| **Balance** | 49.4% pos / 50.6% neg (casi perfecto) |
| **Grupos geográficos** | 4,038 (para GroupShuffleSplit anti-leakage) |

---

## 8. Criterios de Selección de Zonas

### 8.1. Para muestras POSITIVAS (label=1)
1. **Campo geotérmico confirmado**: Documentado por servicio geológico nacional  
2. **Volcán activo con fumarolas**: Actividad en el Holoceno (<11,700 años)  
3. **Manifestación hidrotermal**: Aguas termales con T > 40°C  
4. **Buffer de 5 km**: Cada punto se descarga con buffer de 5 km  
5. **Subdivisiones**: Center, N, S, E, W, NE, SW para maximizar cobertura  

### 8.2. Para muestras NEGATIVAS (label=0)
1. **Ausencia de vulcanismo**: Sin volcanes en radio de 50 km mínimo  
2. **Estabilidad geológica**: Cuencas sedimentarias, llanuras costeras  
3. **Diversidad geomorfológica**: Incluir costa, llanura, selva, altiplano  
4. **Distancia mínima entre puntos**: >20 km para evitar solapamiento de tiles  

---

## 9. Referencias Principales

1. SGC - Servicio Geológico Colombiano. (2019). *Mapa de amenaza volcánica del Volcán Nevado del Ruiz*.
2. Lahsen, A. (1982). Upper Cenozoic volcanism and tectonism in the Andes of northern Chile. *Earth-Science Reviews*, 18(3), 285–302.
3. Munoz-Saez, C., Manga, M., & Hurwitz, S. (2018). Hydrothermal discharge from the El Tatio basin, Atacama, Chile. *Journal of Volcanology and Geothermal Research*, 361, 25–35.
4. Bona, P., & Coviello, M. (2016). *Valoración y gobernanza de los proyectos geotérmicos en América del Sur*. CEPAL.
5. INGEMMET - Instituto Geológico, Minero y Metalúrgico del Perú. (2014). *Inventario de fuentes termales del Perú*.
6. Cruz, V., & Vargas, V. (2016). *Exploración geotérmica de la región de Tacna*. INGEMMET.
7. CELEC EP. (2013). *Proyecto Geotérmico Chachimbiro, Ecuador*.
8. Rueda, D. (2015). *Desarrollo del Proyecto Geotérmico Binacional Tufiño-Chiles*. CELEC EP.
9. Alfaro, C., et al. (2017). Geothermal potential of the Paipa volcano-hydrothermal system, Colombia. *Proceedings World Geothermal Congress*.
10. Siebert, L., & Simkin, T. (2002–present). *Volcanoes of the World*. Smithsonian Institution.

---

> **Nota**: Las coordenadas de las subdivisiones se generan automáticamente en
> `download_dataset.py` mediante `_generate_grid()`, que crea 9 tiles por zona base
> (center + N, S, E, W, NE, NW, SE, SW) con offsets de ±0.04° (~4 km).
