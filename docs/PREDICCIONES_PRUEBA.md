# Predicciones de Prueba — Modelo v1 (Baseline)

> **Fecha:** 18 de febrero de 2026
> **Modelo:** `geotermia_cnn_custom_best.keras` (~57 MB)
> **Entrenamiento:** 2,635 imagenes, mejor epoca 8/23
> **Metricas test:** Accuracy 68.43% · Precision 86.32% · ROC AUC 0.8198 · F1 61.77%
> **Hardware:** Intel i5-10300H (CPU) · 12 GB RAM

---

## Resumen de Predicciones

| # | Ubicacion | Lat | Lon | Prob. | Resultado | Confianza | Zona cercana | Dist. |
|---|-----------|-----|-----|-------|-----------|-----------|--------------|-------|
| 1 | Nevado del Ruiz (zona conocida) | 4.8951 | -75.3222 | **40.1%** | BAJO POTENCIAL | Media-Baja | Nevado del Ruiz | 0.0 km |
| 2 | Paipa-Iza (campo geotermico) | 5.7781 | -73.1124 | **29.1%** | BAJO POTENCIAL | Baja | Paipa-Iza | 0.0 km |
| 3 | Bogota (control negativo) | 4.7110 | -74.0721 | **26.2%** | BAJO POTENCIAL | Baja | Nevado del Ruiz | 140.3 km |
| 4 | Choconta, Cundinamarca | 5.1460 | -73.6870 | **44.0%** | BAJO POTENCIAL | Media-Baja | Paipa-Iza | 94.8 km |

---

## Analisis Detallado por Ubicacion

### 1. Nevado del Ruiz (zona geotermico conocida)

- **Coordenadas:** 4.8951, -75.3222
- **Probabilidad:** 40.1%
- **Resultado:** BAJO POTENCIAL (Confianza Media-Baja)
- **Imagen:** 112x112 px, 38.0 KB
- **Tiempos:** descarga 0.7s | prediccion 0.421s | total 1.2s
- **Nota:** Se detectaron valores NoData (-9999) en las bandas ASTER, lo que afecta
  la calidad del analisis. Esto explica la probabilidad mas baja de lo esperado
  para una zona volcanica activa conocida.

| Banda | Min | Max | Media | Desv. Est. |
|-------|-----|-----|-------|------------|
| B10 (8.3 um) | -9999.0 | 998.0 | 792.40 | 1155.71 |
| B11 (8.6 um) | -9999.0 | 1969.0 | 821.74 | 1158.66 |
| B12 (9.1 um) | -9999.0 | 2392.0 | 824.77 | 1159.70 |
| B13 (10.6 um) | -9999.0 | 2280.0 | 835.42 | 1160.63 |
| B14 (11.3 um) | -9999.0 | 978.0 | 798.17 | 1155.94 |

### 2. Paipa-Iza (campo geotermico)

- **Coordenadas:** 5.7781, -73.1124
- **Probabilidad:** 29.1%
- **Resultado:** BAJO POTENCIAL (Confianza Baja)
- **Imagen:** 112x113 px, 26.0 KB
- **Tiempos:** descarga 0.9s | prediccion 0.087s | total 1.0s
- **Nota:** Datos ASTER limpios (sin NoData). Paipa-Iza es un campo geotermico
  confirmado, pero su firma termica en emisividad TIR es mas sutil que la de zonas
  volcanicas. El modelo no lo identifica correctamente como positivo.

| Banda | Min | Max | Media | Desv. Est. |
|-------|-----|-----|-------|------------|
| B10 (8.3 um) | 933.0 | 983.0 | 971.91 | 5.32 |
| B11 (8.6 um) | 932.0 | 984.0 | 971.44 | 4.51 |
| B12 (9.1 um) | 922.0 | 985.0 | 970.83 | 4.97 |
| B13 (10.6 um) | 952.0 | 991.0 | 969.66 | 4.17 |
| B14 (11.3 um) | 932.0 | 990.0 | 967.98 | 5.15 |

### 3. Bogota (control negativo)

- **Coordenadas:** 4.7110, -74.0721
- **Probabilidad:** 26.2%
- **Resultado:** BAJO POTENCIAL (Confianza Baja) ✅
- **Imagen:** 112x112 px, 30.6 KB
- **Tiempos:** descarga 1.0s | prediccion 0.1s | total 1.1s
- **Nota:** Resultado correcto. Bogota no tiene potencial geotermico y el modelo
  predice correctamente una probabilidad baja. Se detectaron valores NoData en
  algunas bandas (posiblemente nubes/sombras).

| Banda | Min | Max | Media | Desv. Est. |
|-------|-----|-----|-------|------------|
| B10 (8.3 um) | -9999.0 | 987.0 | 955.04 | 366.29 |
| B11 (8.6 um) | -9999.0 | 987.0 | 954.27 | 366.22 |
| B12 (9.1 um) | -9999.0 | 986.0 | 951.39 | 366.17 |
| B13 (10.6 um) | -9999.0 | 991.0 | 960.26 | 366.37 |
| B14 (11.3 um) | -9999.0 | 990.0 | 957.71 | 366.33 |

### 4. Choconta, Cundinamarca

- **Coordenadas:** 5.1460, -73.6870
- **Probabilidad:** 44.0%
- **Resultado:** BAJO POTENCIAL (Confianza Media-Baja)
- **Imagen:** 112x112 px, 28.1 KB
- **Tiempos:** descarga 1.0s | prediccion 0.09s | total 1.1s
- **Nota:** Zona intermedia sin actividad geotermica conocida. La probabilidad
  es mayor que Bogota, posiblemente por la mayor variabilidad termica de la
  zona montanosa andina. Datos ASTER limpios.

| Banda | Min | Max | Media | Desv. Est. |
|-------|-----|-----|-------|------------|
| B10 (8.3 um) | 908.0 | 986.0 | 970.94 | 8.00 |
| B11 (8.6 um) | 937.0 | 986.0 | 970.05 | 5.64 |
| B12 (9.1 um) | 935.0 | 984.0 | 969.78 | 5.43 |
| B13 (10.6 um) | 941.0 | 984.0 | 969.81 | 5.21 |
| B14 (11.3 um) | 913.0 | 984.0 | 968.09 | 7.98 |

---

## Observaciones y Conclusiones

### Hallazgos clave

1. **Todas las predicciones resultaron BAJO POTENCIAL (< 50%)**, incluyendo zonas
   geotermicas conocidas. Esto indica que el modelo v1 tiene un recall bajo
   (48.10% confirmado en evaluacion).

2. **Problema de NoData:** Las imagenes ASTER del Nevado del Ruiz y Bogota contienen
   valores -9999 (NoData) que no se filtran antes de la normalizacion. Esto distorsiona
   significativamente las estadisticas y la prediccion.

3. **Orden relativo parcialmente correcto:** Choconta (44.0%) > Ruiz (40.1%) > Paipa (29.1%)
   ≈ Bogota (26.2%). El modelo diferencia ligeramente entre zonas, pero no alcanza
   el umbral de 50%.

4. **Tiempos de respuesta excelentes:** ~1 segundo total por prediccion (descarga + inferencia).
   La primera prediccion es mas lenta (0.4s) por carga del modelo en memoria.

### Mejoras necesarias para modelo v2

| Mejora | Impacto esperado | Prioridad |
|--------|------------------|-----------|
| **Filtrar valores NoData (-9999)** antes de normalizar | Alto — corrige distorsion | Critica |
| **Mas datos de entrenamiento** (> 500 imagenes originales) | Alto — mejora generalizacion | Alta |
| **Entrenamiento en GPU** (mas epocas, mejor convergencia) | Medio-Alto | Alta |
| **Data augmentation mas agresiva** | Medio | Media |
| **Fine-tuning de hiperparametros** (learning rate, dropout) | Medio | Media |
| **Transfer learning** (ResNet50 pre-entrenado) | Alto — mejor feature extraction | Media |
| **Validacion cruzada** (k-fold) | Medio — evaluacion mas robusta | Baja |

### Linea base establecida

Estos resultados sirven como **baseline (v1)** para comparar con versiones futuras
del modelo entrenadas con mas datos y en mejor hardware. El objetivo para v2 es:

- Accuracy > 80%
- Recall > 70% (detectar correctamente zonas geotermicas)
- Probabilidad > 50% para Nevado del Ruiz y Paipa-Iza
- F1-Score > 75%

---

*Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín*
*Asesor: Prof. Yeison Eduardo Conejo Sandoval*
*Universidad de San Buenaventura — Bogotá — 2025-2026*
