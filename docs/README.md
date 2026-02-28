# Documentación del Proyecto — Geotermia Colombia CNN (v3)

Esta carpeta contiene toda la documentación técnica del proyecto CNN Geotermia Colombia.
**Versión actual: v3** (branch v3) — Dataset Región Andina: 2,019 imágenes base, 22,209 augmentadas.

## Índice de Documentos

| Documento | Descripción |
|-----------|-------------|
| [CONTEXTO_GEOTERMICO.md](CONTEXTO_GEOTERMICO.md) | Fundamentos de energía geotérmica, rol del modelo CNN y logros v2 |
| [RESUMEN_PROYECTO.md](RESUMEN_PROYECTO.md) | Vista general del proyecto, estado, bitácora cronológica y configuración |
| [CAMPOS_GEOTERMICOS_REGION_ANDINA.md](CAMPOS_GEOTERMICOS_REGION_ANDINA.md) | Catálogo de campos geotérmicos de Colombia, Ecuador, Perú y Chile |
| [MODELO_PREDICTIVO.md](MODELO_PREDICTIVO.md) | Documentación técnica completa del modelo CNN (arquitectura, pipeline, métricas) |
| [ANALISIS_ENTRENAMIENTO.md](ANALISIS_ENTRENAMIENTO.md) | Análisis del entrenamiento v2 (22 épocas) con comparativa v1 vs v2 |
| [CHANGELOG_V2.md](CHANGELOG_V2.md) | Auditoría profunda (28 bugs corregidos), mejoras implementadas y resultados v2 |
| [GUIA_PASO_A_PASO.md](GUIA_PASO_A_PASO.md) | Guía completa para reproducir el pipeline (2,019 imgs, 7 bandas, disco externo) |
| [PREDICCIONES_PRUEBA.md](PREDICCIONES_PRUEBA.md) | Predicciones baseline v1 por zona + comparativa de métricas v1 vs v2 |

## Orden de Lectura Recomendado

1. **CONTEXTO_GEOTERMICO.md** — Entender qué es la geotermia y por qué este proyecto es relevante
2. **CAMPOS_GEOTERMICOS_REGION_ANDINA.md** — Catálogo de campos y justificación de la expansión andina
3. **RESUMEN_PROYECTO.md** — Visión rápida del proyecto completo y estado actual (v3)
4. **MODELO_PREDICTIVO.md** — Documentación técnica detallada (arquitectura, bandas, métricas)
5. **ANALISIS_ENTRENAMIENTO.md** — Resultados del entrenamiento v2 y comparativa con v1
6. **GUIA_PASO_A_PASO.md** — Reproducir el pipeline desde cero o entrenar en otra máquina
7. **PREDICCIONES_PRUEBA.md** — Predicciones por zona (v1 baseline) y mejora cuantificada en v2
8. **CHANGELOG_V2.md** — Historial detallado de auditoría, correcciones y mejoras

## Archivos consolidados

Los siguientes archivos fueron absorbidos en la actualización a v2:

- `REGISTRO_PROCESO.md` → Integrado en **RESUMEN_PROYECTO.md** (sección Bitácora Cronológica)
- `MEJORAS_MODELO.md` → Integrado en **CHANGELOG_V2.md** (sección Mejoras Implementadas)

---

**Universidad de San Buenaventura — Bogotá**  
Proyecto de Grado 2025-2026
