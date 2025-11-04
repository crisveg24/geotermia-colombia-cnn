# 🔬 RAMA DEVELOP - Entrenamiento en Máquina Externa

**Branch:** `develop`  
**Propósito:** Rama temporal para entrenar el modelo CNN en máquina con mejor hardware  
**Estado:** ✅ Configurada y lista para clonar

---

## ⚠️ IMPORTANTE

Esta es una **rama temporal** que se eliminará después de:
1. Completar el entrenamiento del modelo
2. Subir el modelo entrenado y resultados
3. Hacer merge a `main`

**No hacer commits directos a `main` desde la máquina de entrenamiento.**

---

## 🚀 INICIO RÁPIDO

### En la máquina de entrenamiento:

```bash
# 1. Clonar repositorio
git clone https://github.com/crisveg24/geotermia-colombia-cnn.git
cd geotermia-colombia-cnn

# 2. Cambiar a rama develop
git checkout develop

# 3. Ver la guía completa
# Lee: ENTRENAMIENTO_EXTERNO.md (paso a paso detallado)
```

---

## 📋 FLUJO DE TRABAJO

```
┌─────────────────────────────────────────────────────────────┐
│  MÁQUINA ORIGINAL                                           │
│  ├─ main branch                                             │
│  ├─ Documentación completa ✅                               │
│  ├─ Scripts listos ✅                                       │
│  ├─ 30 épocas entrenadas (parcial) ⚠️                      │
│  └─ Crear rama develop ✅                                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ git clone + checkout develop
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  MÁQUINA DE ENTRENAMIENTO (mejor hardware)                 │
│  ├─ Clonar develop branch                                   │
│  ├─ Configurar entorno Python                               │
│  ├─ Descargar datos (Google Earth Engine)                   │
│  ├─ Entrenar modelo completo (100 épocas) 🚀               │
│  ├─ Evaluar y generar visualizaciones                       │
│  ├─ Commit modelo entrenado                                 │
│  └─ Push a origin/develop                                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ git merge develop → main
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  MÁQUINA ORIGINAL                                           │
│  ├─ Pull develop                                            │
│  ├─ Revisar resultados                                      │
│  ├─ Merge develop → main                                    │
│  ├─ Push main                                               │
│  └─ Eliminar rama develop ✅                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 QUÉ ESTÁ INCLUIDO EN DEVELOP

### Scripts Completos ✅
```
scripts/
  ├── download_dataset.py         - Descarga imágenes ASTER
  ├── augment_full_dataset.py     - Genera 5,518 imágenes
  ├── prepare_dataset.py          - Prepara datos para training
  ├── train_model.py              - Entrena modelo CNN
  ├── evaluate_model.py           - Evalúa en test set
  └── visualize_results.py        - Genera gráficos
```

### Documentación ✅
```
ENTRENAMIENTO_EXTERNO.md     - Guía paso a paso completa
MONITOREO_ENTRENAMIENTO.md   - Cómo monitorear progreso
ANALISIS_ENTRENAMIENTO.md    - Análisis de métricas
REGISTRO_PROCESO.md          - Historial completo del proyecto
RESUMEN_EJECUTIVO.md         - Vista general
```

### Metadata (Ligero) ✅
```
data/raw/
  ├── dataset_images.csv       - Lista de imágenes y coordenadas
  ├── dataset_metadata.json    - Info del dataset
  └── labels.csv               - Etiquetas binarias
```

### Modelos y Arquitectura ✅
```
models/
  └── cnn_geotermia.py         - Arquitectura del modelo CNN
```

---

## 🚫 QUÉ NO ESTÁ INCLUIDO (Por Tamaño)

### Datos Grandes (se regeneran en máquina de entrenamiento)
```
❌ data/raw/*.tif              (~2.5 MB) - Descargar con script
❌ data/augmented/             (~1.24 GB) - Generar con script
❌ data/processed/             (~2.5 GB) - Generar con script
❌ models/saved_models/*.keras (~19 MB) - Se genera al entrenar
❌ logs/tensorboard/           (variable) - Se genera al entrenar
```

**Estos archivos se generan automáticamente siguiendo la guía.**

---

## ⏱️ TIEMPO ESTIMADO TOTAL

### En máquina de entrenamiento:

```
Setup inicial:                   15-20 min
├─ Clonar repo:                  1 min
├─ Configurar Python:            5 min
├─ Instalar dependencias:        5-10 min
└─ Autenticar Earth Engine:      1-2 min

Preparación de datos:            35-40 min
├─ Descargar imágenes:           5 min
├─ Augmentar dataset:            30 sec
└─ Preparar para training:       2 min

Entrenamiento:                   2-4 horas (depende del hardware)
├─ CPU (8 cores):               ~4 horas
├─ CPU (16+ cores):             ~2-3 horas
└─ GPU (NVIDIA):                ~20-60 min

Evaluación y visualización:      15-20 min

TOTAL (con CPU):                 3-5 horas
TOTAL (con GPU):                 1-2 horas
```

---

## 📊 ESTADO ACTUAL DEL PROYECTO

### Completado en Máquina Original ✅
- Documentación técnica completa
- Descarga de 85 imágenes ASTER
- Augmentación a 5,518 imágenes
- Preparación del dataset
- **30 épocas entrenadas** (30% completado)

### Métricas Parciales (Época 30)
```
Accuracy:   65.26%
AUC:        0.6252
Loss:       0.9241
Precision:  84.61%
Recall:     68.27%
F1-Score:   ~75.54%
```

### Pendiente (En Máquina de Entrenamiento) ⏳
- Completar 70 épocas restantes
- Evaluar en test set
- Generar visualizaciones finales
- Documentar resultados

---

## 📤 CÓMO SUBIR RESULTADOS

### Después del Entrenamiento:

```bash
# 1. Verificar que estás en develop
git branch
# * develop

# 2. Agregar modelo y resultados
git add models/saved_models/*.keras
git add models/training_history.json
git add results/
git add logs/*.csv

# 3. Commit con métricas
git commit -m "feat: Modelo entrenado completo - Accuracy XX.XX% | AUC X.XX"

# 4. Push a develop
git push origin develop
```

### Si el modelo es muy grande (>100 MB):

Ver opciones en `ENTRENAMIENTO_EXTERNO.md`:
- Git LFS
- Google Drive
- Solo métricas

---

## 🔄 MERGE DE VUELTA A MAIN

### En máquina original (después del entrenamiento):

```bash
# 1. Volver a main
git checkout main

# 2. Pull últimos cambios
git pull origin main

# 3. Fetch develop
git fetch origin develop

# 4. Ver qué cambió
git log origin/develop --oneline

# 5. Merge develop → main
git merge develop

# 6. Resolver conflictos si hay
# (probablemente no habrá)

# 7. Push main
git push origin main

# 8. Eliminar develop (opcional)
git branch -d develop
git push origin --delete develop
```

---

## 🆘 SOPORTE

### Si tienes problemas:

1. **Consulta:** `ENTRENAMIENTO_EXTERNO.md` (guía paso a paso)
2. **Monitoreo:** `MONITOREO_ENTRENAMIENTO.md`
3. **Análisis:** `ANALISIS_ENTRENAMIENTO.md`
4. **GitHub Issues:** Crea un issue en el repositorio

### Contacto:
- **Desarrollador:** Cristian Camilo Vega Sánchez
- **GitHub:** @crisveg24
- **Repo:** https://github.com/crisveg24/geotermia-colombia-cnn

---

## ✅ CHECKLIST RÁPIDO

**Antes de empezar:**
```
□ Tienes acceso a máquina con mejor hardware
□ Tienes cuenta de Google Cloud configurada
□ Python 3.10+ instalado
□ Git instalado
□ 10 GB espacio en disco
```

**Durante el proceso:**
```
□ Clonar repo y checkout develop
□ Seguir ENTRENAMIENTO_EXTERNO.md paso a paso
□ Monitorear entrenamiento con TensorBoard
□ Esperar a completar 100 épocas (o EarlyStopping)
□ Evaluar modelo en test set
□ Commit y push resultados
```

**Al terminar:**
```
□ Modelo entrenado disponible
□ Métricas finales calculadas
□ Visualizaciones generadas
□ Resultados en develop branch
□ Merge a main
□ Eliminar develop
```

---

## 🎯 OBJETIVO FINAL

✅ Modelo CNN entrenado completamente (100 épocas o early stop)  
✅ Accuracy > 85% (objetivo)  
✅ Todas las métricas calculadas  
✅ Visualizaciones para tesis (300 DPI)  
✅ Documentación completa actualizada  

---

**Última actualización:** 3 de noviembre de 2025  
**Rama creada:** develop  
**Estado:** ✅ Lista para clonar y entrenar  

**¡Buena suerte con el entrenamiento! 🚀**
