# Mini-Prueba: Validación del Pipeline

Esta carpeta contiene scripts de prueba para validar el pipeline completo del proyecto con un dataset reducido.

## Propósito

Estos scripts fueron creados para:
- Validar que la descarga de datos desde Google Earth Engine funciona
- Probar el preprocesamiento de imágenes ASTER
- Verificar que el modelo puede entrenar (aunque con datos insuficientes)
- Confirmar que la evaluación genera métricas y gráficos
- Demostrar el pipeline de predicción

## Scripts

| Script | Descripción |
|--------|-------------|
| `download_mini_dataset.py` | Descarga 20 imágenes ASTER (10 geotérmicas + 10 control) |
| `prepare_mini_dataset.py` | Convierte .tif a numpy y divide en train/val/test |
| `train_mini_model.py` | Entrena modelo simplificado (~30 seg en CPU) |
| `evaluate_mini_model.py` | Genera métricas y visualizaciones |
| `predict_images.py` | Ejecuta predicciones en imágenes |

## Limitaciones

- **20 imágenes NO son suficientes** para entrenar un modelo real
- El modelo tiende a predecir todo como "geotérmico" debido a la falta de datos
- Estos scripts son **solo para validación del pipeline**

## Uso

```bash
# 1. Descargar mini-dataset
python scripts/miniprueba/download_mini_dataset.py

# 2. Preparar datos
python scripts/miniprueba/prepare_mini_dataset.py

# 3. Entrenar modelo
python scripts/miniprueba/train_mini_model.py

# 4. Evaluar
python scripts/miniprueba/evaluate_mini_model.py

# 5. Predecir
python scripts/miniprueba/predict_images.py
```

## Resultados Generados

- `results/figures/`: Gráficos de evaluación
- `results/metrics/`: Métricas en CSV y JSON
- `models/saved_models/`: Modelos entrenados

---

**Nota**: Para entrenamiento real, usar los scripts principales con un dataset de mínimo 200+ imágenes y GPU.
