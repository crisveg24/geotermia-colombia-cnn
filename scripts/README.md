# Scripts Directory

Scripts principales para ejecutar el proyecto.

## Orden de ejecución

1. `download_dataset.py`: Descargar imágenes ASTER desde Google Earth Engine
2. `augment_full_dataset.py`: Augmentación de datos (30 variaciones por imagen)
3. `prepare_dataset.py`: Preparar y procesar imágenes para CNN (normalización z-score, split train/val/test)
4. `train_model.py`: Entrenar el modelo CNN
5. `evaluate_model.py`: Evaluar métricas del modelo
6. `visualize_results.py`: Generar gráficos y visualizaciones
7. `predict.py`: Hacer predicciones en nuevas imágenes
