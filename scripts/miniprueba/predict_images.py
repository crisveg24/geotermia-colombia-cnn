"""
Script de Predicción
=====================

Usa el modelo entrenado para clasificar nuevas imágenes.
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import rasterio
from skimage.transform import resize
import argparse


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_PATH = PROJECT_ROOT / 'models' / 'saved_models'
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'images'


def load_model():
    """Carga el modelo."""
    model_path = MODELS_PATH / 'mini_model_best.keras'
    if not model_path.exists():
        model_path = MODELS_PATH / 'mini_model_final.keras'
    return keras.models.load_model(model_path)


def load_and_preprocess_image(image_path):
    """Carga y preprocesa una imagen .tif."""
    with rasterio.open(image_path) as src:
        bands = [src.read(i) for i in range(1, src.count + 1)]
        image = np.stack(bands, axis=-1).astype(np.float32)
    
    # Asegurar 5 bandas
    while image.shape[-1] < 5:
        image = np.concatenate([image, image[..., -1:]], axis=-1)
    if image.shape[-1] > 5:
        image = image[..., :5]
    
    # Resize a 224x224
    image = resize(image, (224, 224, 5), preserve_range=True)
    
    # Normalizar
    for i in range(5):
        band = image[:, :, i]
        min_val, max_val = np.min(band), np.max(band)
        if max_val - min_val > 0:
            image[:, :, i] = ((band - min_val) / (max_val - min_val)) * 255.0
    
    return image


def predict_single(model, image_path):
    """Predice una sola imagen."""
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Batch de 1
    
    prob = model.predict(image, verbose=0)[0][0]
    pred_class = "🔥 GEOTÉRMICO" if prob > 0.5 else "❄️ SIN POTENCIAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    return pred_class, prob, confidence


def main():
    print("=" * 60)
    print("🔮 SISTEMA DE PREDICCIÓN GEOTÉRMICA")
    print("=" * 60)
    
    # Cargar modelo
    print("\n📂 Cargando modelo...")
    model = load_model()
    print("   ✅ Modelo cargado")
    
    # Obtener imágenes disponibles
    images = list(DATA_PATH.glob('*.tif'))
    
    if not images:
        print("❌ No hay imágenes en data/raw/images/")
        return
    
    print(f"\n📋 Imágenes disponibles: {len(images)}")
    print("-" * 60)
    
    # Predecir todas las imágenes
    results = []
    
    for img_path in sorted(images):
        pred_class, prob, confidence = predict_single(model, img_path)
        results.append({
            'imagen': img_path.name,
            'prediccion': pred_class,
            'probabilidad': prob,
            'confianza': confidence
        })
        
        print(f"  {img_path.stem:30} → {pred_class} (conf: {confidence:.1%})")
    
    # Resumen
    geo_count = sum(1 for r in results if "GEOTÉRMICO" in r['prediccion'])
    no_geo_count = len(results) - geo_count
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PREDICCIONES")
    print("=" * 60)
    print(f"   🔥 Zonas Geotérmicas:     {geo_count}")
    print(f"   ❄️ Zonas Sin Potencial:   {no_geo_count}")
    print(f"   📊 Total analizado:       {len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
