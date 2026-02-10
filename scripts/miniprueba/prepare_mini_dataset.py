"""
Preparar Mini-Dataset para Entrenamiento
==========================================

Convierte las 20 imágenes ASTER descargadas en arrays numpy
listos para entrenar el modelo CNN.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent
RAW_IMAGES_PATH = PROJECT_ROOT / 'data' / 'raw' / 'images'
LABELS_PATH = PROJECT_ROOT / 'data' / 'raw' / 'labels_mini.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed'
TARGET_SIZE = (224, 224) # Tamaño que espera la CNN


def load_tif_image(file_path: Path) -> np.ndarray:
 """Carga imagen .tif y retorna array con 5 bandas."""
 try:
 with rasterio.open(file_path) as src:
 bands = []
 for i in range(1, src.count + 1):
 band = src.read(i)
 bands.append(band)
 
 image = np.stack(bands, axis=-1)
 
 # Asegurar 5 bandas
 while image.shape[-1] < 5:
 image = np.concatenate([image, image[..., -1:]], axis=-1)
 if image.shape[-1] > 5:
 image = image[..., :5]
 
 return image.astype(np.float32)
 except Exception as e:
 logger.error(f"Error cargando {file_path}: {e}")
 return None


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
 """Redimensiona imagen a target_size."""
 target_shape = (*target_size, image.shape[-1])
 resized = resize(
 image,
 target_shape,
 mode='reflect',
 anti_aliasing=True,
 preserve_range=True
 )
 return resized.astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
 """Normaliza imagen entre 0 y 255 para el modelo."""
 # Min-max normalization por banda
 normalized = np.zeros_like(image, dtype=np.float32)
 
 for i in range(image.shape[-1]):
 band = image[:, :, i]
 min_val = np.min(band)
 max_val = np.max(band)
 
 if max_val - min_val > 0:
 normalized[:, :, i] = ((band - min_val) / (max_val - min_val)) * 255.0
 else:
 normalized[:, :, i] = 0
 
 return normalized


def main():
 """Prepara el dataset para entrenamiento."""
 print("=" * 60)
 print("PREPARANDO MINI-DATASET")
 print("=" * 60)
 
 # Crear directorio de salida
 OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
 
 # Cargar etiquetas
 if not LABELS_PATH.exists():
 logger.error(f"No se encontró archivo de etiquetas: {LABELS_PATH}")
 logger.info("Ejecuta primero: python scripts/download_mini_dataset.py")
 return
 
 df = pd.read_csv(LABELS_PATH)
 logger.info(f"Etiquetas cargadas: {len(df)} registros")
 logger.info(f" - Geotérmicas (label=1): {len(df[df['label']==1])}")
 logger.info(f" - Control (label=0): {len(df[df['label']==0])}")
 
 # Cargar y procesar imágenes
 X = []
 y = []
 filenames = []
 
 print("\nProcesando imágenes...")
 for idx, row in df.iterrows():
 filepath = RAW_IMAGES_PATH / row['filename']
 
 if not filepath.exists():
 logger.warning(f"No encontrada: {row['filename']}")
 continue
 
 # Cargar imagen
 image = load_tif_image(filepath)
 if image is None:
 continue
 
 # Redimensionar
 image = resize_image(image, TARGET_SIZE)
 
 # Normalizar
 image = normalize_image(image)
 
 X.append(image)
 y.append(row['label'])
 filenames.append(row['filename'])
 
 logger.info(f"{row['filename']} - Shape: {image.shape} - Label: {row['label']}")
 
 # Convertir a numpy arrays
 X = np.array(X)
 y = np.array(y)
 
 print(f"\nDataset total: {X.shape}")
 print(f" - Clase 0 (control): {np.sum(y == 0)}")
 print(f" - Clase 1 (geotérmico): {np.sum(y == 1)}")
 
 # Split: 70% train, 15% val, 15% test
 # Con dataset pequeño, usar stratify para mantener balance
 X_temp, X_test, y_temp, y_test = train_test_split(
 X, y, test_size=0.15, random_state=42, stratify=y
 )
 
 X_train, X_val, y_train, y_val = train_test_split(
 X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp # 0.176 ≈ 15/85
 )
 
 print(f"\nSplits creados:")
 print(f" - Train: {X_train.shape} (label 0: {np.sum(y_train==0)}, label 1: {np.sum(y_train==1)})")
 print(f" - Val: {X_val.shape} (label 0: {np.sum(y_val==0)}, label 1: {np.sum(y_val==1)})")
 print(f" - Test: {X_test.shape} (label 0: {np.sum(y_test==0)}, label 1: {np.sum(y_test==1)})")
 
 # Guardar arrays
 print("\nGuardando archivos...")
 np.save(OUTPUT_PATH / 'X_train.npy', X_train)
 np.save(OUTPUT_PATH / 'y_train.npy', y_train)
 np.save(OUTPUT_PATH / 'X_val.npy', X_val)
 np.save(OUTPUT_PATH / 'y_val.npy', y_val)
 np.save(OUTPUT_PATH / 'X_test.npy', X_test)
 np.save(OUTPUT_PATH / 'y_test.npy', y_test)
 
 # Guardar metadata
 split_info = {
 'total_samples': len(X),
 'train_samples': len(X_train),
 'val_samples': len(X_val),
 'test_samples': len(X_test),
 'input_shape': list(X_train.shape[1:]),
 'num_classes': 2,
 'class_distribution': {
 'train': {'0': int(np.sum(y_train==0)), '1': int(np.sum(y_train==1))},
 'val': {'0': int(np.sum(y_val==0)), '1': int(np.sum(y_val==1))},
 'test': {'0': int(np.sum(y_test==0)), '1': int(np.sum(y_test==1))}
 }
 }
 
 with open(OUTPUT_PATH / 'split_info.json', 'w') as f:
 json.dump(split_info, f, indent=2)
 
 print(f"\nArchivos guardados en: {OUTPUT_PATH}")
 print(" - X_train.npy, y_train.npy")
 print(" - X_val.npy, y_val.npy")
 print(" - X_test.npy, y_test.npy")
 print(" - split_info.json")
 
 print("\n" + "=" * 60)
 print(" ¡Dataset listo para entrenamiento!")
 print("Siguiente paso: python scripts/train_model.py")
 print("=" * 60)


if __name__ == "__main__":
 main()
