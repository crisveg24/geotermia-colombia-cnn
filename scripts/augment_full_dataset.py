"""
Script para Aplicar Data Augmentation al Dataset Completo Descargado
====================================================================

Este script toma TODAS las imágenes descargadas desde Google Earth Engine
y genera múltiples variaciones mediante augmentación de datos.

Autor: Cristian Camilo Vega Sánchez
Universidad de San Buenaventura - Bogotá
Fecha: Noviembre 2025
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Tuple, List
from sklearn.utils import shuffle
import pandas as pd
import time

# Importar funciones de augmentación
from scipy import ndimage
from skimage import exposure, transform

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullDatasetAugmenter:
    """Clase para ampliar el dataset completo mediante augmentación."""
    
    def __init__(self, input_dir: str = None, output_dir: str = None):
        """
        Inicializar augmentador.
        
        Args:
            input_dir: Directorio con imágenes descargadas (positive/ y negative/).
                       Si es None, usa la ruta de config.py (soporta disco externo).
            output_dir: Directorio de salida para dataset ampliado.
                        Si es None, usa la ruta de config.py.
        """
        # Importar configuración centralizada
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import cfg
        
        self.input_dir = Path(input_dir) if input_dir else cfg.raw_dir
        self.output_dir = Path(output_dir) if output_dir else cfg.augmented_dir
        
        # Crear estructura de directorios
        self.positive_dir_out = self.output_dir / "positive"
        self.negative_dir_out = self.output_dir / "negative"
        self.positive_dir_out.mkdir(parents=True, exist_ok=True)
        self.negative_dir_out.mkdir(parents=True, exist_ok=True)
        
        # Directorios de entrada
        self.positive_dir_in = self.input_dir / "positive"
        self.negative_dir_in = self.input_dir / "negative"
        
        self.metadata = {
            'dataset_name': 'Colombia_Geothermal_Full_Augmented',
            'creation_date': datetime.now().isoformat(),
            'original_images': 0,
            'augmented_images': 0,
            'total_images': 0,
            'positive_images': 0,
            'negative_images': 0,
            'augmentation_techniques': [],
            'image_details': []
        }
    
    def load_image(self, image_path: Path) -> Tuple[np.ndarray, dict]:
        """Cargar imagen GeoTIFF."""
        with rasterio.open(image_path) as src:
            image = src.read()
            metadata = src.meta.copy()
            
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            
            return image, metadata
    
    def save_image(self, image: np.ndarray, output_path: Path, metadata: dict):
        """Guardar imagen como GeoTIFF."""
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        
        metadata.update({
            'driver': 'GTiff',
            'count': image.shape[0],
            'height': image.shape[1],
            'width': image.shape[2],
            'dtype': image.dtype
        })
        
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(image)
    
    def augment_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotar imagen."""
        return transform.rotate(image, angle, preserve_range=True, mode='reflect')
    
    def augment_flip(self, image: np.ndarray, axis: int) -> np.ndarray:
        """Voltear imagen."""
        return np.flip(image, axis=axis)
    
    def augment_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajustar brillo."""
        image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image_adjusted = np.clip(image_normalized * factor, 0, 1)
        return image_adjusted * (image.max() - image.min()) + image.min()
    
    def augment_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajustar contraste."""
        if image.ndim == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                band = image[:, :, i]
                band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
                mean = 0.5
                band_contrast = (band_norm - mean) * factor + mean
                band_contrast = np.clip(band_contrast, 0, 1)
                result[:, :, i] = band_contrast * (band.max() - band.min()) + band.min()
            return result
        return image
    
    def augment_noise(self, image: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Agregar ruido gaussiano."""
        noise = np.random.normal(0, sigma * image.std(), image.shape)
        return image + noise
    
    def augment_gaussian_blur(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Aplicar desenfoque gaussiano."""
        if image.ndim == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.gaussian_filter(image[:, :, i], sigma=sigma)
            return result
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    def augment_crop_and_resize(self, image: np.ndarray, crop_factor: float = 0.8) -> np.ndarray:
        """Recortar y redimensionar."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        
        if image.ndim == 3:
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w, :]
        else:
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
        
        return transform.resize(cropped, (h, w), preserve_range=True, anti_aliasing=True)
    
    def generate_augmentations(
        self, 
        image: np.ndarray, 
        base_name: str,
        label: int,
        metadata: dict,
        output_dir: Path,
        num_augmentations: int = 30
    ) -> int:
        """Generar augmentaciones de una imagen."""
        count = 0
        
        # Configuraciones de augmentación
        augmentations = [
            ('rotation_90', lambda img: self.augment_rotation(img, 90)),
            ('rotation_180', lambda img: self.augment_rotation(img, 180)),
            ('rotation_270', lambda img: self.augment_rotation(img, 270)),
            ('rotation_45', lambda img: self.augment_rotation(img, 45)),
            ('rotation_neg45', lambda img: self.augment_rotation(img, -45)),
            ('flip_horizontal', lambda img: self.augment_flip(img, axis=1)),
            ('flip_vertical', lambda img: self.augment_flip(img, axis=0)),
            ('brightness_1.2', lambda img: self.augment_brightness(img, 1.2)),
            ('brightness_0.8', lambda img: self.augment_brightness(img, 0.8)),
            ('contrast_1.3', lambda img: self.augment_contrast(img, 1.3)),
            ('contrast_0.7', lambda img: self.augment_contrast(img, 0.7)),
            ('noise_small', lambda img: self.augment_noise(img, 0.005)),
            ('noise_medium', lambda img: self.augment_noise(img, 0.01)),
            ('blur_light', lambda img: self.augment_gaussian_blur(img, 0.5)),
            ('blur_medium', lambda img: self.augment_gaussian_blur(img, 1.0)),
            ('crop_0.9', lambda img: self.augment_crop_and_resize(img, 0.9)),
            ('crop_0.85', lambda img: self.augment_crop_and_resize(img, 0.85)),
            ('rot90_flip_h', lambda img: self.augment_flip(self.augment_rotation(img, 90), axis=1)),
            ('rot180_bright', lambda img: self.augment_brightness(self.augment_rotation(img, 180), 1.2)),
            ('flip_v_contrast', lambda img: self.augment_contrast(self.augment_flip(img, axis=0), 1.3)),
            ('rot45_noise', lambda img: self.augment_noise(self.augment_rotation(img, 45), 0.005)),
            ('crop_blur', lambda img: self.augment_gaussian_blur(self.augment_crop_and_resize(img, 0.9), 0.5)),
            ('bright_blur', lambda img: self.augment_gaussian_blur(self.augment_brightness(img, 1.2), 0.5)),
            ('contrast_noise', lambda img: self.augment_noise(self.augment_contrast(img, 1.3), 0.005)),
            ('rot90_crop', lambda img: self.augment_crop_and_resize(self.augment_rotation(img, 90), 0.9)),
            ('rot180_contrast', lambda img: self.augment_contrast(self.augment_rotation(img, 180), 1.2)),
            ('flip_h_bright', lambda img: self.augment_brightness(self.augment_flip(img, axis=1), 1.1)),
            ('rot270_blur', lambda img: self.augment_gaussian_blur(self.augment_rotation(img, 270), 0.7)),
            ('crop_contrast_noise', lambda img: self.augment_noise(self.augment_contrast(self.augment_crop_and_resize(img, 0.85), 1.2), 0.007)),
            ('rot45_bright_blur', lambda img: self.augment_gaussian_blur(self.augment_brightness(self.augment_rotation(img, 45), 1.15), 0.6)),
        ]
        
        # Generar augmentaciones
        for aug_name, aug_func in augmentations[:num_augmentations]:
            try:
                aug_image = aug_func(image.copy())
                filename = f"{base_name}_{aug_name}.tif"
                output_path = output_dir / filename
                
                self.save_image(aug_image, output_path, metadata)
                
                self.metadata['image_details'].append({
                    'filename': filename,
                    'original': base_name,
                    'label': label,
                    'augmentation': aug_name,
                    'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2)
                })
                
                count += 1
                
            except Exception as e:
                logger.warning(f"  Error generando {aug_name} para {base_name}: {e}")
        
        return count
    
    def process_all_images(self, num_augmentations_per_image: int = 30):
        """Procesar TODAS las imágenes descargadas."""
        logger.info("="*80)
        logger.info("AMPLIANDO DATASET COMPLETO MEDIANTE DATA AUGMENTATION")
        logger.info("="*80)
        
        total_generated = 0
        positive_count = 0
        negative_count = 0
        
        # Procesar imágenes POSITIVAS
        if self.positive_dir_in.exists():
            positive_images = list(self.positive_dir_in.glob("*.tif"))
            logger.info(f"\n Procesando {len(positive_images)} imagenes POSITIVAS...")
            
            for idx, img_path in enumerate(positive_images, 1):
                logger.info(f"\n[{idx}/{len(positive_images)}] Procesando: {img_path.name}")
                
                try:
                    # Cargar imagen
                    image, metadata = self.load_image(img_path)
                    
                    # Guardar original
                    base_name = img_path.stem
                    original_output = self.positive_dir_out / f"{base_name}_original.tif"
                    self.save_image(image, original_output, metadata)
                    
                    self.metadata['image_details'].append({
                        'filename': original_output.name,
                        'original': img_path.name,
                        'label': 1,
                        'augmentation': 'original',
                        'file_size_mb': round(original_output.stat().st_size / (1024 * 1024), 2)
                    })
                    
                    # Generar augmentaciones
                    count = self.generate_augmentations(
                        image, base_name, 1, metadata, self.positive_dir_out, num_augmentations_per_image
                    )
                    
                    total_generated += count + 1
                    positive_count += count + 1
                    logger.info(f"  Generadas: {count + 1} imagenes")
                    
                except Exception as e:
                    logger.error(f"  Error procesando {img_path.name}: {e}")
        
        # Procesar imágenes NEGATIVAS
        if self.negative_dir_in.exists():
            negative_images = list(self.negative_dir_in.glob("*.tif"))
            logger.info(f"\n Procesando {len(negative_images)} imagenes NEGATIVAS...")
            
            for idx, img_path in enumerate(negative_images, 1):
                logger.info(f"\n[{idx}/{len(negative_images)}] Procesando: {img_path.name}")
                
                try:
                    # Cargar imagen
                    image, metadata = self.load_image(img_path)
                    
                    # Guardar original
                    base_name = img_path.stem
                    original_output = self.negative_dir_out / f"{base_name}_original.tif"
                    self.save_image(image, original_output, metadata)
                    
                    self.metadata['image_details'].append({
                        'filename': original_output.name,
                        'original': img_path.name,
                        'label': 0,
                        'augmentation': 'original',
                        'file_size_mb': round(original_output.stat().st_size / (1024 * 1024), 2)
                    })
                    
                    # Generar augmentaciones
                    count = self.generate_augmentations(
                        image, base_name, 0, metadata, self.negative_dir_out, num_augmentations_per_image
                    )
                    
                    total_generated += count + 1
                    negative_count += count + 1
                    logger.info(f"  Generadas: {count + 1} imagenes")
                    
                except Exception as e:
                    logger.error(f"  Error procesando {img_path.name}: {e}")
        
        # Actualizar metadata
        self.metadata['augmented_images'] = total_generated - (len(list(self.positive_dir_in.glob("*.tif"))) + len(list(self.negative_dir_in.glob("*.tif"))))
        self.metadata['total_images'] = total_generated
        self.metadata['positive_images'] = positive_count
        self.metadata['negative_images'] = negative_count
        self.metadata['augmentation_techniques'] = [
            'rotation', 'flip', 'brightness', 'contrast', 'noise', 
            'gaussian_blur', 'crop_and_resize', 'combinations'
        ]
        
        return total_generated, positive_count, negative_count
    
    def save_metadata(self):
        """Guardar metadata."""
        # JSON
        json_path = self.output_dir / "dataset_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"\nMetadata guardada: {json_path}")
        
        # CSV
        if self.metadata['image_details']:
            df = pd.DataFrame(self.metadata['image_details'])
            csv_path = self.output_dir / "dataset_images.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Lista de imagenes: {csv_path}")
            
            # Estadísticas
            logger.info("\n" + "="*80)
            logger.info("ESTADISTICAS DEL DATASET AMPLIADO")
            logger.info("="*80)
            logger.info(f"Total de imagenes: {len(df)}")
            logger.info(f"Imagenes positivas: {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")
            logger.info(f"Imagenes negativas: {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
            logger.info(f"Tamano total: {df['file_size_mb'].sum():.2f} MB")
            logger.info("="*80)
    
    def create_labels_file(self):
        """Crear archivo de etiquetas."""
        labels_data = []
        for detail in self.metadata['image_details']:
            labels_data.append({
                'filename': detail['filename'],
                'label': detail['label'],
                'original_image': detail['original'],
                'augmentation': detail['augmentation']
            })
        
        df = pd.DataFrame(labels_data)
        labels_path = self.output_dir / "labels.csv"
        df.to_csv(labels_path, index=False, encoding='utf-8')
        logger.info(f"Etiquetas guardadas: {labels_path}")


def main():
    """Función principal."""
    logger.info("\n" + "="*80)
    logger.info("AMPLIACION DE DATASET COMPLETO")
    logger.info("Universidad de San Buenaventura - Bogota")
    logger.info("="*80 + "\n")
    
    # Crear augmentador
    augmenter = FullDatasetAugmenter(
        input_dir="data/raw",
        output_dir="data/augmented"
    )
    
    # Contar imágenes originales
    pos_orig = len(list(augmenter.positive_dir_in.glob("*.tif"))) if augmenter.positive_dir_in.exists() else 0
    neg_orig = len(list(augmenter.negative_dir_in.glob("*.tif"))) if augmenter.negative_dir_in.exists() else 0
    total_orig = pos_orig + neg_orig
    
    # Configuración
    NUM_AUG_PER_IMAGE = 30
    logger.info("Configuracion:")
    logger.info(f"   - Imagenes originales: {total_orig} ({pos_orig} positivas + {neg_orig} negativas)")
    logger.info(f"   - Augmentaciones por imagen: {NUM_AUG_PER_IMAGE}")
    logger.info(f"   - Total esperado: ~{total_orig * (NUM_AUG_PER_IMAGE + 1)} imagenes")
    logger.info(f"   - Directorio entrada: data/raw/")
    logger.info(f"   - Directorio salida: data/augmented/\n")
    
    # Confirmar
    response = input("Iniciar ampliacion del dataset completo? (s/n): ")
    if response.lower() != 's':
        logger.info("Operacion cancelada")
        return
    
    # Procesar
    start_time = time.time()
    
    total, pos_count, neg_count = augmenter.process_all_images(num_augmentations_per_image=NUM_AUG_PER_IMAGE)
    
    # Guardar metadata
    augmenter.save_metadata()
    augmenter.create_labels_file()
    
    # Resumen
    elapsed = time.time() - start_time
    logger.info(f"\nTiempo total: {elapsed/60:.2f} minutos")
    logger.info(f"Dataset ampliado exitosamente: {total} imagenes totales")
    logger.info(f"   - Positivas: {pos_count}")
    logger.info(f"   - Negativas: {neg_count}")
    logger.info(f"Ubicacion: data/augmented/")


if __name__ == "__main__":
    main()
