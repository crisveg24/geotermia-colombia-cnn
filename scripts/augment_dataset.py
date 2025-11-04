"""
Script Alternativo: Ampliar Dataset mediante Data Augmentation
=====================================================================

Este script toma las imágenes existentes y genera múltiples variaciones mediante
técnicas de augmentación de datos para crear un dataset más amplio.

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

# Importar funciones de augmentación
from scipy import ndimage
from skimage import exposure, transform

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetAugmenter:
    """Clase para ampliar el dataset mediante augmentación de datos."""
    
    def __init__(self, input_dir: str = "geotermia_imagenes", output_dir: str = "data/raw"):
        """
        Inicializar augmentador de dataset.
        
        Args:
            input_dir: Directorio con imágenes originales
            output_dir: Directorio de salida para dataset ampliado
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Crear estructura de directorios
        self.positive_dir = self.output_dir / "positive"
        self.negative_dir = self.output_dir / "negative"
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Imágenes originales con sus etiquetas
        self.original_images = {
            'Nevado_del_Ruiz.tif': 1,  # Potencial geotérmico
            'Volcan_Purace.tif': 1,     # Potencial geotérmico
            'Paipa_Iza.tif': 1          # Potencial geotérmico
        }
        
        self.metadata = {
            'dataset_name': 'Colombia_Geothermal_Augmented',
            'creation_date': datetime.now().isoformat(),
            'original_images': len(self.original_images),
            'augmented_images': 0,
            'total_images': 0,
            'augmentation_techniques': [],
            'image_details': []
        }
    
    def load_image(self, image_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Cargar imagen GeoTIFF.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tupla (datos de imagen, metadata)
        """
        with rasterio.open(image_path) as src:
            # Leer todas las bandas
            image = src.read()
            metadata = src.meta.copy()
            
            # Transponer de (bands, height, width) a (height, width, bands)
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            
            return image, metadata
    
    def save_image(self, image: np.ndarray, output_path: Path, metadata: dict):
        """
        Guardar imagen como GeoTIFF.
        
        Args:
            image: Datos de imagen
            output_path: Ruta de salida
            metadata: Metadata de rasterio
        """
        # Asegurar que la imagen tenga la forma correcta (bands, height, width)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        
        # Actualizar metadata
        metadata.update({
            'driver': 'GTiff',
            'count': image.shape[0],
            'height': image.shape[1],
            'width': image.shape[2],
            'dtype': image.dtype
        })
        
        # Guardar
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
        # Normalizar
        image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        # Aplicar factor de brillo
        image_adjusted = np.clip(image_normalized * factor, 0, 1)
        # Desnormalizar
        return image_adjusted * (image.max() - image.min()) + image.min()
    
    def augment_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajustar contraste."""
        if image.ndim == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                band = image[:, :, i]
                # Normalizar
                band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
                # Ajustar contraste
                mean = 0.5
                band_contrast = (band_norm - mean) * factor + mean
                band_contrast = np.clip(band_contrast, 0, 1)
                # Desnormalizar
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
        """Recortar y redimensionar al tamaño original."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        
        # Calcular posición de inicio para crop centrado
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        
        # Crop
        if image.ndim == 3:
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w, :]
        else:
            cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
        
        # Resize back to original size
        return transform.resize(cropped, (h, w), preserve_range=True, anti_aliasing=True)
    
    def generate_augmentations(
        self, 
        image: np.ndarray, 
        base_name: str,
        label: int,
        metadata: dict,
        num_augmentations: int = 30
    ) -> int:
        """
        Generar múltiples augmentaciones de una imagen.
        
        Args:
            image: Imagen original
            base_name: Nombre base del archivo
            label: Etiqueta (1 o 0)
            metadata: Metadata de rasterio
            num_augmentations: Número de augmentaciones a generar
            
        Returns:
            Número de imágenes generadas
        """
        output_dir = self.positive_dir if label == 1 else self.negative_dir
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
            
            # Combinaciones
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
                # Aplicar augmentación
                aug_image = aug_func(image.copy())
                
                # Generar nombre de archivo
                filename = f"{base_name}_{aug_name}.tif"
                output_path = output_dir / filename
                
                # Guardar imagen augmentada
                self.save_image(aug_image, output_path, metadata)
                
                # Registrar en metadata
                self.metadata['image_details'].append({
                    'filename': filename,
                    'original': base_name,
                    'label': label,
                    'augmentation': aug_name,
                    'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2)
                })
                
                count += 1
                logger.info(f"  ✅ Generada: {filename}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Error generando {aug_name}: {e}")
        
        return count
    
    def process_all_images(self, num_augmentations_per_image: int = 30):
        """
        Procesar todas las imágenes originales y generar augmentaciones.
        
        Args:
            num_augmentations_per_image: Número de augmentaciones por imagen original
        """
        logger.info("="*80)
        logger.info("AMPLIANDO DATASET MEDIANTE DATA AUGMENTATION")
        logger.info("="*80)
        
        total_generated = 0
        
        for img_name, label in self.original_images.items():
            img_path = self.input_dir / img_name
            
            if not img_path.exists():
                logger.warning(f"⚠️ Imagen no encontrada: {img_path}")
                continue
            
            logger.info(f"\n📷 Procesando: {img_name} (label={label})")
            
            try:
                # Cargar imagen
                image, metadata = self.load_image(img_path)
                logger.info(f"   Tamaño: {image.shape}")
                
                # Guardar imagen original en carpeta correspondiente
                output_dir = self.positive_dir if label == 1 else self.negative_dir
                original_output = output_dir / f"{img_name.replace('.tif', '_original.tif')}"
                self.save_image(image, original_output, metadata)
                logger.info(f"   ✅ Original guardada: {original_output.name}")
                
                # Registrar original en metadata
                self.metadata['image_details'].append({
                    'filename': original_output.name,
                    'original': img_name,
                    'label': label,
                    'augmentation': 'original',
                    'file_size_mb': round(original_output.stat().st_size / (1024 * 1024), 2)
                })
                
                # Generar augmentaciones
                base_name = img_name.replace('.tif', '')
                count = self.generate_augmentations(
                    image, base_name, label, metadata, num_augmentations_per_image
                )
                
                total_generated += count + 1  # +1 por la original
                logger.info(f"   ✅ Total generadas para {img_name}: {count + 1}")
                
            except Exception as e:
                logger.error(f"   ❌ Error procesando {img_name}: {e}")
        
        # Actualizar metadata final
        self.metadata['augmented_images'] = total_generated - len(self.original_images)
        self.metadata['total_images'] = total_generated
        self.metadata['augmentation_techniques'] = [
            'rotation', 'flip', 'brightness', 'contrast', 'noise', 
            'gaussian_blur', 'crop_and_resize', 'combinations'
        ]
        
        return total_generated
    
    def save_metadata(self):
        """Guardar metadata del dataset."""
        # JSON
        json_path = self.output_dir / "dataset_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"\n📄 Metadata guardada: {json_path}")
        
        # CSV
        if self.metadata['image_details']:
            df = pd.DataFrame(self.metadata['image_details'])
            csv_path = self.output_dir / "dataset_images.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"📊 Lista de imágenes: {csv_path}")
            
            # Estadísticas
            logger.info("\n" + "="*80)
            logger.info("ESTADÍSTICAS DEL DATASET")
            logger.info("="*80)
            logger.info(f"Total de imágenes: {len(df)}")
            logger.info(f"Imágenes originales: {self.metadata['original_images']}")
            logger.info(f"Imágenes augmentadas: {self.metadata['augmented_images']}")
            logger.info(f"Imágenes positivas: {len(df[df['label']==1])}")
            logger.info(f"Imágenes negativas: {len(df[df['label']==0])}")
            logger.info(f"Tamaño total: {df['file_size_mb'].sum():.2f} MB")
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
        logger.info(f"🏷️  Etiquetas guardadas: {labels_path}")


def main():
    """Función principal."""
    logger.info("\n" + "="*80)
    logger.info("AMPLIACIÓN DE DATASET - Proyecto Geotermia Colombia")
    logger.info("Universidad de San Buenaventura - Bogotá")
    logger.info("="*80 + "\n")
    
    # Crear augmentador
    augmenter = DatasetAugmenter(
        input_dir="geotermia_imagenes",
        output_dir="data/raw"
    )
    
    # Configuración
    NUM_AUG_PER_IMAGE = 30
    logger.info("📋 Configuración:")
    logger.info(f"   - Imágenes originales: {len(augmenter.original_images)}")
    logger.info(f"   - Augmentaciones por imagen: {NUM_AUG_PER_IMAGE}")
    logger.info(f"   - Total esperado: ~{len(augmenter.original_images) * (NUM_AUG_PER_IMAGE + 1)}")
    logger.info(f"   - Directorio entrada: geotermia_imagenes/")
    logger.info(f"   - Directorio salida: data/raw/\n")
    
    # Confirmar
    response = input("¿Iniciar ampliación del dataset? (s/n): ")
    if response.lower() != 's':
        logger.info("❌ Operación cancelada")
        return
    
    # Procesar
    import time
    start_time = time.time()
    
    total = augmenter.process_all_images(num_augmentations_per_image=NUM_AUG_PER_IMAGE)
    
    # Guardar metadata
    augmenter.save_metadata()
    augmenter.create_labels_file()
    
    # Resumen
    elapsed = time.time() - start_time
    logger.info(f"\n⏱️  Tiempo total: {elapsed:.2f} segundos")
    logger.info(f"✅ Dataset ampliado exitosamente: {total} imágenes totales")
    logger.info(f"📁 Ubicación: data/raw/")


if __name__ == "__main__":
    main()
