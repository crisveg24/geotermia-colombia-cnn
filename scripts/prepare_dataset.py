"""
Data Preparation Pipeline for Geothermal CNN
==============================================

Script para preparar el dataset de imágenes satelitales ASTER para
entrenamiento del modelo CNN.

Funciones:
- Cargar imágenes .tif de data/raw/
- Normalizar y procesar imágenes
- Crear splits train/validation/test (70/15/15)
- Generar archivos .npy para carga rápida
- Balanceo de clases

Autores: Cristian Vega, Daniel Santiago Arévalo Rubiano
Universidad de San Buenaventura - Bogotá
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import logging
from tqdm import tqdm
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeoDataPreparator:
    """
    Clase para preparar el dataset de imágenes geotérmicas.
    """
    
    def __init__(
        self,
        raw_data_path: str = None,
        processed_data_path: str = None,
        labels_path: str = None,
        target_size: Tuple[int, int] = (224, 224),
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Inicializa el preparador de datos.
        
        Args:
            raw_data_path: Ruta a imágenes (augmented). None = usa config.py (soporta disco externo).
            processed_data_path: Ruta para guardar datos procesados. None = usa config.py.
            labels_path: Ruta al archivo de etiquetas. None = usa config.py.
            target_size: Tamaño objetivo de las imágenes (height, width)
            test_size: Proporción del conjunto de test
            val_size: Proporción del conjunto de validación
            random_state: Semilla para reproducibilidad
        """
        # Importar configuración centralizada
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import cfg
        
        self.raw_data_path = Path(raw_data_path) if raw_data_path else cfg.augmented_dir
        self.processed_data_path = Path(processed_data_path) if processed_data_path else cfg.processed_dir
        self.labels_path = Path(labels_path) if labels_path else cfg.augmented_dir
        self.target_size = target_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Crear directorios si no existen
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GeoDataPreparator inicializado")
        logger.info(f"Raw data path: {self.raw_data_path}")
        logger.info(f"Processed data path: {self.processed_data_path}")
        logger.info(f"Fuente de datos: {cfg.source}")
        logger.info(f"Target size: {self.target_size}")
    
    def load_tif_image(self, file_path: Path) -> np.ndarray:
        """
        Carga una imagen .tif y extrae las bandas especificadas.
        
        Args:
            file_path: Ruta al archivo .tif
            
        Returns:
            Array numpy con las bandas de la imagen (siempre 5 bandas)
        """
        try:
            with rasterio.open(file_path) as src:
                # Leer todas las bandas disponibles
                bands = []
                for i in range(1, src.count + 1):
                    band = src.read(i)
                    bands.append(band)
                
                # Stack de bandas
                image = np.stack(bands, axis=-1)
                
                # Asegurar que siempre haya 5 bandas
                if image.shape[-1] < 5:
                    # Si tiene menos de 5 bandas, duplicar la última banda hasta llegar a 5
                    logger.debug(f"Imagen con {image.shape[-1]} bandas, expandiendo a 5: {file_path.name}")
                    while image.shape[-1] < 5:
                        image = np.concatenate([image, image[..., -1:]], axis=-1)
                elif image.shape[-1] > 5:
                    # Si tiene más de 5 bandas, tomar solo las primeras 5
                    logger.debug(f"Imagen con {image.shape[-1]} bandas, tomando primeras 5: {file_path.name}")
                    image = image[..., :5]
                
                return image
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {e}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensiona la imagen al tamaño objetivo.
        
        Args:
            image: Array numpy de la imagen
            
        Returns:
            Imagen redimensionada
        """
        from skimage.transform import resize
        
        # Preservar el número de canales
        target_shape = (*self.target_size, image.shape[-1])
        
        # Resize preservando el rango de valores
        resized = resize(
            image,
            target_shape,
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True
        )
        
        return resized.astype(np.float32)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza la imagen usando normalización por banda.
        
        Args:
            image: Array numpy de la imagen
            
        Returns:
            Imagen normalizada
        """
        # Normalización por banda (z-score)
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[-1]):
            band = image[:, :, i]
            mean = np.mean(band)
            std = np.std(band)
            
            if std > 0:
                normalized[:, :, i] = (band - mean) / std
            else:
                normalized[:, :, i] = band - mean
        
        return normalized
    
    def create_labels_file(self) -> pd.DataFrame:
        """
        Crea archivo de etiquetas si no existe.
        
        Returns:
            DataFrame con nombres de archivos y etiquetas
        """
        labels_file = self.labels_path / 'labels.csv'
        
        if labels_file.exists():
            logger.info(f"Cargando etiquetas existentes desde {labels_file}")
            return pd.read_csv(labels_file)
        
        logger.info("Creando archivo de etiquetas...")
        
        # Buscar todos los archivos .tif en raw_data_path
        tif_files = list(self.raw_data_path.glob('*.tif'))
        
        # Crear DataFrame con etiquetas por defecto
        data = []
        for file_path in tif_files:
            filename = file_path.name
            
            # Inferir etiqueta basada en el nombre del archivo
            # Zonas geotérmicas conocidas: Ruiz, Purace, Galeras, Paipa, Iza, Azufral
            geothermal_keywords = [
                'ruiz', 'purace', 'galeras', 'paipa', 'iza', 
                'azufral', 'volcan', 'thermal', 'hot_spring'
            ]
            
            label = 0  # Por defecto: sin potencial
            for keyword in geothermal_keywords:
                if keyword.lower() in filename.lower():
                    label = 1  # Con potencial geotérmico
                    break
            
            data.append({
                'filename': filename,
                'label': label,
                'zone_name': filename.replace('.tif', '').replace('_', ' ')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(labels_file, index=False)
        
        logger.info(f"Archivo de etiquetas creado: {labels_file}")
        logger.info(f"Total de imágenes: {len(df)}")
        logger.info(f"Clase 0 (sin potencial): {(df['label'] == 0).sum()}")
        logger.info(f"Clase 1 (con potencial): {(df['label'] == 1).sum()}")
        
        return df
    
    def prepare_dataset(self) -> Dict[str, np.ndarray]:
        """
        Prepara el dataset completo: carga, procesa y divide.
        
        Returns:
            Diccionario con arrays de train, validation y test
        """
        logger.info("="*70)
        logger.info("Iniciando preparación del dataset")
        logger.info("="*70)
        
        # 1. Cargar etiquetas
        labels_df = self.create_labels_file()
        
        if len(labels_df) == 0:
            logger.error("No se encontraron imágenes en data/raw/")
            return None
        
        # 2. Cargar y procesar todas las imágenes
        images = []
        labels = []
        filenames = []
        
        logger.info("\nCargando y procesando imágenes...")
        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
            filename = row['filename']
            label = row['label']
            
            file_path = self.raw_data_path / filename
            
            if not file_path.exists():
                logger.warning(f"Archivo no encontrado: {file_path}")
                continue
            
            # Cargar imagen
            image = self.load_tif_image(file_path)
            
            if image is None:
                continue
            
            # Procesar imagen
            image = self.resize_image(image)
            image = self.normalize_image(image)
            
            images.append(image)
            labels.append(label)
            filenames.append(filename)
        
        # Convertir a arrays numpy
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        logger.info(f"\nDataset cargado:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Labels: {y.shape}")
        logger.info(f"  Clase 0: {(y == 0).sum()} imágenes")
        logger.info(f"  Clase 1: {(y == 1).sum()} imágenes")
        
        # 3. Dividir en train/val/test
        logger.info("\nDividiendo dataset...")
        
        # Primero separar test
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, filenames,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Luego separar train y validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
            X_temp, y_temp, files_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        logger.info(f"  Train: {X_train.shape[0]} imágenes")
        logger.info(f"  Validation: {X_val.shape[0]} imágenes")
        logger.info(f"  Test: {X_test.shape[0]} imágenes")
        
        # 4. Calcular pesos de clase para balanceo
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        logger.info(f"\nPesos de clase (para balanceo):")
        logger.info(f"  Clase 0: {class_weights_dict[0]:.4f}")
        logger.info(f"  Clase 1: {class_weights_dict[1]:.4f}")
        
        # 5. Guardar datos procesados
        logger.info("\nGuardando datos procesados...")
        
        np.save(self.processed_data_path / 'X_train.npy', X_train)
        np.save(self.processed_data_path / 'y_train.npy', y_train)
        np.save(self.processed_data_path / 'X_val.npy', X_val)
        np.save(self.processed_data_path / 'y_val.npy', y_val)
        np.save(self.processed_data_path / 'X_test.npy', X_test)
        np.save(self.processed_data_path / 'y_test.npy', y_test)
        
        # Guardar información de splits
        split_info = {
            'train_files': files_train,
            'val_files': files_val,
            'test_files': files_test,
            'class_weights': class_weights_dict,
            'target_size': self.target_size,
            'num_bands': X.shape[-1]
        }
        
        with open(self.processed_data_path / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Datos guardados en: {self.processed_data_path}")
        logger.info("\n✅ Preparación del dataset completada!")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_weights': class_weights_dict
        }
    
    def load_processed_dataset(self) -> Dict[str, np.ndarray]:
        """
        Carga el dataset ya procesado desde archivos .npy.
        
        Returns:
            Diccionario con arrays de train, validation y test
        """
        logger.info("Cargando dataset procesado...")
        
        try:
            X_train = np.load(self.processed_data_path / 'X_train.npy')
            y_train = np.load(self.processed_data_path / 'y_train.npy')
            X_val = np.load(self.processed_data_path / 'X_val.npy')
            y_val = np.load(self.processed_data_path / 'y_val.npy')
            X_test = np.load(self.processed_data_path / 'X_test.npy')
            y_test = np.load(self.processed_data_path / 'y_test.npy')
            
            with open(self.processed_data_path / 'split_info.json', 'r') as f:
                split_info = json.load(f)
            
            logger.info("✅ Dataset cargado exitosamente")
            logger.info(f"  Train: {X_train.shape}")
            logger.info(f"  Validation: {X_val.shape}")
            logger.info(f"  Test: {X_test.shape}")
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'class_weights': split_info['class_weights']
            }
        
        except FileNotFoundError as e:
            logger.error(f"Dataset procesado no encontrado: {e}")
            logger.info("Ejecuta prepare_dataset() primero.")
            return None


def main():
    """Función principal para ejecutar la preparación de datos."""
    
    print("="*70)
    print("DATA PREPARATION PIPELINE")
    print("Geothermal CNN - Universidad de San Buenaventura")
    print("="*70)
    
    # Importar configuración (soporta disco externo vía GEOTERMIA_DATA_ROOT)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import cfg
    print(cfg.summary())
    
    # Inicializar preparador (rutas desde config.py)
    preparator = GeoDataPreparator(
        target_size=cfg.INPUT_SHAPE[:2],
        test_size=cfg.TEST_SIZE,
        val_size=cfg.VAL_SIZE,
        random_state=cfg.RANDOM_STATE
    )
    
    # Preparar dataset
    dataset = preparator.prepare_dataset()
    
    if dataset is not None:
        print("\n" + "="*70)
        print("✅ Dataset preparado y guardado exitosamente!")
        print("="*70)
        print(f"\nPuedes cargar los datos con:")
        print("  data = np.load('data/processed/X_train.npy')")
    else:
        print("\n❌ Error al preparar el dataset.")
        print("Verifica que existan archivos .tif en data/raw/")


if __name__ == '__main__':
    main()
