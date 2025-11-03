"""
Training Pipeline for Geothermal CNN
=====================================

Script de entrenamiento robusto con características modernas:
- Mixed Precision Training para mejor rendimiento
- Data Augmentation avanzado
- Learning Rate Scheduling
- Early Stopping y Model Checkpointing
- TensorBoard logging
- Class weighting para desbalanceo

Autores: Cristian Vega, Daniel Santiago Arévalo Rubiano
Universidad de San Buenaventura - Bogotá
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, Optional

# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_geotermia import create_geotermia_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeotermiaCNNTrainer:
    """
    Clase para entrenar el modelo CNN de geotermia con configuración avanzada.
    """
    
    def __init__(
        self,
        processed_data_path: str = 'data/processed',
        model_save_path: str = 'models/saved_models',
        logs_path: str = 'logs',
        input_shape: tuple = (224, 224, 5),
        batch_size: int = 32,
        epochs: int = 100,
        use_mixed_precision: bool = True,
        use_augmentation: bool = True
    ):
        """
        Inicializa el trainer.
        
        Args:
            processed_data_path: Ruta a datos procesados
            model_save_path: Ruta para guardar modelos
            logs_path: Ruta para logs de TensorBoard
            input_shape: Forma de entrada del modelo
            batch_size: Tamaño del batch
            epochs: Número máximo de épocas
            use_mixed_precision: Si True, usa Mixed Precision Training
            use_augmentation: Si True, aplica data augmentation
        """
        self.processed_data_path = Path(processed_data_path)
        self.model_save_path = Path(model_save_path)
        self.logs_path = Path(logs_path)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_mixed_precision = use_mixed_precision
        self.use_augmentation = use_augmentation
        
        # Crear directorios
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar Mixed Precision
        if self.use_mixed_precision:
            logger.info("Configurando Mixed Precision Training...")
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            logger.info(f"Mixed Precision Policy: {policy.name}")
        
        # Configurar GPU
        self._configure_gpu()
        
        logger.info("GeotermiaCNNTrainer inicializado")
    
    def _configure_gpu(self):
        """Configura las GPUs disponibles."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Configurar crecimiento dinámico de memoria
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                logger.info(f"GPUs disponibles: {len(gpus)}")
                logger.info(f"GPU names: {[gpu.name for gpu in gpus]}")
            except RuntimeError as e:
                logger.warning(f"Error configurando GPU: {e}")
        else:
            logger.warning("No se detectaron GPUs. Usando CPU.")
    
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Carga los datos procesados.
        
        Returns:
            Diccionario con arrays de datos
        """
        logger.info("Cargando datos procesados...")
        
        try:
            X_train = np.load(self.processed_data_path / 'X_train.npy')
            y_train = np.load(self.processed_data_path / 'y_train.npy')
            X_val = np.load(self.processed_data_path / 'X_val.npy')
            y_val = np.load(self.processed_data_path / 'y_val.npy')
            X_test = np.load(self.processed_data_path / 'X_test.npy')
            y_test = np.load(self.processed_data_path / 'y_test.npy')
            
            with open(self.processed_data_path / 'split_info.json', 'r') as f:
                split_info = json.load(f)
            
            logger.info(f"✅ Datos cargados:")
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
                'class_weights': split_info.get('class_weights', None)
            }
        
        except FileNotFoundError as e:
            logger.error(f"Error cargando datos: {e}")
            logger.error("Ejecuta 'python scripts/prepare_dataset.py' primero")
            return None
    
    def create_data_generators(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> tuple:
        """
        Crea generadores de datos con augmentation.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            
        Returns:
            Tupla de (train_dataset, val_dataset)
        """
        logger.info("Creando generadores de datos...")
        
        if self.use_augmentation:
            logger.info("Data Augmentation ACTIVADO")
            
            # Data Augmentation Layer (moderno)
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.RandomTranslation(0.1, 0.1),
                layers.RandomContrast(0.2),
            ], name='data_augmentation')
            
            # Crear datasets con augmentation
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            logger.info("Data Augmentation DESACTIVADO")
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset (sin augmentation)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def create_callbacks(self, model_name: str) -> list:
        """
        Crea callbacks para el entrenamiento.
        
        Args:
            model_name: Nombre del modelo para guardar
            
        Returns:
            Lista de callbacks
        """
        logger.info("Configurando callbacks...")
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        callbacks = []
        
        # 1. ModelCheckpoint - Guardar mejor modelo
        checkpoint_path = self.model_save_path / f'{model_name}_best.keras'
        callbacks.append(ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ))
        
        # 2. EarlyStopping - Parar si no mejora
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ))
        
        # 3. ReduceLROnPlateau - Reducir learning rate
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        # 4. TensorBoard - Visualización
        tensorboard_path = self.logs_path / f'{model_name}_{timestamp}'
        callbacks.append(TensorBoard(
            log_dir=str(tensorboard_path),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
        
        # 5. CSVLogger - Log en CSV
        csv_path = self.logs_path / f'{model_name}_{timestamp}.csv'
        callbacks.append(CSVLogger(
            filename=str(csv_path),
            separator=',',
            append=False
        ))
        
        logger.info(f"  ModelCheckpoint: {checkpoint_path}")
        logger.info(f"  TensorBoard: {tensorboard_path}")
        logger.info(f"  CSVLogger: {csv_path}")
        
        return callbacks
    
    def train(
        self,
        model_type: str = 'custom',
        class_weights: Optional[Dict] = None
    ) -> keras.Model:
        """
        Entrena el modelo CNN.
        
        Args:
            model_type: Tipo de modelo ('custom' o 'transfer_learning')
            class_weights: Pesos de clase para balanceo
            
        Returns:
            Modelo entrenado
        """
        logger.info("="*70)
        logger.info("INICIANDO ENTRENAMIENTO")
        logger.info("="*70)
        
        # 1. Cargar datos
        data = self.load_data()
        if data is None:
            return None
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        
        if class_weights is None:
            class_weights = data.get('class_weights')
        
        # Convertir class_weights de str keys a int keys
        if class_weights and isinstance(list(class_weights.keys())[0], str):
            class_weights = {int(k): v for k, v in class_weights.items()}
        
        logger.info(f"Class weights: {class_weights}")
        
        # 2. Crear modelo
        logger.info(f"\nCreando modelo: {model_type}")
        model = create_geotermia_model(
            input_shape=self.input_shape,
            num_classes=2,
            model_type=model_type,
            dropout_rate=0.5,
            l2_reg=0.0001
        )
        
        model.summary(print_fn=logger.info)
        
        # 3. Crear generadores de datos
        train_dataset, val_dataset = self.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # 4. Crear callbacks
        callbacks = self.create_callbacks(model_name=f'geotermia_cnn_{model_type}')
        
        # 5. Entrenar modelo
        logger.info("\n" + "="*70)
        logger.info(f"Iniciando entrenamiento - {self.epochs} épocas máximo")
        logger.info("="*70 + "\n")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # 6. Guardar modelo final
        final_model_path = self.model_save_path / f'geotermia_cnn_{model_type}_final.keras'
        model.save(str(final_model_path))
        logger.info(f"\n✅ Modelo final guardado: {final_model_path}")
        
        # 7. Guardar historial
        history_path = self.logs_path / f'history_{model_type}.json'
        with open(history_path, 'w') as f:
            # Convertir arrays numpy a listas para JSON
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        logger.info(f"Historial guardado: {history_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ ENTRENAMIENTO COMPLETADO")
        logger.info("="*70)
        
        return model


def main():
    """Función principal para ejecutar el entrenamiento."""
    
    print("="*70)
    print("GEOTHERMAL CNN TRAINING PIPELINE")
    print("Universidad de San Buenaventura - Bogotá")
    print("="*70)
    
    # Configuración
    trainer = GeotermiaCNNTrainer(
        processed_data_path='data/processed',
        model_save_path='models/saved_models',
        logs_path='logs',
        input_shape=(224, 224, 5),
        batch_size=32,
        epochs=100,
        use_mixed_precision=True,
        use_augmentation=True
    )
    
    # Entrenar modelo custom
    print("\n🚀 Entrenando modelo CNN custom...")
    model = trainer.train(model_type='custom')
    
    if model is not None:
        print("\n" + "="*70)
        print("✅ Entrenamiento completado exitosamente!")
        print("="*70)
        print("\nArchivos generados:")
        print("  - models/saved_models/geotermia_cnn_custom_best.keras")
        print("  - models/saved_models/geotermia_cnn_custom_final.keras")
        print("  - logs/history_custom.json")
        print("\nPara visualizar con TensorBoard:")
        print("  tensorboard --logdir=logs")
    else:
        print("\n❌ Error durante el entrenamiento")


if __name__ == '__main__':
    # Importar layers aquí para evitar error antes de configurar TensorFlow
    from tensorflow.keras import layers
    
    main()
