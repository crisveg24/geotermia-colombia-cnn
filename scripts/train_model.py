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

Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
         Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín
Asesor: Prof. Yeison Eduardo Conejo Sandoval
Universidad de San Buenaventura - Bogotá
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision, layers
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

from models.cnn_geotermia import create_geotermia_model, get_cosine_decay_schedule
from scripts.prepare_dataset import _load_chunked, get_part_paths

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
        input_shape: tuple = (224, 224, 7),
        batch_size: int = 32,
        epochs: int = 100,
        use_mixed_precision: bool = True,
        use_augmentation: bool = False
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
        # Importar configuración centralizada (soporta disco externo)
        sys.path.append(str(Path(__file__).parent.parent))
        from config import cfg
        self._cfg = cfg
        project_root = Path(__file__).parent.parent.absolute()

        # Rutas de datos desde config (puede ser disco externo)
        self.processed_data_path = cfg.processed_dir if processed_data_path == 'data/processed' else project_root / processed_data_path
        self.model_save_path = cfg.models_dir if model_save_path == 'models/saved_models' else project_root / model_save_path
        self.logs_path = project_root / logs_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_mixed_precision = use_mixed_precision
        self.use_augmentation = use_augmentation

        # Crear directorios
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Configurar GPU (antes de mixed precision para saber si hay GPU)
        self._configure_gpu()

        # Configurar Mixed Precision (solo con GPU que soporte float16)
        gpus = tf.config.list_physical_devices('GPU')
        if self.use_mixed_precision and gpus:
            logger.info("Configurando Mixed Precision Training (GPU detectada)...")
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            logger.info(f"Mixed Precision Policy: {policy.name}")
        elif self.use_mixed_precision and not gpus:
            logger.info("Mixed Precision DESACTIVADO: no hay GPU. Usando float32 en CPU.")
            self.use_mixed_precision = False

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

    def load_data(self) -> Dict:
        """
        Carga los datos procesados.

        Train se retorna como rutas a partes (FAT32-safe, ~670 MB c/u)
        para que el generador cargue un lote a la vez.
        Val y test se cargan completos (~1.3 GB c/u, caben en RAM).

        Returns:
        Diccionario con datos y rutas
        """
        logger.info("Cargando datos procesados...")

        try:
            # Train: solo rutas a partes (no cargar 5.5 GB en RAM)
            train_parts = get_part_paths(self.processed_data_path, 'train')
            y_train = np.load(self.processed_data_path / 'y_train.npy')

            # Val/Test: carga completa (caben en RAM)
            X_val = _load_chunked(self.processed_data_path, 'val')
            y_val = np.load(self.processed_data_path / 'y_val.npy')
            X_test = _load_chunked(self.processed_data_path, 'test')
            y_test = np.load(self.processed_data_path / 'y_test.npy')

            with open(self.processed_data_path / 'split_info.json', 'r') as f:
                split_info = json.load(f)

            logger.info(f"Datos cargados:")
            logger.info(f"Train: {len(y_train)} imgs en {len(train_parts)} parte(s)")
            logger.info(f"Validation: {X_val.shape}")
            logger.info(f"Test: {X_test.shape}")

            return {
                'train_parts': train_parts,
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
        train_parts: list,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> tuple:
        """
        Crea generadores de datos con augmentation.

        Train: generador part-aware que carga un lote (~670 MB) a la vez
        desde USB. Shuffle a nivel de partes + shuffle dentro de cada parte.
        Val: generador simple desde array en RAM.

        Args:
        train_parts: Lista de rutas a X_train_part*.npy
        y_train: Labels completos de train
        X_val, y_val: Datos de validacion (en RAM)

        Returns:
        Tupla de (train_dataset, val_dataset, steps_per_epoch)
        """
        logger.info("Creando generadores de datos (part-aware streaming)...")

        n_val = len(X_val)
        # Obtener input_shape del header del primer part sin cargarlo
        with open(str(train_parts[0]), 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape_info = np.lib.format._read_array_header(f, version)
        input_shape = shape_info[0][1:]  # (224, 224, 7)
        n_train = len(y_train)

        # Calcular offsets de cada parte en y_train
        part_offsets = []  # [(start, end), ...]
        offset = 0
        for p in train_parts:
            with open(str(p), 'rb') as f:
                ver = np.lib.format.read_magic(f)
                sh = np.lib.format._read_array_header(f, ver)
            part_n = sh[0][0]
            part_offsets.append((offset, offset + part_n))
            offset += part_n

        logger.info(f"Train: {n_train} imgs, {len(train_parts)} partes, input={input_shape}")

        # Generador de train: carga 1 parte a la vez (~670 MB)
        def train_gen():
            part_order = np.arange(len(train_parts))
            np.random.shuffle(part_order)
            for pi in part_order:
                X_part = np.load(str(train_parts[pi]))
                s, e = part_offsets[pi]
                y_part = y_train[s:e]
                local_idx = np.arange(len(X_part))
                np.random.shuffle(local_idx)
                for li in local_idx:
                    yield X_part[li], y_part[li].reshape(1).astype(np.float32)
                del X_part

        # Generador de val (sin shuffle, todo en RAM)
        def val_gen():
            for i in range(n_val):
                yield X_val[i], y_val[i].reshape(1).astype(np.float32)

        output_sig = (
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )

        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=output_sig
        )
        train_dataset = train_dataset.batch(self.batch_size)

        if self.use_augmentation:
            logger.info("Data Augmentation ACTIVADO")

            # v2: Eliminado RandomContrast -- espera datos [0,1] pero nuestros
            # datos estan normalizados con z-score (media~0, rango [-3,3])
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.1),
                layers.RandomTranslation(0.1, 0.1),
            ], name='data_augmentation')

            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            logger.info("Data Augmentation DESACTIVADO")

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Validation dataset (sin augmentation)
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=output_sig
        )
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # steps_per_epoch para que TF sepa cuando termina una epoca
        steps_per_epoch = (n_train + self.batch_size - 1) // self.batch_size

        return train_dataset, val_dataset, steps_per_epoch

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

        # v2: ReduceLROnPlateau eliminado — conflicta con AdamW.
        # Ahora se usa CosineDecay schedule directamente en el optimizador.
        # Ver build_model() en cnn_geotermia.py.

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

        logger.info(f"ModelCheckpoint: {checkpoint_path}")
        logger.info(f"TensorBoard: {tensorboard_path}")
        logger.info(f"CSVLogger: {csv_path}")

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

        train_parts = data['train_parts']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

        # Nota: el reshape de labels a 2D (N,1) se hace dentro del
        # generador (create_data_generators) para no modificar arrays

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
        train_dataset, val_dataset, steps_per_epoch = self.create_data_generators(
            train_parts, y_train, X_val, y_val
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
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # 6. Guardar modelo final
        final_model_path = self.model_save_path / f'geotermia_cnn_{model_type}_final.keras'
        model.save(str(final_model_path))
        logger.info(f"\nModelo final guardado: {final_model_path}")

        # 7. Guardar historial
        history_path = self.logs_path / f'history_{model_type}.json'
        with open(history_path, 'w') as f:
            # Convertir arrays numpy a listas para JSON
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        logger.info(f"Historial guardado: {history_path}")

        logger.info("\n" + "="*70)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("="*70)

        return model


def main():
    """Función principal para ejecutar el entrenamiento."""

    print("="*70)
    print("GEOTHERMAL CNN TRAINING PIPELINE")
    print("Universidad de San Buenaventura - Bogotá")
    print("="*70)

    # Importar configuración (soporta disco externo vía GEOTERMIA_DATA_ROOT)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import cfg
    print(cfg.summary())

    # Configuración (rutas desde config.py)
    # v2: use_augmentation=False porque el dataset ya fue augmentado 30x offline.
    # Activar augmentation online causaría "augmentación de augmentaciones",
    # generando transformaciones compuestas irrealistas.
    trainer = GeotermiaCNNTrainer(
        processed_data_path='data/processed',
        model_save_path='models/saved_models',
        logs_path='logs',
        input_shape=cfg.INPUT_SHAPE,
        batch_size=cfg.BATCH_SIZE,
        epochs=cfg.EPOCHS,
        use_mixed_precision=True,
        use_augmentation=False
    )

    # Entrenar modelo custom
    print("\nEntrenando modelo CNN custom...")
    model = trainer.train(model_type='custom')

    if model is not None:
        print("\n" + "="*70)
        print("Entrenamiento completado exitosamente!")
        print("="*70)
        print("\nArchivos generados:")
        print(" - models/saved_models/geotermia_cnn_custom_best.keras")
        print(" - models/saved_models/geotermia_cnn_custom_final.keras")
        print(" - logs/history_custom.json")
        print("\nPara visualizar con TensorBoard:")
        print(" tensorboard --logdir=logs")
    else:
        print("\nError durante el entrenamiento")


if __name__ == '__main__':
    main()
