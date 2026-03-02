# Copyright (c) 2025-2026 Vega Sánchez · Arévalo Rubiano · Espitia Ayala · Rivera Martín
# Universidad de San Buenaventura — Bogotá | github.com/crisveg24/geotermia-colombia-cnn
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _read_npy_header(f, version):
    """Wrapper compatible con NumPy 1.x y 2.x para leer headers .npy."""
    if hasattr(np.lib.format, '_read_array_header'):
        return np.lib.format._read_array_header(f, version)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    return np.lib.format.read_array_header_2_0(f)


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

    def _load_partitioned_or_single(self, prefix: str) -> np.ndarray:
        """
        Carga un array .npy que puede estar partido en multiples archivos.

        Si existe <prefix>.npy lo carga directamente.
        Si existen <prefix>_part0.npy, <prefix>_part1.npy, ... los concatena.

        Args:
            prefix: Nombre base sin extension (e.g. 'X_train')

        Returns:
            Array numpy concatenado
        """
        single = self.processed_data_path / f'{prefix}.npy'
        if single.exists():
            logger.info(f"  Cargando {prefix}.npy (archivo unico)")
            return np.load(single)

        # Buscar partes
        import glob as _glob
        pattern = str(self.processed_data_path / f'{prefix}_part*.npy')
        parts = sorted(_glob.glob(pattern))
        if not parts:
            raise FileNotFoundError(
                f"No se encontro {single} ni archivos {prefix}_part*.npy"
            )

        logger.info(f"  Cargando {prefix} desde {len(parts)} partes...")
        arrays = []
        for p in parts:
            logger.info(f"    -> {Path(p).name}")
            arrays.append(np.load(p))
        return np.concatenate(arrays, axis=0)

    def _get_train_part_paths(self) -> list:
        """
        Obtiene rutas de X_train particionado sin cargar en RAM.
        Si existe X_train.npy unico, retorna [path].
        """
        return self._get_split_part_paths('X_train')

    def _get_split_part_paths(self, prefix: str) -> list:
        """
        Obtiene rutas de un split particionado sin cargar en RAM.
        Si existe <prefix>.npy unico, retorna [path].
        Si existen <prefix>_part*.npy, retorna lista ordenada.
        """
        single = self.processed_data_path / f'{prefix}.npy'
        if single.exists():
            return [single]

        import glob as _glob
        pattern = str(self.processed_data_path / f'{prefix}_part*.npy')
        parts = sorted(_glob.glob(pattern))
        if not parts:
            raise FileNotFoundError(
                f"No se encontro {single} ni archivos {prefix}_part*.npy"
            )
        return [Path(p) for p in parts]

    def load_data(self) -> Dict:
        """
        Carga los datos procesados (soporta archivos particionados).

        Train X se retorna como rutas a partes (FAT32-safe, ~670 MB c/u)
        para que el generador cargue un lote a la vez.
        Val y test se cargan completos (~1.3 GB c/u, caben en RAM).

        Returns:
            Diccionario con datos y rutas
        """
        logger.info("Cargando datos procesados...")
        logger.info(f"Ruta: {self.processed_data_path}")

        try:
            # Train X: solo rutas a partes (no cargar ~20 GB en RAM)
            train_parts = self._get_train_part_paths()
            y_train = self._load_partitioned_or_single('y_train')

            # Val: también puede estar particionado (~4.7 GB)
            val_parts = self._get_split_part_paths('X_val')
            y_val = self._load_partitioned_or_single('y_val')

            # Test: cargar info pero no los datos (no se usan en train)
            y_test = self._load_partitioned_or_single('y_test')

            with open(self.processed_data_path / 'split_info.json', 'r') as f:
                split_info = json.load(f)

            # Auto-detectar input_shape desde header del primer part
            with open(str(train_parts[0]), 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape_info = _read_npy_header(f, version)
            actual_shape = shape_info[0][1:]  # (224, 224, N_bands)
            if actual_shape != self.input_shape:
                logger.warning(
                    f"input_shape configurado {self.input_shape} != "
                    f"shape real de datos {actual_shape}. "
                    f"Actualizando a {actual_shape}."
                )
                self.input_shape = actual_shape

            logger.info(f"Datos cargados:")
            logger.info(f"  Train: {len(y_train)} imgs en {len(train_parts)} parte(s)")
            logger.info(f"  Validation: {len(y_val)} imgs en {len(val_parts)} parte(s)")
            logger.info(f"  Test: {len(y_test)} imgs (no se cargan en train)")

            return {
                'train_parts': train_parts,
                'y_train': y_train,
                'val_parts': val_parts,
                'y_val': y_val,
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
        val_parts: list,
        y_val: np.ndarray
    ) -> tuple:
        """
        Crea generadores de datos con augmentation.

        v3.3 FIX — Estrategia I/O-eficiente:
        El generador anterior hacía shuffle global con random access sobre
        31 partes y un cache LRU de 2. Esto causaba evict constante
        (~15 recargas de 670 MB por época desde USB) → cuello de botella
        catastrófico y "épocas fantasma" donde TF esperaba datos.

        Nueva estrategia:
        1. Shuffle el orden de las partes cada época
        2. Cargar UNA parte completa a la vez (~670 MB)
        3. Shuffle todas las muestras DENTRO de esa parte
        4. Yield batch a batch (no sample a sample)
        Resultado: lectura secuencial, máximo throughput, sin thrashing.

        El shuffle de partes + shuffle intra-parte da buena aleatoriedad
        estadística sin penalizar I/O.

        Args:
        train_parts: Lista de rutas a X_train_part*.npy
        y_train: Labels completos de train
        val_parts: Lista de rutas a X_val_part*.npy
        y_val: Labels completos de validación

        Returns:
        Tupla de (train_dataset, val_dataset, steps_per_epoch, validation_steps)
        """
        logger.info("Creando generadores de datos (part-sequential v3.3)...")

        n_val = len(y_val)
        # Obtener input_shape del header del primer part sin cargarlo
        with open(str(train_parts[0]), 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape_info = _read_npy_header(f, version)
        input_shape = shape_info[0][1:]  # (224, 224, 7)
        n_train = len(y_train)

        # Calcular offsets de cada parte en y_train
        part_offsets = []  # [(start, end), ...]
        offset = 0
        for p in train_parts:
            with open(str(p), 'rb') as f:
                ver = np.lib.format.read_magic(f)
                sh = _read_npy_header(f, ver)
            part_n = sh[0][0]
            part_offsets.append((offset, offset + part_n))
            offset += part_n

        # Calcular offsets de validación
        val_offsets = []
        offset = 0
        for p in val_parts:
            with open(str(p), 'rb') as f:
                ver = np.lib.format.read_magic(f)
                sh = _read_npy_header(f, ver)
            part_n = sh[0][0]
            val_offsets.append((offset, offset + part_n))
            offset += part_n

        logger.info(f"Train: {n_train} imgs, {len(train_parts)} partes, input={input_shape}")
        logger.info(f"  Labels: clase0={int((y_train==0).sum())} clase1={int((y_train==1).sum())}")
        logger.info(f"Val: {n_val} imgs, {len(val_parts)} partes")

        # Verificar mezcla de clases en partes
        segregated = 0
        for pi, (s, e) in enumerate(part_offsets):
            c0 = int((y_train[s:e] == 0).sum())
            c1 = int((y_train[s:e] == 1).sum())
            if c0 == 0 or c1 == 0:
                segregated += 1
        if segregated > 0:
            logger.warning(f"  {segregated}/{len(train_parts)} partes tienen UNA sola clase!")
            logger.warning("  Ejecuta el script de reshuffle para mezclar clases en cada parte.")
        else:
            logger.info("  ✓ Todas las partes tienen ambas clases (pre-shuffled)")

        # ── Generador de train: secuencial por partes + shuffle intra-parte ──
        # Las partes YA están pre-shuffleadas con clases mezcladas (50/50),
        # así que la lectura secuencial parte-por-parte produce batches
        # balanceados. Shuffle de orden de partes + shuffle intra-parte
        # da buena aleatoriedad sin penalizar I/O.
        bs = self.batch_size

        def train_gen():
            rng = np.random.RandomState()
            while True:
                # Shuffle orden de partes cada época
                part_order = rng.permutation(len(train_parts))

                for pi in part_order:
                    X_part = np.load(str(train_parts[pi]))
                    s, e = part_offsets[pi]
                    y_part = y_train[s:e]

                    # Shuffle intra-parte
                    idx = rng.permutation(len(X_part))
                    X_part = X_part[idx]
                    y_part = y_part[idx]

                    for i in range(len(X_part)):
                        yield X_part[i], y_part[i].reshape(1).astype(np.float32)

                    del X_part, y_part

        # ── Generador de val: secuencial por partes, desde disco ──
        def val_gen():
            while True:
                for vi in range(len(val_parts)):
                    X_vpart = np.load(str(val_parts[vi]))
                    vs, ve = val_offsets[vi]
                    y_vpart = y_val[vs:ve]
                    for i in range(len(X_vpart)):
                        yield X_vpart[i], y_vpart[i].reshape(1).astype(np.float32)
                    del X_vpart, y_vpart

        output_sig = (
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )

        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=output_sig
        )
        train_dataset = train_dataset.batch(bs)

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

        # prefetch 2 batches para solapar I/O con cómputo GPU
        train_dataset = train_dataset.prefetch(2)

        # Validation dataset (sin augmentation)
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=output_sig
        )
        val_dataset = val_dataset.batch(bs)
        val_dataset = val_dataset.prefetch(2)

        # steps_per_epoch y validation_steps para que TF sepa cuándo
        # termina cada época (generadores son infinitos)
        steps_per_epoch = n_train // bs  # drop_last para batches uniformes
        validation_steps = max(1, n_val // bs)

        logger.info(f"  steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

        return train_dataset, val_dataset, steps_per_epoch, validation_steps

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
        val_parts = data['val_parts']
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
        train_dataset, val_dataset, steps_per_epoch, validation_steps = self.create_data_generators(
            train_parts, y_train, val_parts, y_val
        )

        # 3.5 Aplicar CosineDecay con steps_per_epoch REAL
        # (El modelo se compiló con LR fijo; ahora lo recompilamos con el
        #  schedule correcto basado en el tamaño real del dataset)
        total_steps = self.epochs * steps_per_epoch
        logger.info(f"CosineDecay: {total_steps} steps totales "
                     f"({self.epochs} epochs × {steps_per_epoch} steps/epoch)")
        lr_schedule = get_cosine_decay_schedule(
            initial_learning_rate=self._cfg.LEARNING_RATE,
            decay_steps=total_steps,
            alpha=0.0001
        )
        model.optimizer.learning_rate = lr_schedule

        # 4. Crear callbacks
        callbacks = self.create_callbacks(model_name=f'geotermia_cnn_{model_type}')

        # 5. Entrenar modelo
        logger.info("\n" + "="*70)
        logger.info(f"Iniciando entrenamiento - {self.epochs} épocas máximo")
        logger.info(f"  steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")
        logger.info("="*70 + "\n")

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
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
