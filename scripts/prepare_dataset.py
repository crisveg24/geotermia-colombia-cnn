# Copyright (c) 2025-2026 Vega Sánchez · Arévalo Rubiano · Espitia Ayala · Rivera Martín
# Universidad de San Buenaventura — Bogotá | github.com/crisveg24/geotermia-colombia-cnn
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

Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
         Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín
Asesor: Prof. Yeison Eduardo Conejo Sandoval
Universidad de San Buenaventura - Bogotá
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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


def _load_chunked(directory: Path, split_name: str) -> np.ndarray:
    """Carga X_{split}_part*.npy y los concatena, o carga X_{split}.npy.

    Usar solo para splits pequenos (val, test) que caben en RAM.
    Para train, usar get_part_paths() + generador part-aware.
    """
    parts = sorted(directory.glob(f'X_{split_name}_part*.npy'))
    if parts:
        arrays = [np.load(str(p)) for p in parts]
        return np.concatenate(arrays, axis=0)
    single = directory / f'X_{split_name}.npy'
    if single.exists():
        return np.load(str(single))
    raise FileNotFoundError(f"No se encontro X_{split_name}*.npy en {directory}")


def get_part_paths(directory: Path, split_name: str) -> list:
    """Retorna lista ordenada de rutas a X_{split}_part*.npy."""
    parts = sorted(directory.glob(f'X_{split_name}_part*.npy'))
    if not parts:
        single = directory / f'X_{split_name}.npy'
        if single.exists():
            return [single]
    return parts


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
        self.cfg = cfg

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
        Array numpy con las bandas de la imagen (siempre 7 bandas)
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

                # v2: Filtrar valores NoData (-9999)
                # ASTER GED usa -9999 como NoData; estos valores corrompen
                # la normalización z-score si no se eliminan.
                nodata_mask = image <= -9999
                nodata_pct = nodata_mask.any(axis=-1).mean() * 100
                if nodata_pct > 50:
                    logger.warning(f"Imagen con {nodata_pct:.1f}% NoData, descartando: {file_path.name}")
                    return None
                if nodata_mask.any():
                    # Reemplazar NoData con la mediana de valores válidos por banda
                    for b in range(image.shape[-1]):
                        band = image[:, :, b]
                        valid = band[band > -9999]
                        if len(valid) > 0:
                            band[band <= -9999] = np.median(valid)
                        else:
                            band[band <= -9999] = 0
                    logger.debug(f"NoData interpolado ({nodata_pct:.1f}% píxeles): {file_path.name}")

                # Asegurar que siempre haya 7 bandas (5 emisividad + temperature + NDVI)
                expected_bands = self.cfg.NUM_BANDS  # 7
                if image.shape[-1] < expected_bands:
                    logger.debug(f"Imagen con {image.shape[-1]} bandas, expandiendo a {expected_bands}: {file_path.name}")
                    while image.shape[-1] < expected_bands:
                        image = np.concatenate([image, image[..., -1:]], axis=-1)
                elif image.shape[-1] > expected_bands:
                    logger.debug(f"Imagen con {image.shape[-1]} bandas, tomando primeras {expected_bands}: {file_path.name}")
                    image = image[..., :expected_bands]

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

        # Buscar todos los archivos .tif en raw_data_path (incluye subdirectorios)
        tif_files = list(self.raw_data_path.glob('*.tif'))
        tif_files += list(self.raw_data_path.glob('positive/*.tif'))
        tif_files += list(self.raw_data_path.glob('negative/*.tif'))

        # Crear DataFrame con etiquetas por defecto
        data = []
        for file_path in tif_files:
            filename = file_path.name

            # Inferir etiqueta basada en el directorio padre o nombre del archivo
            # v2: Priorizar directorio (positive/negative) sobre keywords (BUG 16)
            if 'positive' in str(file_path.parent).lower():
                label = 1
            elif 'negative' in str(file_path.parent).lower():
                label = 0
            else:
                # Fallback: keywords ampliadas con todas las zonas de download_dataset.py
                geothermal_keywords = [
                    'ruiz', 'purace', 'galeras', 'paipa', 'iza', 
                    'azufral', 'volcan', 'thermal', 'hot_spring',
                    'cumbal', 'sotara', 'tolima', 'manizales',
                    'santa_rosa', 'herveo', 'coconuco', 'villa_maria',
                ]

                label = 0 # Por defecto: sin potencial
                for keyword in geothermal_keywords:
                    if keyword.lower() in filename.lower():
                        label = 1 # Con potencial geotérmico
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

        v2: El split se realiza a nivel de IMAGEN ORIGINAL para evitar
        data leakage. Todas las augmentaciones de una misma imagen caen
        en el mismo subset (train, val o test).

        v3: Procesamiento memory-efficient — el split se calcula sobre
        metadatos (no imágenes en RAM) y los .npy se construyen por lotes.

        Returns:
        Diccionario con arrays de train, validation y test
        """
        logger.info("="*70)
        logger.info("Iniciando preparacion del dataset")
        logger.info("="*70)

        # 1. Cargar etiquetas
        labels_df = self.create_labels_file()

        if len(labels_df) == 0:
            logger.error("No se encontraron imagenes en data/raw/")
            return None

        # 2. Resolver rutas y extraer grupos (sin cargar pixeles)
        valid_rows = []  # (file_path, label, filename, group)
        logger.info("\nResolviendo rutas de imagenes...")
        for idx, row in labels_df.iterrows():
            filename = row['filename']
            label = row['label']

            file_path = self.raw_data_path / filename
            if not file_path.exists():
                sub_pos = self.raw_data_path / "positive" / filename
                sub_neg = self.raw_data_path / "negative" / filename
                if sub_pos.exists():
                    file_path = sub_pos
                elif sub_neg.exists():
                    file_path = sub_neg
                else:
                    continue

            # Extraer grupo (imagen original)
            if 'original_image' in labels_df.columns:
                group = row['original_image']
            else:
                stem = Path(filename).stem
                aug_suffixes = [
                    '_original', '_rotation_90', '_rotation_180', '_rotation_270',
                    '_rotation_45', '_rotation_neg45', '_flip_horizontal', '_flip_vertical',
                    '_brightness_1.2', '_brightness_0.8', '_contrast_1.3', '_contrast_0.7',
                    '_noise_small', '_noise_medium', '_blur_light', '_blur_medium',
                    '_crop_0.9', '_crop_0.85', '_rot90_flip_h', '_rot180_bright',
                    '_flip_v_contrast', '_rot45_noise', '_crop_blur', '_bright_blur',
                    '_contrast_noise', '_rot90_crop', '_rot180_contrast', '_flip_h_bright',
                    '_rot270_blur', '_crop_contrast_noise', '_rot45_bright_blur',
                ]
                group = stem
                for suffix in aug_suffixes:
                    if group.endswith(suffix):
                        group = group[:-len(suffix)]
                        break

            valid_rows.append((file_path, label, filename, group))

        n_total = len(valid_rows)
        labels_arr = np.array([r[1] for r in valid_rows], dtype=np.int32)
        groups_arr = np.array([r[3] for r in valid_rows])
        filenames_list = [r[2] for r in valid_rows]

        logger.info(f"Imagenes validas: {n_total}")
        logger.info(f"Clase 0: {(labels_arr == 0).sum()}")
        logger.info(f"Clase 1: {(labels_arr == 1).sum()}")
        logger.info(f"Imagenes originales unicas: {len(np.unique(groups_arr))}")

        # 3. Dividir INDICES en train/val/test con GroupShuffleSplit
        logger.info("\nDividiendo dataset (GroupShuffleSplit por imagen original)...")

        all_indices = np.arange(n_total)
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        temp_idx, test_idx = next(gss_test.split(all_indices, labels_arr, groups_arr))

        val_size_adjusted = self.val_size / (1 - self.test_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=self.random_state
        )
        train_idx, val_idx = next(gss_val.split(
            all_indices[temp_idx], labels_arr[temp_idx], groups_arr[temp_idx]
        ))
        # Mapear val/train indices de vuelta al espacio global
        train_idx_global = temp_idx[train_idx]
        val_idx_global = temp_idx[val_idx]
        test_idx_global = test_idx

        # Verificar no hay leakage
        groups_train = set(groups_arr[train_idx_global])
        groups_val = set(groups_arr[val_idx_global])
        groups_test = set(groups_arr[test_idx_global])
        leak_tv = groups_train & groups_val
        leak_tt = groups_train & groups_test
        leak_vt = groups_val & groups_test
        if leak_tv or leak_tt or leak_vt:
            logger.error(f"DATA LEAKAGE DETECTADO! train&val={len(leak_tv)}, "
                         f"train&test={len(leak_tt)}, val&test={len(leak_vt)}")
        else:
            logger.info("Sin data leakage: ningun grupo original compartido entre splits")

        logger.info(f"Train: {len(train_idx_global)} imagenes")
        logger.info(f"Validation: {len(val_idx_global)} imagenes")
        logger.info(f"Test: {len(test_idx_global)} imagenes")

        # 4. Cargar, procesar y guardar por split (en lotes, FAT32-safe)
        # FAT32 limita archivos a 4 GB y memmap no funciona en USB.
        # Estrategia: procesar en lotes de BATCH_SAVE imagenes (~670 MB),
        # guardar con np.save, y liberar RAM entre lotes.
        BATCH_SAVE = 500  # ~670 MB por lote
        splits = {
            'train': train_idx_global,
            'val': val_idx_global,
            'test': test_idx_global,
        }
        split_files = {}
        split_parts = {}  # {split_name: num_parts}

        h, w = self.target_size
        n_bands = self.cfg.NUM_BANDS

        for split_name, indices in splits.items():
            n = len(indices)
            num_parts = max(1, (n + BATCH_SAVE - 1) // BATCH_SAVE)
            logger.info(f"\nProcesando split '{split_name}' ({n} imagenes, {num_parts} lote(s))...")

            y_all = []
            part_count = 0
            batch_X = []
            batch_y = []

            for i, global_idx in enumerate(tqdm(indices, desc=split_name)):
                file_path, label, filename, group = valid_rows[global_idx]
                image = self.load_tif_image(file_path)
                if image is None:
                    continue
                image = self.resize_image(image)
                image = self.normalize_image(image)
                batch_X.append(image)
                batch_y.append(label)

                # Guardar lote cuando se llena
                if len(batch_X) >= BATCH_SAVE:
                    X_arr = np.array(batch_X, dtype=np.float32)
                    part_path = self.processed_data_path / f'X_{split_name}_part{part_count}.npy'
                    np.save(str(part_path), X_arr)
                    y_all.extend(batch_y)
                    size_mb = os.path.getsize(part_path) / (1024 * 1024)
                    logger.info(f"  Guardado lote {part_count}: {X_arr.shape} ({size_mb:.0f} MB)")
                    del X_arr, batch_X, batch_y
                    batch_X = []
                    batch_y = []
                    part_count += 1

            # Guardar ultimo lote (parcial)
            if batch_X:
                X_arr = np.array(batch_X, dtype=np.float32)
                part_path = self.processed_data_path / f'X_{split_name}_part{part_count}.npy'
                np.save(str(part_path), X_arr)
                y_all.extend(batch_y)
                size_mb = os.path.getsize(part_path) / (1024 * 1024)
                logger.info(f"  Guardado lote {part_count}: {X_arr.shape} ({size_mb:.0f} MB)")
                del X_arr, batch_X, batch_y
                part_count += 1

            # Guardar labels completo
            y_split = np.array(y_all, dtype=np.int32)
            np.save(self.processed_data_path / f'y_{split_name}.npy', y_split)
            split_files[split_name] = [filenames_list[i] for i in indices]
            split_parts[split_name] = part_count

            total_mb = sum(
                os.path.getsize(self.processed_data_path / f'X_{split_name}_part{p}.npy')
                for p in range(part_count)
            ) / (1024 * 1024)
            logger.info(f"Split '{split_name}': {len(y_split)} imagenes, "
                        f"{part_count} lote(s), {total_mb:.0f} MB total")

            # Guardar y_train para class weights
            if split_name == 'train':
                y_train = y_split.copy()
            del y_split, y_all

        # 5. Calcular pesos de clase
        class_weights_val = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: float(w) for i, w in enumerate(class_weights_val)}
        del y_train

        logger.info(f"\nPesos de clase (para balanceo):")
        logger.info(f"Clase 0: {class_weights_dict[0]:.4f}")
        logger.info(f"Clase 1: {class_weights_dict[1]:.4f}")

        # 6. Guardar metadata
        split_info = {
            'train_files': split_files.get('train', []),
            'val_files': split_files.get('val', []),
            'test_files': split_files.get('test', []),
            'class_weights': class_weights_dict,
            'target_size': self.target_size,
            'num_bands': n_bands,
            'split_parts': split_parts,  # partes por split (FAT32)
        }

        with open(self.processed_data_path / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"Datos guardados en: {self.processed_data_path}")
        logger.info("Preparacion del dataset completada!")

        # Retornar solo metadatos (NO recargar arrays gigantes en RAM)
        return {
            'class_weights': class_weights_dict,
            'split_sizes': {
                'train': len(train_idx_global),
                'val': len(val_idx_global),
                'test': len(test_idx_global),
            },
            'processed_path': str(self.processed_data_path),
        }

    def load_processed_dataset(self) -> Dict:
        """
        Carga el dataset ya procesado desde archivos .npy.

        Val y test se cargan completos en RAM (~1.3 GB cada uno).
        Train se retorna como lista de rutas a partes (part-aware)
        para que el generador de entrenamiento cargue un lote a la vez.

        Returns:
        Diccionario con datos y metadatos
        """
        logger.info("Cargando dataset procesado...")

        try:
            # Train: solo rutas (demasiado grande para RAM completo)
            train_parts = get_part_paths(self.processed_data_path, 'train')
            y_train = np.load(self.processed_data_path / 'y_train.npy')

            # Val/Test: carga completa (caben en RAM)
            X_val = _load_chunked(self.processed_data_path, 'val')
            y_val = np.load(self.processed_data_path / 'y_val.npy')
            X_test = _load_chunked(self.processed_data_path, 'test')
            y_test = np.load(self.processed_data_path / 'y_test.npy')

            with open(self.processed_data_path / 'split_info.json', 'r') as f:
                split_info = json.load(f)

            logger.info("Dataset cargado exitosamente")
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
        print("Dataset preparado y guardado exitosamente!")
        print("="*70)
        print(f"\nPuedes cargar los datos con:")
        print(" data = np.load('data/processed/X_train.npy')")
    else:
        print("\nError al preparar el dataset.")
        print("Verifica que existan archivos .tif en data/raw/")


if __name__ == '__main__':
    main()
