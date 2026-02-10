"""
Prediction Script for Geothermal CNN
=====================================

Script para hacer predicciones en nuevas imágenes usando el modelo entrenado.

Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano
Asesor: Prof. Yeison Eduardo Conejo Sandoval
Universidad de San Buenaventura - Bogotá
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import rasterio
from pathlib import Path
import argparse
import logging
from typing import Tuple, Optional
import json

# Configurar logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeotermalPredictor:
 """
 Clase para hacer predicciones de potencial geotérmico en nuevas imágenes.
 """
 
 def __init__(
 self,
 model_path: str,
 target_size: Tuple[int, int] = (224, 224)
 ):
 """
 Inicializa el predictor.
 
 Args:
 model_path: Ruta al modelo entrenado (.keras)
 target_size: Tamaño objetivo de las imágenes
 """
 self.model_path = Path(model_path)
 self.target_size = target_size
 self.model = None
 
 logger.info("GeotermalPredictor inicializado")
 
 def load_model(self) -> bool:
 """
 Carga el modelo entrenado.
 
 Returns:
 True si se cargó exitosamente, False en caso contrario
 """
 logger.info(f"Cargando modelo desde: {self.model_path}")
 
 try:
 self.model = keras.models.load_model(str(self.model_path))
 logger.info(" Modelo cargado exitosamente")
 return True
 except Exception as e:
 logger.error(f"Error cargando modelo: {e}")
 return False
 
 def load_tif_image(self, file_path: Path) -> Optional[np.ndarray]:
 """
 Carga una imagen .tif.
 
 Args:
 file_path: Ruta al archivo .tif
 
 Returns:
 Array numpy con la imagen o None si hay error
 """
 try:
 with rasterio.open(file_path) as src:
 # Leer todas las bandas
 bands = []
 for i in range(1, src.count + 1):
 band = src.read(i)
 bands.append(band)
 
 # Stack de bandas
 image = np.stack(bands, axis=-1)
 
 logger.info(f"Imagen cargada: {image.shape}")
 return image
 except Exception as e:
 logger.error(f"Error al cargar {file_path}: {e}")
 return None
 
 def preprocess_image(self, image: np.ndarray) -> np.ndarray:
 """
 Preprocesa la imagen para la predicción.
 
 Args:
 image: Array numpy de la imagen
 
 Returns:
 Imagen preprocesada
 """
 from skimage.transform import resize
 
 # 1. Resize
 target_shape = (*self.target_size, image.shape[-1])
 resized = resize(
 image,
 target_shape,
 mode='reflect',
 anti_aliasing=True,
 preserve_range=True
 )
 
 # 2. Normalización por banda
 normalized = np.zeros_like(resized, dtype=np.float32)
 
 for i in range(resized.shape[-1]):
 band = resized[:, :, i]
 mean = np.mean(band)
 std = np.std(band)
 
 if std > 0:
 normalized[:, :, i] = (band - mean) / std
 else:
 normalized[:, :, i] = band - mean
 
 return normalized
 
 def predict(
 self,
 image_path: str,
 show_confidence: bool = True
 ) -> dict:
 """
 Realiza predicción en una imagen.
 
 Args:
 image_path: Ruta a la imagen .tif
 show_confidence: Si True, muestra nivel de confianza
 
 Returns:
 Diccionario con resultados de la predicción
 """
 image_path = Path(image_path)
 
 if not image_path.exists():
 logger.error(f"Archivo no encontrado: {image_path}")
 return None
 
 logger.info("="*70)
 logger.info(f"Realizando predicción para: {image_path.name}")
 logger.info("="*70)
 
 # 1. Cargar imagen
 image = self.load_tif_image(image_path)
 if image is None:
 return None
 
 # 2. Preprocesar
 processed = self.preprocess_image(image)
 
 # 3. Añadir dimensión de batch
 input_tensor = np.expand_dims(processed, axis=0)
 
 # 4. Predicción
 prediction = self.model.predict(input_tensor, verbose=0)
 
 # 5. Interpretar resultado
 if prediction.shape[1] == 1:
 # Clasificación binaria
 probability = float(prediction[0, 0])
 predicted_class = 1 if probability > 0.5 else 0
 confidence = probability if predicted_class == 1 else (1 - probability)
 else:
 # Clasificación multiclase
 predicted_class = int(np.argmax(prediction, axis=1)[0])
 confidence = float(prediction[0, predicted_class])
 probability = float(prediction[0, 1]) # Probabilidad de clase positiva
 
 # 6. Interpretación
 class_names = {
 0: "SIN POTENCIAL GEOTÉRMICO",
 1: "CON POTENCIAL GEOTÉRMICO"
 }
 
 result = {
 'filename': image_path.name,
 'predicted_class': predicted_class,
 'class_name': class_names[predicted_class],
 'probability': probability,
 'confidence': confidence
 }
 
 # 7. Mostrar resultados
 logger.info("\n" + "="*70)
 logger.info("RESULTADO DE LA PREDICCIÓN")
 logger.info("="*70)
 logger.info(f"Archivo: {result['filename']}")
 logger.info(f"Predicción: {result['class_name']}")
 logger.info(f"Probabilidad: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
 
 if show_confidence:
 logger.info(f"Confianza: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
 
 logger.info("="*70 + "\n")
 
 return result
 
 def predict_batch(
 self,
 image_folder: str,
 output_file: Optional[str] = None
 ) -> list:
 """
 Realiza predicciones en múltiples imágenes.
 
 Args:
 image_folder: Carpeta con imágenes .tif
 output_file: Archivo para guardar resultados (opcional)
 
 Returns:
 Lista de diccionarios con resultados
 """
 image_folder = Path(image_folder)
 
 if not image_folder.exists():
 logger.error(f"Carpeta no encontrada: {image_folder}")
 return None
 
 # Buscar archivos .tif
 tif_files = list(image_folder.glob('*.tif'))
 
 if len(tif_files) == 0:
 logger.warning(f"No se encontraron archivos .tif en {image_folder}")
 return []
 
 logger.info(f"Encontrados {len(tif_files)} archivos .tif")
 
 results = []
 
 for tif_file in tif_files:
 result = self.predict(str(tif_file), show_confidence=False)
 if result:
 results.append(result)
 
 # Guardar resultados si se especifica
 if output_file and len(results) > 0:
 output_path = Path(output_file)
 output_path.parent.mkdir(parents=True, exist_ok=True)
 
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 logger.info(f"\n Resultados guardados en: {output_path}")
 
 # Resumen
 logger.info("\n" + "="*70)
 logger.info("RESUMEN DE PREDICCIONES")
 logger.info("="*70)
 logger.info(f"Total de imágenes procesadas: {len(results)}")
 
 with_potential = sum(1 for r in results if r['predicted_class'] == 1)
 without_potential = len(results) - with_potential
 
 logger.info(f"Con potencial geotérmico: {with_potential}")
 logger.info(f"Sin potencial geotérmico: {without_potential}")
 logger.info("="*70 + "\n")
 
 return results


def main():
 """Función principal."""
 
 parser = argparse.ArgumentParser(
 description='Predicción de Potencial Geotérmico con CNN',
 formatter_class=argparse.RawDescriptionHelpFormatter,
 epilog="""
Ejemplos de uso:

 # Predicción en una sola imagen:
 python scripts/predict.py --image data/raw/Nevado_del_Ruiz.tif

 # Predicción en todas las imágenes de una carpeta:
 python scripts/predict.py --folder data/raw --output results/predictions.json

 # Especificar modelo custom:
 python scripts/predict.py --image test_image.tif --model models/saved_models/mi_modelo.keras
 """
 )
 
 parser.add_argument(
 '--image',
 type=str,
 help='Ruta a una imagen .tif para predicción'
 )
 
 parser.add_argument(
 '--folder',
 type=str,
 help='Carpeta con múltiples imágenes .tif'
 )
 
 parser.add_argument(
 '--model',
 type=str,
 default='models/saved_models/geotermia_cnn_custom_best.keras',
 help='Ruta al modelo entrenado (default: mejor modelo custom)'
 )
 
 parser.add_argument(
 '--output',
 type=str,
 help='Archivo para guardar resultados (solo con --folder)'
 )
 
 args = parser.parse_args()
 
 # Validar argumentos
 if not args.image and not args.folder:
 parser.error("Debes especificar --image o --folder")
 
 if args.image and args.folder:
 parser.error("Especifica solo --image o --folder, no ambos")
 
 print("="*70)
 print("GEOTHERMAL POTENTIAL PREDICTION")
 print("Universidad de San Buenaventura - Bogotá")
 print("="*70)
 
 # Verificar que el modelo existe
 if not Path(args.model).exists():
 print(f"\n Error: Modelo no encontrado en {args.model}")
 print("Entrena el modelo primero con: python scripts/train_model.py")
 return
 
 # Crear predictor
 predictor = GeotermalPredictor(
 model_path=args.model,
 target_size=(224, 224)
 )
 
 # Cargar modelo
 if not predictor.load_model():
 return
 
 # Realizar predicción
 if args.image:
 # Predicción única
 result = predictor.predict(args.image)
 
 if result:
 print("\n Predicción completada!")
 
 elif args.folder:
 # Predicción batch
 results = predictor.predict_batch(
 image_folder=args.folder,
 output_file=args.output
 )
 
 if results:
 print(f"\n Predicciones completadas! ({len(results)} imágenes procesadas)")


if __name__ == '__main__':
 main()
