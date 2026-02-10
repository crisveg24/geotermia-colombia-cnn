"""
Entrenamiento Rápido del Mini-Modelo CNN
=========================================

Entrena la CNN con el mini-dataset para validar que todo funciona.
Configuración ligera para CPU.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import json
import logging

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_geotermia import create_geotermia_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
MODELS_PATH = PROJECT_ROOT / 'models' / 'saved_models'
LOGS_PATH = PROJECT_ROOT / 'logs'

# Hiperparámetros para entrenamiento rápido en CPU
BATCH_SIZE = 4 # Pequeño para dataset pequeño
EPOCHS = 20 # Suficiente para ver si aprende
LEARNING_RATE = 0.001


def load_data():
 """Carga los datos procesados."""
 logger.info(" Cargando datos...")
 
 X_train = np.load(DATA_PATH / 'X_train.npy')
 y_train = np.load(DATA_PATH / 'y_train.npy')
 X_val = np.load(DATA_PATH / 'X_val.npy')
 y_val = np.load(DATA_PATH / 'y_val.npy')
 X_test = np.load(DATA_PATH / 'X_test.npy')
 y_test = np.load(DATA_PATH / 'y_test.npy')
 
 logger.info(f" Train: {X_train.shape}, labels: {y_train.shape}")
 logger.info(f" Val: {X_val.shape}, labels: {y_val.shape}")
 logger.info(f" Test: {X_test.shape}, labels: {y_test.shape}")
 
 return X_train, y_train, X_val, y_val, X_test, y_test


def create_data_augmentation():
 """Crea pipeline de data augmentation."""
 return keras.Sequential([
 keras.layers.RandomFlip("horizontal_and_vertical"),
 keras.layers.RandomRotation(0.2),
 keras.layers.RandomZoom(0.1),
 ], name='data_augmentation')


def main():
 print("=" * 60)
 print(" ENTRENAMIENTO MINI-MODELO CNN")
 print("=" * 60)
 
 # Verificar dispositivo
 gpus = tf.config.list_physical_devices('GPU')
 if gpus:
 print(f" GPU detectada: {gpus[0].name}")
 else:
 print(" Usando CPU (entrenamiento será más lento)")
 
 # Crear directorios
 MODELS_PATH.mkdir(parents=True, exist_ok=True)
 LOGS_PATH.mkdir(parents=True, exist_ok=True)
 
 # Cargar datos
 X_train, y_train, X_val, y_val, X_test, y_test = load_data()
 
 # Verificar forma de entrada
 input_shape = X_train.shape[1:] # (224, 224, 5)
 print(f"\n Input shape: {input_shape}")
 print(f" Clases: 2 (binario)")
 
 # Crear modelo
 print("\n Construyendo modelo CNN...")
 model = create_geotermia_model(
 input_shape=input_shape,
 num_classes=2,
 model_type='custom',
 dropout_rate=0.3, # Menor dropout para dataset pequeño
 l2_reg=0.001 # Mayor regularización
 )
 
 # Resumen del modelo
 total_params = model.count_params()
 print(f" Parámetros totales: {total_params:,}")
 
 # Callbacks
 callbacks = [
 EarlyStopping(
 monitor='val_loss',
 patience=5,
 restore_best_weights=True,
 verbose=1
 ),
 ModelCheckpoint(
 filepath=str(MODELS_PATH / 'mini_model_best.keras'),
 monitor='val_loss',
 save_best_only=True,
 verbose=1
 )
 ]
 
 # Entrenar
 print(f"\n Iniciando entrenamiento...")
 print(f" Epochs: {EPOCHS}")
 print(f" Batch size: {BATCH_SIZE}")
 print(f" Learning rate: {LEARNING_RATE}")
 print("-" * 60)
 
 history = model.fit(
 X_train, y_train,
 validation_data=(X_val, y_val),
 epochs=EPOCHS,
 batch_size=BATCH_SIZE,
 callbacks=callbacks,
 verbose=1
 )
 
 # Guardar modelo final
 model.save(MODELS_PATH / 'mini_model_final.keras')
 print(f"\n Modelos guardados en: {MODELS_PATH}")
 
 # Guardar historial
 history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
 with open(LOGS_PATH / 'history_mini.json', 'w') as f:
 json.dump(history_dict, f, indent=2)
 
 # Evaluar en test
 print("\n Evaluación en conjunto de test:")
 print("-" * 60)
 results = model.evaluate(X_test, y_test, verbose=0)
 
 metrics_names = model.metrics_names
 for name, value in zip(metrics_names, results):
 print(f" {name}: {value:.4f}")
 
 # Predicciones de ejemplo
 print("\n Predicciones de ejemplo:")
 predictions = model.predict(X_test, verbose=0)
 for i, (pred, true) in enumerate(zip(predictions, y_test)):
 pred_class = "Geotérmico" if pred[0] > 0.5 else "Control"
 true_class = "Geotérmico" if true == 1 else "Control"
 confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
 status = "" if (pred[0] > 0.5) == true else ""
 print(f" Test {i+1}: Pred={pred_class} ({confidence:.1%}) | Real={true_class} {status}")
 
 print("\n" + "=" * 60)
 print(" ¡Entrenamiento completado!")
 print("=" * 60)
 
 # Resumen final
 final_acc = history.history['accuracy'][-1]
 final_val_acc = history.history['val_accuracy'][-1]
 print(f"\n Accuracy final: {final_acc:.1%}")
 print(f" Val Accuracy: {final_val_acc:.1%}")
 
 if final_val_acc > 0.6:
 print("\n El modelo está aprendiendo patrones básicos.")
 print(" Con más datos debería mejorar significativamente.")
 else:
 print("\n El modelo necesita más datos o ajustes.")
 print(" Esto es esperado con solo 20 imágenes.")


if __name__ == "__main__":
 main()
