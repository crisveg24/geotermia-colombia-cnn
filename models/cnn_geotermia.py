"""
CNN Architecture for Geothermal Potential Classification
=========================================================

Modelo de Deep Learning basado en Redes Neuronales Convolucionales (CNN)
para la identificación de zonas con alto potencial geotérmico en Colombia.

Características:
- Arquitectura moderna con bloques residuales (ResNet-inspired)
- SpatialDropout2D para regularización espacial efectiva
- AdamW optimizer con weight decay correcto
- Label Smoothing para reducir overfitting
- Batch Normalization para estabilidad de entrenamiento
- Global Average Pooling para reducir parámetros
- Mixed Precision Training compatible
- Transfer Learning ready

Autores: 
 - Cristian Camilo Vega Sánchez
 - Daniel Santiago Arévalo Rubiano
 - Yuliet Katerin Espitia Ayala
 - Laura Sophie Rivera Martin
 
Asesor: Prof. Yeison Eduardo Conejo Sandoval
Universidad de San Buenaventura - Bogotá
Programa: Ingeniería de Sistemas (Pregrado)
Fecha: 2025-2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from typing import Tuple, Optional
import logging
import math

# Importar F1Score (TensorFlow 2.13+)
try:
 from tensorflow.keras.metrics import F1Score
 HAS_F1_METRIC = True
except ImportError:
 HAS_F1_METRIC = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeotermiaCNN:
 """
 Clase principal para la arquitectura CNN de clasificación geotérmica.
 
 Esta clase implementa un modelo de Deep Learning para clasificar imágenes
 satelitales según su potencial geotérmico (binario: alto/bajo).
 
 Attributes:
 input_shape (tuple): Dimensiones de entrada (height, width, channels)
 num_classes (int): Número de clases (2 para clasificación binaria)
 model (keras.Model): Modelo compilado de Keras
 """
 
 def __init__(
 self,
 input_shape: Tuple[int, int, int] = (224, 224, 5),
 num_classes: int = 2,
 dropout_rate: float = 0.5,
 l2_reg: float = 0.0001,
 use_batch_norm: bool = True
 ):
 """
 Inicializa la arquitectura CNN.
 
 Args:
 input_shape: Dimensiones de entrada (height, width, channels)
 Por defecto: (224, 224, 5) para 5 bandas térmicas ASTER
 num_classes: Número de clases de salida (2 para binario)
 dropout_rate: Tasa de dropout para regularización
 l2_reg: Factor de regularización L2
 use_batch_norm: Si True, usa Batch Normalization
 """
 self.input_shape = input_shape
 self.num_classes = num_classes
 self.dropout_rate = dropout_rate
 self.l2_reg = l2_reg
 self.use_batch_norm = use_batch_norm
 self.model = None
 
 logger.info(f"Inicializando GeotermiaCNN con input_shape={input_shape}")
 
 def _conv_block(
 self,
 x: tf.Tensor,
 filters: int,
 kernel_size: int = 3,
 strides: int = 1,
 name: str = ""
 ) -> tf.Tensor:
 """
 Bloque convolucional con Conv2D + BatchNorm + ReLU + Dropout.
 
 Args:
 x: Tensor de entrada
 filters: Número de filtros
 kernel_size: Tamaño del kernel
 strides: Stride de la convolución
 name: Nombre del bloque
 
 Returns:
 Tensor procesado
 """
 x = layers.Conv2D(
 filters=filters,
 kernel_size=kernel_size,
 strides=strides,
 padding='same',
 kernel_regularizer=regularizers.l2(self.l2_reg),
 name=f'{name}_conv'
 )(x)
 
 if self.use_batch_norm:
 x = layers.BatchNormalization(name=f'{name}_bn')(x)
 
 x = layers.Activation('relu', name=f'{name}_relu')(x)
 # SpatialDropout2D es más efectivo para CNNs - apaga canales completos
 x = layers.SpatialDropout2D(self.dropout_rate * 0.3, name=f'{name}_spatial_dropout')(x)
 
 return x
 
 def _residual_block(
 self,
 x: tf.Tensor,
 filters: int,
 name: str = ""
 ) -> tf.Tensor:
 """
 Bloque residual (ResNet-inspired) para mejor flujo de gradientes.
 
 Args:
 x: Tensor de entrada
 filters: Número de filtros
 name: Nombre del bloque
 
 Returns:
 Tensor con conexión residual
 """
 shortcut = x
 
 # Primera convolución
 x = self._conv_block(x, filters, kernel_size=3, name=f'{name}_1')
 
 # Segunda convolución (sin ReLU final)
 x = layers.Conv2D(
 filters=filters,
 kernel_size=3,
 padding='same',
 kernel_regularizer=regularizers.l2(self.l2_reg),
 name=f'{name}_2_conv'
 )(x)
 
 if self.use_batch_norm:
 x = layers.BatchNormalization(name=f'{name}_2_bn')(x)
 
 # Ajustar dimensiones del shortcut si es necesario
 if shortcut.shape[-1] != filters:
 shortcut = layers.Conv2D(
 filters=filters,
 kernel_size=1,
 padding='same',
 name=f'{name}_shortcut'
 )(shortcut)
 
 # Conexión residual
 x = layers.Add(name=f'{name}_add')([x, shortcut])
 x = layers.Activation('relu', name=f'{name}_relu')(x)
 
 return x
 
 def build_model(self) -> keras.Model:
 """
 Construye la arquitectura CNN completa.
 
 Arquitectura:
 1. Input Layer (224x224x5)
 2. Initial Conv Block (32 filters)
 3. Residual Block 1 (64 filters) + MaxPooling
 4. Residual Block 2 (128 filters) + MaxPooling
 5. Residual Block 3 (256 filters) + MaxPooling
 6. Residual Block 4 (512 filters) + MaxPooling
 7. Global Average Pooling
 8. Dense Layer (256 units)
 9. Output Layer (num_classes)
 
 Returns:
 Modelo Keras compilado
 """
 logger.info("Construyendo arquitectura CNN...")
 
 inputs = layers.Input(shape=self.input_shape, name='input_layer')
 
 # Normalización de entrada
 x = layers.Rescaling(1./255, name='rescaling')(inputs)
 
 # Bloque convolucional inicial
 x = self._conv_block(x, filters=32, kernel_size=7, strides=2, name='initial_conv')
 x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='initial_pool')(x)
 
 # Bloques residuales con downsampling progresivo
 x = self._residual_block(x, filters=64, name='res_block_1')
 x = layers.MaxPooling2D(pool_size=2, name='pool_1')(x)
 
 x = self._residual_block(x, filters=128, name='res_block_2')
 x = layers.MaxPooling2D(pool_size=2, name='pool_2')(x)
 
 x = self._residual_block(x, filters=256, name='res_block_3')
 x = layers.MaxPooling2D(pool_size=2, name='pool_3')(x)
 
 x = self._residual_block(x, filters=512, name='res_block_4')
 
 # Global Average Pooling (reduce parámetros vs Flatten)
 x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
 
 # Dense layers con regularización
 x = layers.Dense(
 256,
 kernel_regularizer=regularizers.l2(self.l2_reg),
 name='dense_1'
 )(x)
 
 if self.use_batch_norm:
 x = layers.BatchNormalization(name='dense_1_bn')(x)
 
 x = layers.Activation('relu', name='dense_1_relu')(x)
 x = layers.Dropout(self.dropout_rate, name='dense_1_dropout')(x)
 
 # Capa de salida
 if self.num_classes == 2:
 # Clasificación binaria
 outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
 # Label Smoothing reduce overfitting (0.1 = 90% confianza máxima)
 loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
 metrics = [
 'accuracy',
 keras.metrics.Precision(name='precision'),
 keras.metrics.Recall(name='recall'),
 keras.metrics.AUC(name='auc'),
 keras.metrics.AUC(curve='PR', name='auc_pr'), # Mejor para clases desbalanceadas
 ]
 # Agregar F1Score si está disponible (TensorFlow 2.13+)
 if HAS_F1_METRIC:
 metrics.append(F1Score(name='f1_score', threshold=0.5))
 else:
 # Clasificación multiclase
 outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
 loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
 metrics = ['accuracy']
 
 # Crear modelo
 model = models.Model(inputs=inputs, outputs=outputs, name='GeotermiaCNN')
 
 # AdamW: Mejor regularización que Adam estándar (weight decay correcto)
 optimizer = keras.optimizers.AdamW(
 learning_rate=0.001,
 weight_decay=0.0001, # Regularización L2 correcta
 beta_1=0.9,
 beta_2=0.999,
 epsilon=1e-07
 )
 
 # Compilar modelo
 model.compile(
 optimizer=optimizer,
 loss=loss,
 metrics=metrics
 )
 
 self.model = model
 logger.info(f"Modelo construido exitosamente. Total de parámetros: {model.count_params():,}")
 
 return model
 
 def build_transfer_learning_model(
 self,
 base_model_name: str = 'efficientnet',
 freeze_base: bool = True
 ) -> keras.Model:
 """
 Construye modelo usando Transfer Learning con modelos pre-entrenados.
 
 Args:
 base_model_name: Nombre del modelo base ('efficientnet' o 'resnet50')
 freeze_base: Si True, congela las capas del modelo base
 
 Returns:
 Modelo Keras con transfer learning
 """
 logger.info(f"Construyendo modelo con Transfer Learning ({base_model_name})...")
 
 # Seleccionar modelo base
 if base_model_name.lower() == 'efficientnet':
 base_model = EfficientNetB0(
 include_top=False,
 weights='imagenet',
 input_shape=(224, 224, 3), # EfficientNet requiere 3 canales
 pooling='avg'
 )
 elif base_model_name.lower() == 'resnet50':
 base_model = ResNet50V2(
 include_top=False,
 weights='imagenet',
 input_shape=(224, 224, 3),
 pooling='avg'
 )
 else:
 raise ValueError(f"Modelo base '{base_model_name}' no soportado")
 
 # Congelar capas del modelo base si se especifica
 if freeze_base:
 base_model.trainable = False
 logger.info("Capas del modelo base congeladas")
 
 # Construcción del modelo completo
 inputs = layers.Input(shape=self.input_shape, name='input_layer')
 
 # Adaptar canales de entrada si son diferentes de 3
 if self.input_shape[-1] != 3:
 # Proyección de canales a 3 usando Conv2D 1x1
 x = layers.Conv2D(3, kernel_size=1, padding='same', name='channel_adapter')(inputs)
 else:
 x = inputs
 
 # Aplicar modelo base pre-entrenado
 x = base_model(x, training=False)
 
 # Capas de clasificación personalizadas
 x = layers.Dense(
 256,
 kernel_regularizer=regularizers.l2(self.l2_reg),
 activation='relu',
 name='dense_1'
 )(x)
 x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
 
 x = layers.Dense(
 128,
 kernel_regularizer=regularizers.l2(self.l2_reg),
 activation='relu',
 name='dense_2'
 )(x)
 x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
 
 # Capa de salida
 if self.num_classes == 2:
 outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
 loss = 'binary_crossentropy'
 metrics = [
 'accuracy',
 keras.metrics.Precision(name='precision'),
 keras.metrics.Recall(name='recall'),
 keras.metrics.AUC(name='auc')
 ]
 else:
 outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
 loss = 'categorical_crossentropy'
 metrics = ['accuracy']
 
 # Crear y compilar modelo
 model = models.Model(inputs=inputs, outputs=outputs, name=f'GeotermiaCNN_{base_model_name}')
 
 model.compile(
 optimizer=keras.optimizers.Adam(learning_rate=0.001),
 loss=loss,
 metrics=metrics
 )
 
 self.model = model
 logger.info(f"Modelo con Transfer Learning construido. Parámetros: {model.count_params():,}")
 
 return model
 
 def summary(self):
 """Imprime el resumen del modelo."""
 if self.model is None:
 logger.warning("Modelo no construido. Llama a build_model() primero.")
 return
 
 self.model.summary()
 
 def get_model(self) -> Optional[keras.Model]:
 """Retorna el modelo construido."""
 return self.model


def get_cosine_decay_schedule(
 initial_learning_rate: float = 0.001,
 decay_steps: int = 1000,
 alpha: float = 0.0001
) -> keras.optimizers.schedules.LearningRateSchedule:
 """
 Crea un schedule de Cosine Decay para el learning rate.
 
 El learning rate decrece siguiendo una función coseno, lo que permite:
 - Exploración amplia al inicio (LR alto)
 - Convergencia fina al final (LR bajo)
 
 Args:
 initial_learning_rate: LR inicial
 decay_steps: Pasos totales de decay (epochs * steps_per_epoch)
 alpha: LR mínimo final como fracción del inicial
 
 Returns:
 Schedule de learning rate para usar en el optimizador
 
 Example:
 >>> schedule = get_cosine_decay_schedule(0.001, 10000)
 >>> optimizer = keras.optimizers.AdamW(learning_rate=schedule)
 """
 return keras.optimizers.schedules.CosineDecay(
 initial_learning_rate=initial_learning_rate,
 decay_steps=decay_steps,
 alpha=alpha,
 name='cosine_decay'
 )


def create_geotermia_model(
 input_shape: Tuple[int, int, int] = (224, 224, 5),
 num_classes: int = 2,
 model_type: str = 'custom',
 **kwargs
) -> keras.Model:
 """
 Función helper para crear rápidamente un modelo de geotermia.
 
 Args:
 input_shape: Dimensiones de entrada
 num_classes: Número de clases
 model_type: Tipo de modelo ('custom' o 'transfer_learning')
 **kwargs: Argumentos adicionales para GeotermiaCNN
 
 Returns:
 Modelo Keras compilado
 
 Example:
 >>> model = create_geotermia_model(input_shape=(224, 224, 5))
 >>> model.summary()
 """
 cnn = GeotermiaCNN(
 input_shape=input_shape,
 num_classes=num_classes,
 **kwargs
 )
 
 if model_type == 'custom':
 return cnn.build_model()
 elif model_type == 'transfer_learning':
 base_model = kwargs.get('base_model_name', 'efficientnet')
 return cnn.build_transfer_learning_model(base_model_name=base_model)
 else:
 raise ValueError(f"Tipo de modelo '{model_type}' no válido. Usa 'custom' o 'transfer_learning'")


if __name__ == '__main__':
 # Test de la arquitectura
 print("=" * 70)
 print("Testing GeotermiaCNN Architecture")
 print("=" * 70)
 
 # Crear modelo custom
 print("\n1. Modelo Custom CNN:")
 model_custom = create_geotermia_model(
 input_shape=(224, 224, 5),
 num_classes=2,
 model_type='custom'
 )
 model_custom.summary()
 
 # Crear modelo con Transfer Learning (opcional)
 print("\n2. Modelo con Transfer Learning (EfficientNetB0):")
 model_transfer = create_geotermia_model(
 input_shape=(224, 224, 5),
 num_classes=2,
 model_type='transfer_learning',
 base_model_name='efficientnet'
 )
 model_transfer.summary()
 
 print("\n Arquitecturas creadas exitosamente!")
