# 🔧 Mejoras Sugeridas para el Modelo CNN

## Estado Actual del Código

El código del modelo `cnn_geotermia.py` está **bien estructurado** y sigue buenas prácticas:
- ✅ Arquitectura modular con bloques residuales
- ✅ Batch Normalization y Dropout
- ✅ Regularización L2
- ✅ Global Average Pooling
- ✅ Documentación completa
- ✅ Soporte para Transfer Learning

## Mejoras Recomendadas

### 1. 🎯 Mejoras en la Arquitectura

```python
# ANTES: Dropout fijo
x = layers.Dropout(self.dropout_rate * 0.5)(x)

# DESPUÉS: Dropout adaptativo por capa (Spatial Dropout para Conv)
x = layers.SpatialDropout2D(self.dropout_rate * 0.3)(x)  # Más efectivo para CNNs
```

### 2. 📊 Data Augmentation Mejorado

Agregar en `train_model.py`:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    # Nuevo: augmentación específica para imágenes térmicas
    layers.GaussianNoise(0.05),
])
```

### 3. 🧠 Attention Mechanism (Mejora significativa)

Agregar Spatial Attention después de los bloques residuales:
```python
def _attention_block(self, x, name=""):
    """Bloque de atención espacial."""
    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    
    dense = layers.Dense(x.shape[-1] // 16, activation='relu')
    avg_out = dense(avg_pool)
    max_out = dense(max_pool)
    
    channel_attention = layers.Dense(x.shape[-1], activation='sigmoid')(avg_out + max_out)
    x = x * tf.reshape(channel_attention, [-1, 1, 1, x.shape[-1]])
    
    return x
```

### 4. 📈 Learning Rate Scheduling Mejorado

```python
# Usar Cosine Annealing con Warm Restarts
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9
)
```

### 5. 🔄 Mixup y CutMix (Data Augmentation Avanzado)

```python
def mixup(images, labels, alpha=0.2):
    """Aplica Mixup augmentation."""
    batch_size = tf.shape(images)[0]
    lam = np.random.beta(alpha, alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels
```

### 6. 📊 Métricas Adicionales

```python
# Agregar métricas específicas para geotermia
metrics = [
    'accuracy',
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(curve='PR', name='auc_pr'),  # Mejor para desbalanceo
    tfa.metrics.F1Score(num_classes=2, average='macro', name='f1'),  # F1 directo
]
```

### 7. 🎛️ Manejo de Desbalanceo de Clases

```python
# Calcular pesos de clase automáticamente
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Usar en fit()
model.fit(X_train, y_train, class_weight=class_weight_dict, ...)
```

### 8. 🔍 Explainability (Interpretabilidad)

Agregar Grad-CAM para visualizar qué áreas del imagen activan el modelo:
```python
import tensorflow as tf

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Genera heatmap de activación."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

## Prioridad de Implementación

| Mejora | Impacto | Dificultad | Prioridad |
|--------|---------|------------|-----------|
| Data Augmentation mejorado | Alto | Baja | 🔴 Alta |
| Manejo desbalanceo | Alto | Baja | 🔴 Alta |
| Learning Rate Scheduling | Medio | Baja | 🟡 Media |
| Attention Mechanism | Alto | Media | 🟡 Media |
| Mixup/CutMix | Medio | Media | 🟢 Baja |
| Grad-CAM | Bajo | Media | 🟢 Baja |

## Próximos Pasos

1. **Inmediato:** Aplicar class weights y mejor data augmentation
2. **Corto plazo:** Implementar learning rate scheduling mejorado
3. **Medio plazo:** Agregar attention mechanism
4. **Largo plazo:** Implementar Grad-CAM para interpretabilidad

---

**Nota:** Estas mejoras son opcionales. El modelo actual ya tiene una buena base y puede lograr buenos resultados con suficientes datos de entrenamiento.
