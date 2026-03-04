#!/usr/bin/env python3
"""
train_model_v7.py - Transfer Learning con EfficientNetB0 (CLEAN SPLITS)
========================================================================
Cambios vs V6:
  1. USA processed_v2 (ZERO data leakage entre splits)
  2. Channel adapter mejorado: Conv2D 7->16->3 para mejor extraccion
  3. Backbone: EfficientNetB0 con warmup + cosine decay
  4. Mayor regularización: MixUp + más agresivo augmentation
  5. Phase 1 más larga (30 epochs) para que la cabeza aprenda bien
  6. Phase 2: fine-tune últimas 50 capas (no todas)

Arquitectura:
  Input (224,224,7) -> Conv2D 3x3 (7->16) -> BN+ReLU
  -> Conv2D 1x1 (16->3) -> BN+ReLU
  -> EfficientNetB0 (ImageNet, pooling=avg)
  -> Dense 256+BN+ReLU+Drop(0.5)
  -> Dense 64+BN+ReLU+Drop(0.3)
  -> Dense 1 sigmoid
"""
import os
import sys
import json
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====================== CONFIG ======================
DATA_DIR = "/home/cristian/geotermia/data/processed_v2"
SAVE_DIR = "/home/cristian/geotermia/models/saved_models"
LOG_DIR = "/home/cristian/geotermia/logs"

INPUT_SHAPE = (224, 224, 7)
BATCH_SIZE = 32
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 50
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
DROPOUT_HEAD = 0.5
DROPOUT_TAIL = 0.3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
PATIENCE_P1 = 12
PATIENCE_P2 = 15
SHUFFLE_BUFFER = 1000
MIXUP_ALPHA = 0.2  # MixUp regularization
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU with memory growth and mixed precision."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU: {gpus[0].name}")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision float16 ON")
    else:
        logger.warning("No GPU detected")


def get_split_info():
    """Read split_info.json and labels from processed_v2."""
    with open(os.path.join(DATA_DIR, "split_info.json")) as f:
        info = json.load(f)
    sp = info["split_parts"]
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return sp["train"], sp["val"], sp["test"], y_train, y_val, y_test


def make_part_dataset(prefix, num_parts, y_all, batch_size,
                      shuffle=True, augment=False, mixup=False):
    """
    Create tf.data.Dataset from .npy parts using generator.
    Uses .repeat() to avoid end-of-sequence bug.
    """
    def generator():
        indices = list(range(num_parts))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            path = os.path.join(DATA_DIR, f"X_{prefix}_part{idx}.npy")
            X_part = np.load(path)
            start = idx * 500
            end = start + X_part.shape[0]
            y_part = y_all[start:end]

            if shuffle:
                perm = np.random.permutation(len(X_part))
                X_part = X_part[perm]
                y_part = y_part[perm]

            for i in range(len(X_part)):
                yield X_part[i].astype(np.float32), y_part[i].astype(np.float32)

            del X_part, y_part

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )

    # CRITICAL: .repeat() so generator re-instantiates each epoch
    ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    # Online augmentation for training
    if augment:
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
            # Per-band Gaussian noise (stddev 0.03-0.08)
            noise_std = tf.random.uniform([], 0.03, 0.08)
            image = image + tf.random.normal(tf.shape(image), mean=0.0, stddev=noise_std)
            # Random per-band brightness shift
            shift = tf.random.normal(shape=(1, 1, 7), mean=0.0, stddev=0.05)
            image = image + shift
            return image, label

        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True if shuffle else False)

    # MixUp regularization (only training)
    if mixup and MIXUP_ALPHA > 0:
        def mixup_fn(x, y):
            batch_size_t = tf.shape(x)[0]
            # Sample lambda from Beta distribution
            lam = tf.random.uniform([], 0.0, MIXUP_ALPHA)
            # Shuffle indices for mixing
            indices_perm = tf.random.shuffle(tf.range(batch_size_t))
            x_mixed = lam * tf.gather(x, indices_perm) + (1 - lam) * x
            y_mixed = lam * tf.gather(y, indices_perm) + (1 - lam) * y
            return x_mixed, y_mixed

        ds = ds.map(mixup_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    """
    EfficientNetB0 with improved channel adapter.
    Two-stage adapter: 7->16->3 channels.
    """
    from tensorflow.keras.applications import EfficientNetB0

    inputs = layers.Input(shape=INPUT_SHAPE, name='input')

    # Channel adapter stage 1: 7 -> 16 (learn spectral features)
    x = layers.Conv2D(
        16, kernel_size=3, padding='same',
        use_bias=False, name='adapter_conv1'
    )(inputs)
    x = layers.BatchNormalization(name='adapter_bn1')(x)
    x = layers.Activation('relu', name='adapter_relu1')(x)

    # Channel adapter stage 2: 16 -> 3 (project to RGB-like space)
    x = layers.Conv2D(
        3, kernel_size=1, padding='same',
        use_bias=False, name='adapter_conv2'
    )(x)
    x = layers.BatchNormalization(name='adapter_bn2')(x)
    x = layers.Activation('relu', name='adapter_relu2')(x)

    # EfficientNetB0 backbone
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model._name = 'efficientnet_base'
    base_model.trainable = False

    x = base_model(x, training=False)

    # Classification head
    x = layers.Dense(256, use_bias=False, name='dense1')(x)
    x = layers.BatchNormalization(name='dense1_bn')(x)
    x = layers.Activation('relu', name='dense1_relu')(x)
    x = layers.Dropout(DROPOUT_HEAD, name='dense1_drop')(x)

    x = layers.Dense(64, use_bias=False, name='dense2')(x)
    x = layers.BatchNormalization(name='dense2_bn')(x)
    x = layers.Activation('relu', name='dense2_relu')(x)
    x = layers.Dropout(DROPOUT_TAIL, name='dense2_drop')(x)

    x = layers.Dense(1, dtype='float32', name='output_logit')(x)
    outputs = layers.Activation('sigmoid', dtype='float32', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='GeotermiaCNN_V7')

    return model, base_model


def compile_model(model, learning_rate):
    """Compile with AdamW optimizer."""
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY
        ),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='auc_pr', curve='PR'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )


def get_callbacks(phase_name):
    """Training callbacks."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    patience = PATIENCE_P1 if 'phase1' in phase_name else PATIENCE_P2
    best_path = os.path.join(SAVE_DIR, f"geotermia_v7_{phase_name}_best.keras")
    csv_path = os.path.join(LOG_DIR, f"geotermia_v7_{phase_name}.csv")

    return [
        keras.callbacks.ModelCheckpoint(
            best_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(csv_path, append=False),
    ]


def evaluate_model(model, test_ds, y_test, test_steps, phase_name):
    """Complete evaluation with classification report."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION {phase_name}")
    logger.info(f"{'='*60}")

    results = model.evaluate(test_ds, steps=test_steps, verbose=1)
    metric_names = ['loss'] + [m.name for m in model.metrics]
    for name, val in zip(metric_names, results):
        logger.info(f"  {name}: {val:.4f}")

    y_pred_prob = model.predict(test_ds, steps=test_steps, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_test_trimmed = y_test[:len(y_pred_prob)]

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_test_trimmed, y_pred,
                                target_names=['No Geotermico', 'Geotermico']))
    cm = confusion_matrix(y_test_trimmed, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return dict(zip(metric_names, results))


def main():
    setup_gpu()

    logger.info("=" * 60)
    logger.info("TRAIN MODEL V7 - CLEAN SPLITS (zero leakage)")
    logger.info(f"Data: {DATA_DIR}")
    logger.info("=" * 60)

    n_train, n_val, n_test, y_train, y_val, y_test = get_split_info()
    logger.info(f"Split: train={n_train} parts ({len(y_train)} samples), "
                f"val={n_val} parts ({len(y_val)} samples), "
                f"test={n_test} parts ({len(y_test)} samples)")

    # Class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    total = len(y_train)
    class_weights = {
        0: total / (2.0 * n_neg),
        1: total / (2.0 * n_pos)
    }
    logger.info(f"Class weights: {class_weights}")
    logger.info(f"Train: pos={n_pos}, neg={n_neg}, ratio={n_pos/total:.3f}")

    # Datasets
    train_ds = make_part_dataset('train', n_train, y_train, BATCH_SIZE,
                                  shuffle=True, augment=True, mixup=True)
    val_ds = make_part_dataset('val', n_val, y_val, BATCH_SIZE,
                                shuffle=False, augment=False, mixup=False)
    test_ds = make_part_dataset('test', n_test, y_test, BATCH_SIZE,
                                 shuffle=False, augment=False, mixup=False)

    steps_per_epoch = len(y_train) // BATCH_SIZE
    val_steps = len(y_val) // BATCH_SIZE
    test_steps = len(y_test) // BATCH_SIZE
    logger.info(f"Steps: train={steps_per_epoch}, val={val_steps}, test={test_steps}")

    # Build model
    model, base_model = build_model()

    # ============================================================
    #  PHASE 1: Backbone frozen, train adapter + head
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Backbone FROZEN")
    logger.info(f"  Epochs: {PHASE1_EPOCHS}, LR: {PHASE1_LR}")
    logger.info("=" * 60)

    compile_model(model, PHASE1_LR)
    model.summary()

    trainable_count = sum(np.prod(w.shape) for w in model.trainable_weights)
    logger.info(f"Trainable params: {trainable_count:,}")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=get_callbacks('phase1'),
        verbose=1
    )

    results1 = evaluate_model(model, test_ds, y_test, test_steps, "PHASE 1")

    # ============================================================
    #  PHASE 2: Fine-tune backbone (last 50 layers)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Fine-tuning backbone (last 50 layers)")
    logger.info(f"  Epochs: {PHASE2_EPOCHS}, LR: {PHASE2_LR}")
    logger.info("=" * 60)

    # Unfreeze last 50 layers of backbone
    base_model.trainable = True
    n_layers = len(base_model.layers)
    freeze_until = max(0, n_layers - 50)
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Keep BN frozen for stability

    trainable_base = sum(1 for l in base_model.layers if l.trainable)
    total_base = len(base_model.layers)
    logger.info(f"Backbone trainable layers: {trainable_base}/{total_base}")

    compile_model(model, PHASE2_LR)
    trainable_count = sum(np.prod(w.shape) for w in model.trainable_weights)
    logger.info(f"Total trainable params: {trainable_count:,}")

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=get_callbacks('phase2'),
        verbose=1
    )

    results2 = evaluate_model(model, test_ds, y_test, test_steps, "PHASE 2")

    # Save final model
    final_path = os.path.join(SAVE_DIR, "geotermia_v7_final.keras")
    model.save(final_path)
    logger.info(f"\nFinal model: {final_path}")

    # Save full history
    full_history = {}
    for key in history1.history:
        full_history[key] = (
            [float(v) for v in history1.history[key]] +
            [float(v) for v in history2.history[key]]
        )
    history_path = os.path.join(LOG_DIR, "history_v7.json")
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY V7 (CLEAN SPLITS)")
    logger.info("=" * 60)
    p1_acc = results1.get('accuracy', results1.get('compile_metrics', 'N/A'))
    p1_auc = results1.get('auc', 'N/A')
    p2_acc = results2.get('accuracy', results2.get('compile_metrics', 'N/A'))
    p2_auc = results2.get('auc', 'N/A')
    logger.info(f"Phase 1 - test acc: {p1_acc if isinstance(p1_acc, str) else f'{p1_acc:.4f}'}, "
                f"auc: {p1_auc if isinstance(p1_auc, str) else f'{p1_auc:.4f}'}")
    logger.info(f"Phase 2 - test acc: {p2_acc if isinstance(p2_acc, str) else f'{p2_acc:.4f}'}, "
                f"auc: {p2_auc if isinstance(p2_auc, str) else f'{p2_auc:.4f}'}")
    logger.info(f"\nModels saved to: {SAVE_DIR}")
    logger.info(f"Logs saved to: {LOG_DIR}")


if __name__ == '__main__':
    main()
