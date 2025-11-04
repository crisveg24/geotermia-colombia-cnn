#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para actualizar el archivo labels.csv con las rutas correctas (positive/ y negative/)
"""

import pandas as pd
from pathlib import Path

# Leer el archivo de labels
labels_path = Path('data/augmented/labels.csv')
df = pd.read_csv(labels_path)

print(f"Total de filas: {len(df)}")
print(f"Primeras filas originales:")
print(df.head())

# Actualizar filename con el subdirectorio correcto
df['filename'] = df.apply(
    lambda row: f"{'positive' if row['label'] == 1 else 'negative'}/{row['filename']}", 
    axis=1
)

# Guardar el archivo actualizado
df.to_csv(labels_path, index=False)

print(f"\n✓ Actualizado labels.csv con subdirectorios")
print(f"Primeras filas actualizadas:")
print(df.head())
print(f"\nDistribución de etiquetas:")
print(df['label'].value_counts())
