"""
Evaluación Completa del Modelo CNN
===================================

Genera todas las métricas y visualizaciones para la tesis.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
 accuracy_score, precision_score, recall_score, f1_score,
 roc_auc_score, confusion_matrix, classification_report,
 roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
MODELS_PATH = PROJECT_ROOT / 'models' / 'saved_models'
RESULTS_PATH = PROJECT_ROOT / 'results'
FIGURES_PATH = RESULTS_PATH / 'figures'
METRICS_PATH = RESULTS_PATH / 'metrics'
LOGS_PATH = PROJECT_ROOT / 'logs'


def load_data():
 """Carga datos de test."""
 X_test = np.load(DATA_PATH / 'X_test.npy')
 y_test = np.load(DATA_PATH / 'y_test.npy')
 return X_test, y_test


def load_model():
 """Carga el modelo entrenado."""
 model_path = MODELS_PATH / 'mini_model_best.keras'
 if not model_path.exists():
 model_path = MODELS_PATH / 'mini_model_final.keras'
 
 if not model_path.exists():
 raise FileNotFoundError("No se encontró modelo entrenado")
 
 print(f" Cargando modelo: {model_path.name}")
 return keras.models.load_model(model_path)


def calculate_metrics(y_true, y_pred, y_prob):
 """Calcula todas las métricas."""
 metrics = {
 'accuracy': accuracy_score(y_true, y_pred),
 'precision': precision_score(y_true, y_pred, zero_division=0),
 'recall': recall_score(y_true, y_pred, zero_division=0),
 'f1_score': f1_score(y_true, y_pred, zero_division=0),
 'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
 }
 return metrics


def plot_confusion_matrix(y_true, y_pred, save_path):
 """Genera matriz de confusión."""
 cm = confusion_matrix(y_true, y_pred)
 
 plt.figure(figsize=(8, 6))
 sns.heatmap(
 cm, 
 annot=True, 
 fmt='d', 
 cmap='Blues',
 xticklabels=['Sin Potencial (0)', 'Con Potencial (1)'],
 yticklabels=['Sin Potencial (0)', 'Con Potencial (1)']
 )
 plt.title('Matriz de Confusión\nModelo CNN Geotérmico', fontsize=14, fontweight='bold')
 plt.xlabel('Predicción', fontsize=12)
 plt.ylabel('Valor Real', fontsize=12)
 plt.tight_layout()
 plt.savefig(save_path)
 plt.close()
 print(f" Guardado: {save_path.name}")


def plot_roc_curve(y_true, y_prob, save_path):
 """Genera curva ROC."""
 if len(np.unique(y_true)) < 2:
 print(" No hay suficientes clases para curva ROC")
 return
 
 fpr, tpr, _ = roc_curve(y_true, y_prob)
 auc = roc_auc_score(y_true, y_prob)
 
 plt.figure(figsize=(8, 6))
 plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'CNN (AUC = {auc:.3f})')
 plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
 plt.fill_between(fpr, tpr, alpha=0.3)
 
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
 plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
 plt.title('Curva ROC\nModelo CNN Geotérmico', fontsize=14, fontweight='bold')
 plt.legend(loc='lower right', fontsize=11)
 plt.grid(True, alpha=0.3)
 plt.tight_layout()
 plt.savefig(save_path)
 plt.close()
 print(f" Guardado: {save_path.name}")


def plot_training_history(save_path):
 """Genera gráfico del historial de entrenamiento."""
 history_file = LOGS_PATH / 'history_mini.json'
 
 if not history_file.exists():
 print(" No se encontró historial de entrenamiento")
 return
 
 with open(history_file, 'r') as f:
 history = json.load(f)
 
 epochs = range(1, len(history['loss']) + 1)
 
 fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
 # Loss
 axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss')
 axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
 axes[0].set_xlabel('Época', fontsize=12)
 axes[0].set_ylabel('Loss', fontsize=12)
 axes[0].set_title('Evolución del Loss', fontsize=13, fontweight='bold')
 axes[0].legend(fontsize=11)
 axes[0].grid(True, alpha=0.3)
 
 # Accuracy
 axes[1].plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
 axes[1].plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Val Accuracy')
 axes[1].set_xlabel('Época', fontsize=12)
 axes[1].set_ylabel('Accuracy', fontsize=12)
 axes[1].set_title('Evolución del Accuracy', fontsize=13, fontweight='bold')
 axes[1].legend(fontsize=11)
 axes[1].grid(True, alpha=0.3)
 axes[1].set_ylim([0, 1])
 
 plt.suptitle('Historial de Entrenamiento - CNN Geotérmico', fontsize=14, fontweight='bold')
 plt.tight_layout()
 plt.savefig(save_path)
 plt.close()
 print(f" Guardado: {save_path.name}")


def plot_metrics_comparison(metrics, save_path):
 """Gráfico de barras comparando métricas."""
 metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
 metric_values = [
 metrics['accuracy'],
 metrics['precision'],
 metrics['recall'],
 metrics['f1_score'],
 metrics['roc_auc']
 ]
 
 colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
 
 plt.figure(figsize=(10, 6))
 bars = plt.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.2)
 
 # Añadir valores encima de las barras
 for bar, val in zip(bars, metric_values):
 plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
 f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
 
 plt.ylim([0, 1.15])
 plt.xlabel('Métrica', fontsize=12)
 plt.ylabel('Valor', fontsize=12)
 plt.title('Métricas de Rendimiento\nModelo CNN Geotérmico', fontsize=14, fontweight='bold')
 plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
 plt.grid(axis='y', alpha=0.3)
 plt.tight_layout()
 plt.savefig(save_path)
 plt.close()
 print(f" Guardado: {save_path.name}")


def main():
 print("=" * 60)
 print(" EVALUACIÓN COMPLETA DEL MODELO")
 print("=" * 60)
 
 # Crear directorios
 FIGURES_PATH.mkdir(parents=True, exist_ok=True)
 METRICS_PATH.mkdir(parents=True, exist_ok=True)
 
 # Cargar modelo y datos
 model = load_model()
 X_test, y_test = load_data()
 
 print(f"\n Test set: {X_test.shape}")
 print(f" Clase 0: {np.sum(y_test == 0)}")
 print(f" Clase 1: {np.sum(y_test == 1)}")
 
 # Predicciones
 print("\n Generando predicciones...")
 y_prob = model.predict(X_test, verbose=0).flatten()
 y_pred = (y_prob > 0.5).astype(int)
 
 # Calcular métricas
 print("\n Calculando métricas...")
 metrics = calculate_metrics(y_test, y_pred, y_prob)
 
 print("\n" + "=" * 40)
 print("MÉTRICAS DE RENDIMIENTO")
 print("=" * 40)
 for name, value in metrics.items():
 print(f" {name.upper():12} : {value:.4f} ({value:.1%})")
 
 # Classification Report
 print("\n" + "=" * 40)
 print("CLASSIFICATION REPORT")
 print("=" * 40)
 print(classification_report(
 y_test, y_pred, 
 target_names=['Sin Potencial', 'Con Potencial'],
 zero_division=0
 ))
 
 # Guardar métricas en JSON
 metrics_file = METRICS_PATH / 'evaluation_metrics.json'
 with open(metrics_file, 'w') as f:
 json.dump(metrics, f, indent=2)
 print(f"\n Métricas guardadas: {metrics_file}")
 
 # Guardar métricas en CSV (para la tesis)
 metrics_df = pd.DataFrame([metrics])
 metrics_csv = METRICS_PATH / 'metrics_table.csv'
 metrics_df.to_csv(metrics_csv, index=False)
 print(f" Tabla CSV guardada: {metrics_csv}")
 
 # Generar visualizaciones
 print("\n Generando visualizaciones...")
 
 plot_confusion_matrix(y_test, y_pred, FIGURES_PATH / 'confusion_matrix.png')
 plot_roc_curve(y_test, y_prob, FIGURES_PATH / 'roc_curve.png')
 plot_training_history(FIGURES_PATH / 'training_history.png')
 plot_metrics_comparison(metrics, FIGURES_PATH / 'metrics_comparison.png')
 
 print("\n" + "=" * 60)
 print(" ¡EVALUACIÓN COMPLETADA!")
 print("=" * 60)
 print(f"\n Resultados guardados en: {RESULTS_PATH}")
 print(f" Figuras: {FIGURES_PATH}")
 print(f" Métricas: {METRICS_PATH}")
 
 print("\n Archivos para tu tesis:")
 print(" - confusion_matrix.png → Matriz de confusión")
 print(" - roc_curve.png → Curva ROC")
 print(" - training_history.png → Historia de entrenamiento")
 print(" - metrics_comparison.png → Comparación de métricas")
 print(" - metrics_table.csv → Tabla con valores")


if __name__ == "__main__":
 main()
