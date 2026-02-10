"""
Results Visualization Script
=============================

Genera todas las visualizaciones requeridas para la tesis:
- Training History (Loss y Accuracy)
- Confusion Matrix
- ROC Curve
- Predicciones de ejemplo

Autores: Cristian Vega, Daniel Santiago Arévalo Rubiano
Universidad de San Buenaventura - Bogotá
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, Optional

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configurar logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsVisualizer:
 """
 Clase para generar visualizaciones de resultados.
 """
 
 def __init__(
 self,
 results_path: str = 'results',
 logs_path: str = 'logs',
 dpi: int = 300
 ):
 """
 Inicializa el visualizador.
 
 Args:
 results_path: Ruta para guardar figuras
 logs_path: Ruta a logs de entrenamiento
 dpi: Resolución de las imágenes
 """
 self.results_path = Path(results_path)
 self.figures_path = self.results_path / 'figures'
 self.metrics_path = self.results_path / 'metrics'
 self.logs_path = Path(logs_path)
 self.dpi = dpi
 
 self.figures_path.mkdir(parents=True, exist_ok=True)
 
 logger.info("ResultsVisualizer inicializado")
 
 def plot_training_history(
 self,
 history_file: str = 'history_custom.json',
 figsize: tuple = (15, 5)
 ):
 """
 Grafica la historia de entrenamiento (Loss y Accuracy).
 
 Args:
 history_file: Nombre del archivo de historial
 figsize: Tamaño de la figura
 """
 logger.info("Generando gráfico de Training History...")
 
 # Cargar historial
 history_path = self.logs_path / history_file
 
 if not history_path.exists():
 logger.error(f"Archivo no encontrado: {history_path}")
 return
 
 with open(history_path, 'r') as f:
 history = json.load(f)
 
 # Crear figura con 2 subplots
 fig, axes = plt.subplots(1, 2, figsize=figsize)
 
 # Plot 1: Loss
 axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
 axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
 axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
 axes[0].set_xlabel('Epoch', fontsize=12)
 axes[0].set_ylabel('Loss', fontsize=12)
 axes[0].legend(loc='upper right', fontsize=10)
 axes[0].grid(True, alpha=0.3)
 
 # Plot 2: Accuracy
 axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
 axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
 axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
 axes[1].set_xlabel('Epoch', fontsize=12)
 axes[1].set_ylabel('Accuracy', fontsize=12)
 axes[1].legend(loc='lower right', fontsize=10)
 axes[1].grid(True, alpha=0.3)
 
 plt.tight_layout()
 
 # Guardar
 output_path = self.figures_path / 'training_history.png'
 plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
 logger.info(f" Gráfico guardado: {output_path}")
 
 plt.close()
 
 def plot_confusion_matrix(
 self,
 metrics_file: str = 'evaluation_metrics.json',
 figsize: tuple = (10, 8)
 ):
 """
 Grafica la matriz de confusión.
 
 Args:
 metrics_file: Nombre del archivo de métricas
 figsize: Tamaño de la figura
 """
 logger.info("Generando Confusion Matrix...")
 
 # Cargar métricas
 metrics_path = self.metrics_path / metrics_file
 
 if not metrics_path.exists():
 logger.error(f"Archivo no encontrado: {metrics_path}")
 return
 
 with open(metrics_path, 'r') as f:
 metrics = json.load(f)
 
 cm = np.array(metrics['confusion_matrix'])
 
 # Crear figura
 fig, ax = plt.subplots(figsize=figsize)
 
 # Plot heatmap
 sns.heatmap(
 cm,
 annot=True,
 fmt='d',
 cmap='Blues',
 square=True,
 linewidths=2,
 cbar_kws={'label': 'Count'},
 ax=ax,
 annot_kws={'size': 16, 'weight': 'bold'}
 )
 
 # Etiquetas
 ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
 ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
 ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
 ax.set_xticklabels(['Sin Potencial (0)', 'Con Potencial (1)'], fontsize=12)
 ax.set_yticklabels(['Sin Potencial (0)', 'Con Potencial (1)'], fontsize=12, rotation=0)
 
 plt.tight_layout()
 
 # Guardar
 output_path = self.figures_path / 'confusion_matrix.png'
 plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
 logger.info(f" Confusion Matrix guardada: {output_path}")
 
 plt.close()
 
 def plot_roc_curve(
 self,
 metrics_file: str = 'evaluation_metrics.json',
 figsize: tuple = (10, 8)
 ):
 """
 Grafica la curva ROC.
 
 Args:
 metrics_file: Nombre del archivo de métricas
 figsize: Tamaño de la figura
 """
 logger.info("Generando ROC Curve...")
 
 # Cargar métricas
 metrics_path = self.metrics_path / metrics_file
 
 if not metrics_path.exists():
 logger.error(f"Archivo no encontrado: {metrics_path}")
 return
 
 with open(metrics_path, 'r') as f:
 metrics = json.load(f)
 
 if metrics.get('roc_curve') is None:
 logger.warning("ROC Curve no disponible en las métricas")
 return
 
 roc_data = metrics['roc_curve']
 fpr = np.array(roc_data['fpr'])
 tpr = np.array(roc_data['tpr'])
 roc_auc = metrics.get('roc_auc', 0)
 
 # Crear figura
 fig, ax = plt.subplots(figsize=figsize)
 
 # Plot ROC curve
 ax.plot(
 fpr, tpr,
 color='darkorange',
 linewidth=3,
 label=f'ROC Curve (AUC = {roc_auc:.4f})'
 )
 
 # Plot diagonal (random classifier)
 ax.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier')
 
 # Configuración
 ax.set_xlim([0.0, 1.0])
 ax.set_ylim([0.0, 1.05])
 ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
 ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
 ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold', pad=20)
 ax.legend(loc='lower right', fontsize=12)
 ax.grid(True, alpha=0.3)
 
 plt.tight_layout()
 
 # Guardar
 output_path = self.figures_path / 'roc_curve.png'
 plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
 logger.info(f" ROC Curve guardada: {output_path}")
 
 plt.close()
 
 def plot_metrics_comparison(
 self,
 metrics_file: str = 'evaluation_metrics.json',
 figsize: tuple = (12, 6)
 ):
 """
 Grafica comparación de métricas principales.
 
 Args:
 metrics_file: Nombre del archivo de métricas
 figsize: Tamaño de la figura
 """
 logger.info("Generando gráfico de comparación de métricas...")
 
 # Cargar métricas
 metrics_path = self.metrics_path / metrics_file
 
 if not metrics_path.exists():
 logger.error(f"Archivo no encontrado: {metrics_path}")
 return
 
 with open(metrics_path, 'r') as f:
 metrics = json.load(f)
 
 # Preparar datos
 metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
 metric_values = [
 metrics.get('accuracy', 0),
 metrics.get('precision', 0),
 metrics.get('recall', 0),
 metrics.get('f1_score', 0),
 metrics.get('roc_auc', 0)
 ]
 
 # Crear figura
 fig, ax = plt.subplots(figsize=figsize)
 
 # Barplot
 bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.8)
 
 # Añadir valores encima de las barras
 for bar, value in zip(bars, metric_values):
 height = bar.get_height()
 ax.text(
 bar.get_x() + bar.get_width() / 2.,
 height,
 f'{value:.4f}\n({value*100:.2f}%)',
 ha='center',
 va='bottom',
 fontsize=10,
 fontweight='bold'
 )
 
 # Configuración
 ax.set_ylim([0, 1.1])
 ax.set_ylabel('Score', fontsize=14, fontweight='bold')
 ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
 ax.grid(True, axis='y', alpha=0.3)
 
 plt.tight_layout()
 
 # Guardar
 output_path = self.figures_path / 'metrics_comparison.png'
 plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
 logger.info(f" Métricas comparadas guardadas: {output_path}")
 
 plt.close()
 
 def generate_all_visualizations(self):
 """Genera todas las visualizaciones."""
 logger.info("="*70)
 logger.info("GENERANDO VISUALIZACIONES")
 logger.info("="*70)
 
 # 1. Training History
 self.plot_training_history()
 
 # 2. Confusion Matrix
 self.plot_confusion_matrix()
 
 # 3. ROC Curve
 self.plot_roc_curve()
 
 # 4. Metrics Comparison
 self.plot_metrics_comparison()
 
 logger.info("\n" + "="*70)
 logger.info(" VISUALIZACIONES COMPLETADAS")
 logger.info("="*70)
 logger.info(f"\nFiguras guardadas en: {self.figures_path}")


def main():
 """Función principal."""
 
 print("="*70)
 print("RESULTS VISUALIZATION")
 print("Geothermal CNN - Universidad de San Buenaventura")
 print("="*70)
 
 # Crear visualizador
 visualizer = ResultsVisualizer(
 results_path='results',
 logs_path='logs',
 dpi=300
 )
 
 # Generar todas las visualizaciones
 visualizer.generate_all_visualizations()
 
 print("\n Visualizaciones generadas exitosamente!")
 print(f"\nFiguras disponibles:")
 print(f" - results/figures/training_history.png")
 print(f" - results/figures/confusion_matrix.png")
 print(f" - results/figures/roc_curve.png")
 print(f" - results/figures/metrics_comparison.png")


if __name__ == '__main__':
 main()
