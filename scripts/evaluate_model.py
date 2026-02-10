"""
Model Evaluation Script
=======================

Calcula todas las métricas requeridas para la tesis:
- Accuracy, Precision, Recall, F1-Score
- Matriz de Confusión
- Curva ROC y AUC
- Coeficiente R²

Autores: Cristian Vega, Daniel Santiago Arévalo Rubiano
Universidad de San Buenaventura - Bogotá
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import (
 accuracy_score, precision_score, recall_score, f1_score,
 confusion_matrix, classification_report, roc_curve, auc,
 roc_auc_score, r2_score
)
import json
import logging

# Configurar logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
 """
 Clase para evaluar el modelo CNN entrenado.
 """
 
 def __init__(
 self,
 model_path: str,
 processed_data_path: str = 'data/processed',
 results_path: str = 'results/metrics'
 ):
 """
 Inicializa el evaluador.
 
 Args:
 model_path: Ruta al modelo entrenado (.keras)
 processed_data_path: Ruta a datos procesados
 results_path: Ruta para guardar resultados
 """
 self.model_path = Path(model_path)
 self.processed_data_path = Path(processed_data_path)
 self.results_path = Path(results_path)
 
 self.results_path.mkdir(parents=True, exist_ok=True)
 
 self.model = None
 self.X_test = None
 self.y_test = None
 self.y_pred = None
 self.y_pred_proba = None
 
 logger.info("ModelEvaluator inicializado")
 
 def load_model(self):
 """Carga el modelo entrenado."""
 logger.info(f"Cargando modelo desde: {self.model_path}")
 
 try:
 self.model = keras.models.load_model(str(self.model_path))
 logger.info(" Modelo cargado exitosamente")
 return True
 except Exception as e:
 logger.error(f"Error cargando modelo: {e}")
 return False
 
 def load_test_data(self):
 """Carga los datos de test."""
 logger.info("Cargando datos de test...")
 
 try:
 self.X_test = np.load(self.processed_data_path / 'X_test.npy')
 self.y_test = np.load(self.processed_data_path / 'y_test.npy')
 
 logger.info(f" Datos cargados: {self.X_test.shape}")
 return True
 except Exception as e:
 logger.error(f"Error cargando datos: {e}")
 return False
 
 def predict(self):
 """Realiza predicciones en el conjunto de test."""
 logger.info("Realizando predicciones...")
 
 # Predicciones (probabilidades)
 self.y_pred_proba = self.model.predict(self.X_test, verbose=0)
 
 # Convertir probabilidades a clases
 if self.y_pred_proba.shape[1] == 1:
 # Clasificación binaria
 self.y_pred = (self.y_pred_proba > 0.5).astype(int).flatten()
 self.y_pred_proba = self.y_pred_proba.flatten()
 else:
 # Clasificación multiclase
 self.y_pred = np.argmax(self.y_pred_proba, axis=1)
 
 logger.info(" Predicciones completadas")
 
 def calculate_metrics(self) -> dict:
 """
 Calcula todas las métricas requeridas.
 
 Returns:
 Diccionario con todas las métricas
 """
 logger.info("\n" + "="*70)
 logger.info("CALCULANDO MÉTRICAS")
 logger.info("="*70)
 
 metrics = {}
 
 # 1. Accuracy
 accuracy = accuracy_score(self.y_test, self.y_pred)
 metrics['accuracy'] = accuracy
 logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
 
 # 2. Precision
 precision = precision_score(self.y_test, self.y_pred, average='binary', zero_division=0)
 metrics['precision'] = precision
 logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
 
 # 3. Recall
 recall = recall_score(self.y_test, self.y_pred, average='binary', zero_division=0)
 metrics['recall'] = recall
 logger.info(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
 
 # 4. F1-Score
 f1 = f1_score(self.y_test, self.y_pred, average='binary', zero_division=0)
 metrics['f1_score'] = f1
 logger.info(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
 
 # 5. ROC AUC
 try:
 roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
 metrics['roc_auc'] = roc_auc
 logger.info(f"ROC AUC: {roc_auc:.4f}")
 except Exception as e:
 logger.warning(f"No se pudo calcular ROC AUC: {e}")
 metrics['roc_auc'] = None
 
 # 6. R² (Coeficiente de determinación)
 try:
 r2 = r2_score(self.y_test, self.y_pred)
 metrics['r2_score'] = r2
 logger.info(f"R² Score: {r2:.4f}")
 except Exception as e:
 logger.warning(f"No se pudo calcular R²: {e}")
 metrics['r2_score'] = None
 
 # 7. Matriz de Confusión
 cm = confusion_matrix(self.y_test, self.y_pred)
 metrics['confusion_matrix'] = cm.tolist()
 
 logger.info("\nMatriz de Confusión:")
 logger.info(f" TN: {cm[0,0]} FP: {cm[0,1]}")
 logger.info(f" FN: {cm[1,0]} TP: {cm[1,1]}")
 
 # 8. Curva ROC
 try:
 fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
 metrics['roc_curve'] = {
 'fpr': fpr.tolist(),
 'tpr': tpr.tolist(),
 'thresholds': thresholds.tolist()
 }
 except Exception as e:
 logger.warning(f"No se pudo calcular ROC Curve: {e}")
 metrics['roc_curve'] = None
 
 # 9. Reporte de clasificación completo
 report = classification_report(
 self.y_test,
 self.y_pred,
 target_names=['Sin Potencial', 'Con Potencial'],
 output_dict=True,
 zero_division=0
 )
 metrics['classification_report'] = report
 
 logger.info("\nReporte de Clasificación:")
 logger.info(classification_report(
 self.y_test,
 self.y_pred,
 target_names=['Sin Potencial', 'Con Potencial'],
 zero_division=0
 ))
 
 return metrics
 
 def save_metrics(self, metrics: dict, filename: str = 'evaluation_metrics.json'):
 """
 Guarda las métricas en formato JSON.
 
 Args:
 metrics: Diccionario de métricas
 filename: Nombre del archivo
 """
 output_path = self.results_path / filename
 
 with open(output_path, 'w') as f:
 json.dump(metrics, f, indent=2)
 
 logger.info(f"\n Métricas guardadas: {output_path}")
 
 def save_metrics_table(self, metrics: dict, filename: str = 'metrics_table.csv'):
 """
 Guarda las métricas principales en formato CSV (para la tesis).
 
 Args:
 metrics: Diccionario de métricas
 filename: Nombre del archivo
 """
 # Crear DataFrame para la tabla de la tesis
 df_metrics = pd.DataFrame({
 'Métrica': [
 'Accuracy',
 'Precision',
 'Recall',
 'F1-Score',
 'ROC AUC',
 'R² Score'
 ],
 'Valor': [
 f"{metrics.get('accuracy', 0):.4f}",
 f"{metrics.get('precision', 0):.4f}",
 f"{metrics.get('recall', 0):.4f}",
 f"{metrics.get('f1_score', 0):.4f}",
 f"{metrics.get('roc_auc', 0):.4f}",
 f"{metrics.get('r2_score', 0):.4f}"
 ],
 'Porcentaje': [
 f"{metrics.get('accuracy', 0)*100:.2f}%",
 f"{metrics.get('precision', 0)*100:.2f}%",
 f"{metrics.get('recall', 0)*100:.2f}%",
 f"{metrics.get('f1_score', 0)*100:.2f}%",
 f"{metrics.get('roc_auc', 0)*100:.2f}%",
 f"{metrics.get('r2_score', 0)*100:.2f}%"
 ]
 })
 
 output_path = self.results_path / filename
 df_metrics.to_csv(output_path, index=False)
 
 logger.info(f" Tabla de métricas guardada: {output_path}")
 
 # Mostrar tabla
 print("\n" + "="*70)
 print("TABLA DE MÉTRICAS PARA LA TESIS")
 print("="*70)
 print(df_metrics.to_string(index=False))
 print("="*70)
 
 def evaluate(self):
 """Ejecuta el pipeline completo de evaluación."""
 logger.info("="*70)
 logger.info("EVALUACIÓN DEL MODELO")
 logger.info("="*70)
 
 # 1. Cargar modelo
 if not self.load_model():
 return None
 
 # 2. Cargar datos de test
 if not self.load_test_data():
 return None
 
 # 3. Realizar predicciones
 self.predict()
 
 # 4. Calcular métricas
 metrics = self.calculate_metrics()
 
 # 5. Guardar resultados
 self.save_metrics(metrics)
 self.save_metrics_table(metrics)
 
 logger.info("\n" + "="*70)
 logger.info(" EVALUACIÓN COMPLETADA")
 logger.info("="*70)
 
 return metrics


def main():
 """Función principal."""
 
 print("="*70)
 print("MODEL EVALUATION")
 print("Geothermal CNN - Universidad de San Buenaventura")
 print("="*70)
 
 # Ruta al mejor modelo
 model_path = 'models/saved_models/geotermia_cnn_custom_best.keras'
 
 if not Path(model_path).exists():
 print(f"\n Error: Modelo no encontrado en {model_path}")
 print("Entrena el modelo primero con: python scripts/train_model.py")
 return
 
 # Crear evaluador
 evaluator = ModelEvaluator(
 model_path=model_path,
 processed_data_path='data/processed',
 results_path='results/metrics'
 )
 
 # Evaluar
 metrics = evaluator.evaluate()
 
 if metrics:
 print("\n Evaluación completada exitosamente!")
 print(f"\nResultados guardados en:")
 print(f" - results/metrics/evaluation_metrics.json")
 print(f" - results/metrics/metrics_table.csv")


if __name__ == '__main__':
 main()
