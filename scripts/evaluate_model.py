# Copyright (c) 2025-2026 Vega Sánchez · Arévalo Rubiano · Espitia Ayala · Rivera Martín
# Universidad de San Buenaventura — Bogotá | github.com/crisveg24/geotermia-colombia-cnn
"""
Model Evaluation Script
=======================

Calcula todas las métricas requeridas para la tesis:
- Accuracy, Precision, Recall, F1-Score
- Matriz de Confusión
- Curva ROC y AUC
- Matthews Correlation Coefficient (MCC)
- Intervalos de confianza Bootstrap (95%)

Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
         Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín
Asesor: Prof. Yeison Eduardo Conejo Sandoval
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
    roc_auc_score, matthews_corrcoef
)
from sklearn.utils import resample
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
            logger.info("Modelo cargado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False

    def _load_partitioned_or_single(self, prefix: str) -> np.ndarray:
        """
        Carga un array .npy que puede estar partido en multiples archivos.

        Si existe <prefix>.npy lo carga directamente.
        Si existen <prefix>_part0.npy, <prefix>_part1.npy, ... los concatena.
        """
        single = self.processed_data_path / f'{prefix}.npy'
        if single.exists():
            logger.info(f"  Cargando {prefix}.npy (archivo unico)")
            return np.load(single)

        import glob as _glob
        pattern = str(self.processed_data_path / f'{prefix}_part*.npy')
        parts = sorted(_glob.glob(pattern))
        if not parts:
            raise FileNotFoundError(
                f"No se encontro {single} ni archivos {prefix}_part*.npy"
            )

        logger.info(f"  Cargando {prefix} desde {len(parts)} partes...")
        arrays = []
        for p in parts:
            logger.info(f"    -> {Path(p).name}")
            arrays.append(np.load(p))
        return np.concatenate(arrays, axis=0)

    def load_test_data(self):
        """Carga los datos de test (soporta archivos particionados)."""
        logger.info("Cargando datos de test...")

        try:
            self.X_test = self._load_partitioned_or_single('X_test')
            self.y_test = self._load_partitioned_or_single('y_test')

            logger.info(f"Datos cargados: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
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

        logger.info("Predicciones completadas")

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

        # 6. Matthews Correlation Coefficient (v2: reemplaza R² que no es estándar para clasificación — BUG 26)
        try:
            mcc = matthews_corrcoef(self.y_test, self.y_pred)
            metrics['mcc'] = mcc
            logger.info(f"MCC: {mcc:.4f}")
        except Exception as e:
            logger.warning(f"No se pudo calcular MCC: {e}")
            metrics['mcc'] = None

        # 7. Matriz de Confusión
        cm = confusion_matrix(self.y_test, self.y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        logger.info("\nMatriz de Confusión:")
        logger.info(f"TN: {cm[0,0]} FP: {cm[0,1]}")
        logger.info(f"FN: {cm[1,0]} TP: {cm[1,1]}")

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

    def calculate_confidence_intervals(
        self, n_bootstrap: int = 2000, ci_level: float = 0.95
    ) -> dict:
        """
        Calcula intervalos de confianza Bootstrap para todas las métricas.

        Usa remuestreo con reemplazo del conjunto de test para estimar la
        variabilidad de cada métrica. Esto permite reportar, por ejemplo:
        "Accuracy = 91.45% (IC 95%: 88.2% – 94.1%)"

        Args:
            n_bootstrap: Número de iteraciones de bootstrap (≥1000 recomendado)
            ci_level: Nivel de confianza (0.95 = 95%)

        Returns:
            dict con {metrica: {lower, upper, mean, std}} para cada métrica
        """
        logger.info(f"\nCalculando intervalos de confianza (bootstrap ×{n_bootstrap})...")

        metric_fns = {
            'accuracy': lambda yt, yp, ypr: accuracy_score(yt, yp),
            'precision': lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
            'recall': lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
            'f1_score': lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
            'mcc': lambda yt, yp, ypr: matthews_corrcoef(yt, yp),
        }

        # ROC AUC necesita probabilidades y al menos 2 clases
        def _roc_auc_safe(yt, yp, ypr):
            if len(np.unique(yt)) < 2:
                return np.nan
            return roc_auc_score(yt, ypr)

        metric_fns['roc_auc'] = _roc_auc_safe

        accum = {name: [] for name in metric_fns}
        n = len(self.y_test)

        for i in range(n_bootstrap):
            indices = resample(
                np.arange(n), n_samples=n, replace=True, random_state=i
            )
            yt_b = self.y_test[indices]
            yp_b = self.y_pred[indices]
            ypr_b = self.y_pred_proba[indices]

            # Omitir muestras bootstrap sin ambas clases
            if len(np.unique(yt_b)) < 2:
                continue

            for name, fn in metric_fns.items():
                try:
                    val = fn(yt_b, yp_b, ypr_b)
                    if not np.isnan(val):
                        accum[name].append(val)
                except Exception:
                    pass

        alpha = 1 - ci_level
        ci = {}
        for name, values in accum.items():
            if values:
                arr = np.array(values)
                ci[name] = {
                    'lower': float(np.percentile(arr, alpha / 2 * 100)),
                    'upper': float(np.percentile(arr, (1 - alpha / 2) * 100)),
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                }
                logger.info(
                    f"  {name}: {ci[name]['mean']:.4f} "
                    f"(IC {ci_level*100:.0f}%: {ci[name]['lower']:.4f} – {ci[name]['upper']:.4f})"
                )

        return ci

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

        logger.info(f"\nMétricas guardadas: {output_path}")

    def save_metrics_table(self, metrics: dict, filename: str = 'metrics_table.csv',
                          confidence_intervals: dict = None):
        """
        Guarda las métricas principales en formato CSV (para la tesis).

        Args:
        metrics: Diccionario de métricas
        filename: Nombre del archivo
        confidence_intervals: Intervalos de confianza (bootstrap)
        """
        # v2: Manejar métricas None correctamente (BUG 28)
        def _fmt(val, fmt=".4f"):
            return f"{val:{fmt}}" if val is not None else "N/A"
        def _pct(val):
            return f"{val*100:.2f}%" if val is not None else "N/A"
        def _ci_str(name, ci_dict):
            if ci_dict and name in ci_dict:
                lo = ci_dict[name]['lower']
                hi = ci_dict[name]['upper']
                return f"[{lo:.4f} – {hi:.4f}]"
            return "—"

        metric_names = [
            'Accuracy', 'Precision', 'Recall', 'F1-Score',
            'ROC AUC', 'MCC'
        ]
        metric_keys = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'mcc'
        ]

        # Crear DataFrame para la tabla de la tesis
        df_metrics = pd.DataFrame({
            'Métrica': metric_names,
            'Valor': [_fmt(metrics.get(k)) for k in metric_keys],
            'Porcentaje': [_pct(metrics.get(k)) for k in metric_keys],
            'IC 95%': [_ci_str(k, confidence_intervals) for k in metric_keys],
        })

        output_path = self.results_path / filename
        df_metrics.to_csv(output_path, index=False)

        logger.info(f"Tabla de métricas guardada: {output_path}")

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

        # 5. Calcular intervalos de confianza (bootstrap)
        ci = self.calculate_confidence_intervals(n_bootstrap=2000, ci_level=0.95)
        metrics['confidence_intervals'] = ci

        # 6. Guardar resultados
        self.save_metrics(metrics)
        self.save_metrics_table(metrics, confidence_intervals=ci)

        logger.info("\n" + "="*70)
        logger.info("EVALUACIÓN COMPLETADA")
        logger.info("="*70)

        return metrics


def main():
    """Función principal."""

    # Importar configuración centralizada (soporta disco externo)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import cfg

    print("="*70)
    print("MODEL EVALUATION")
    print("Geothermal CNN - Universidad de San Buenaventura")
    print("="*70)
    print(cfg.summary())

    # Ruta al mejor modelo (V3: EfficientNetB0 + adapter)
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'models' / 'saved_models' / 'geotermia_v7_phase2_best.keras'

    if not model_path.exists():
        # Fallback a modelo custom anterior
        model_path = project_root / 'models' / 'saved_models' / 'geotermia_cnn_custom_best.keras'

    if not model_path.exists():
        print(f"\nError: Modelo no encontrado en {model_path}")
        print("Entrena el modelo primero con: python scripts/train_model.py")
        return

    # Crear evaluador (datos procesados desde config.py = disco externo)
    evaluator = ModelEvaluator(
        model_path=str(model_path),
        processed_data_path=str(cfg.processed_dir),
        results_path=str(project_root / 'results' / 'metrics')
    )

    # Evaluar
    metrics = evaluator.evaluate()

    if metrics:
        print("\nEvaluación completada exitosamente!")
        print(f"\nResultados guardados en:")
        print(f" - results/metrics/evaluation_metrics.json")
        print(f" - results/metrics/metrics_table.csv")


if __name__ == '__main__':
    main()
