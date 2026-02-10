"""
Configuración Centralizada del Proyecto Geotermia CNN
=====================================================

Este módulo centraliza TODAS las rutas y parámetros del proyecto.
Soporta dos modos de operación:

  1. LOCAL   – las imágenes viven dentro del proyecto  (data/raw, data/augmented)
  2. EXTERNO – las imágenes viven en un disco duro externo (USB, SSD, etc.)

Para usar un disco externo, establece la variable de entorno
    GEOTERMIA_DATA_ROOT=D:\geotermia_datos
o pasa la ruta al instanciar ProjectConfig(data_root="D:/geotermia_datos").

La estructura esperada en el disco externo es:
    <DISCO>:\geotermia_datos\
        raw\
            positive\   ← .tif descargados (geotérmicos)
            negative\   ← .tif descargados (control)
            labels.csv
        augmented\
            positive\   ← .tif aumentados
            negative\
            labels.csv
        processed\
            X_train.npy, y_train.npy, ...

Autores: Cristian Vega, Daniel Arévalo, Yuliet Espitia, Laura Rivera
Universidad de San Buenaventura – Bogotá, 2025-2026
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# ─────────────────────────────────────────────────────────────
#  RAÍZ DEL PROYECTO  (siempre el directorio donde vive este archivo)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────────────────────
#  VARIABLE DE ENTORNO PARA DISCO EXTERNO
# ─────────────────────────────────────────────────────────────
ENV_DATA_ROOT = "GEOTERMIA_DATA_ROOT"


class ProjectConfig:
    """Configuración centralizada con soporte para disco externo."""

    def __init__(self, data_root: Optional[str] = None):
        """
        Args:
            data_root: Ruta al directorio raíz de datos.
                       Si es None, revisa la variable de entorno GEOTERMIA_DATA_ROOT.
                       Si tampoco existe, usa <PROJECT_ROOT>/data.
        """
        # 1. Determinar data_root
        if data_root:
            self.data_root = Path(data_root).resolve()
            self.source = "argumento"
        elif os.environ.get(ENV_DATA_ROOT):
            self.data_root = Path(os.environ[ENV_DATA_ROOT]).resolve()
            self.source = f"env ${ENV_DATA_ROOT}"
        else:
            self.data_root = PROJECT_ROOT / "data"
            self.source = "local (data/)"

        # 2. Rutas de datos
        self.raw_dir       = self.data_root / "raw"
        self.positive_dir  = self.raw_dir / "positive"
        self.negative_dir  = self.raw_dir / "negative"
        self.augmented_dir = self.data_root / "augmented"
        self.processed_dir = self.data_root / "processed"
        self.labels_csv    = self.raw_dir / "labels.csv"

        # 3. Rutas dentro del proyecto (nunca cambian)
        self.models_dir     = PROJECT_ROOT / "models" / "saved_models"
        self.logs_dir       = PROJECT_ROOT / "logs"
        self.results_dir    = PROJECT_ROOT / "results"
        self.scripts_dir    = PROJECT_ROOT / "scripts"

        # 4. Hiperparámetros por defecto
        self.INPUT_SHAPE     = (224, 224, 5)
        self.BATCH_SIZE      = 32
        self.EPOCHS          = 100
        self.TEST_SIZE       = 0.15
        self.VAL_SIZE        = 0.15
        self.RANDOM_STATE    = 42
        self.LEARNING_RATE   = 1e-3
        self.WEIGHT_DECAY    = 1e-4
        self.LABEL_SMOOTHING = 0.1

    # ── utilidades ──────────────────────────────────────────

    def is_external(self) -> bool:
        """Retorna True si los datos están en un disco externo."""
        return self.source != "local (data/)"

    def validate(self) -> Dict[str, Any]:
        """Verifica que las carpetas existan y reporta el estado."""
        status = {
            "data_root": str(self.data_root),
            "source": self.source,
            "is_external": self.is_external(),
            "dirs": {}
        }
        for name, path in [
            ("raw/positive", self.positive_dir),
            ("raw/negative", self.negative_dir),
            ("augmented",    self.augmented_dir),
            ("processed",    self.processed_dir),
        ]:
            exists = path.exists()
            count  = len(list(path.glob("*.tif"))) if exists else 0
            npy    = len(list(path.glob("*.npy"))) if exists else 0
            status["dirs"][name] = {
                "path": str(path),
                "exists": exists,
                "tif_files": count,
                "npy_files": npy,
            }
        return status

    def ensure_dirs(self):
        """Crea los directorios de datos si no existen."""
        for d in [
            self.raw_dir, self.positive_dir, self.negative_dir,
            self.augmented_dir / "positive", self.augmented_dir / "negative",
            self.processed_dir, self.models_dir, self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Resumen legible del estado de la configuración."""
        v = self.validate()
        lines = [
            "=" * 60,
            "CONFIGURACIÓN DEL PROYECTO GEOTERMIA CNN",
            "=" * 60,
            f"  Fuente de datos : {v['source']}",
            f"  Disco externo   : {'SÍ' if v['is_external'] else 'NO'}",
            f"  Data root       : {v['data_root']}",
            "",
        ]
        for name, info in v["dirs"].items():
            mark = "✅" if info["exists"] else "❌"
            detail = f"{info['tif_files']} .tif, {info['npy_files']} .npy" if info["exists"] else "no existe"
            lines.append(f"  {mark} {name:20s} → {detail}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return f"ProjectConfig(data_root='{self.data_root}', source='{self.source}')"


# ─────────────────────────────────────────────────────────────
#  INSTANCIA GLOBAL  (se puede importar directamente)
# ─────────────────────────────────────────────────────────────
cfg = ProjectConfig()


if __name__ == "__main__":
    print(cfg.summary())
