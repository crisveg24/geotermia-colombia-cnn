"""
Mini Dataset para Pruebas - Descarga 20 imágenes ASTER
=======================================================

10 zonas geotérmicas (volcanes) + 10 zonas control (llanos)
Para validar el pipeline antes de entrenar con dataset completo.
"""

import os
import ee
import geemap
import pandas as pd
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
PROJECT_ID = 'geotermia-col' # Tu proyecto de GCP
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'images'
REGION_SIZE = 0.05 # ~5km de radio

# 10 zonas geotérmicas (label=1) - Volcanes colombianos
GEOTHERMAL_ZONES = {
 "Nevado_del_Ruiz": [-75.3222, 4.8951],
 "Volcan_Purace": [-76.4036, 2.3206],
 "Volcan_Galeras": [-77.3600, 1.2200],
 "Paipa_Iza": [-73.1124, 5.7781],
 "Nevado_del_Tolima": [-75.3300, 4.6600],
 "Volcan_Cumbal": [-77.8800, 0.9500],
 "Volcan_Sotara": [-76.5900, 2.1100],
 "Volcan_Azufral": [-77.6800, 1.0850],
 "Manizales_thermal": [-75.5200, 5.0700],
 "Santa_Rosa_Cabal": [-75.6200, 4.8700],
}

# 10 zonas control (label=0) - Llanos/Amazonia (sin actividad volcánica)
CONTROL_ZONES = {
 "Casanare_Yopal": [-72.3950, 5.3378],
 "Meta_Villavicencio": [-73.6200, 4.1420],
 "Arauca_centro": [-70.7600, 7.0900],
 "Vichada_Puerto_Carreno": [-67.4900, 6.1900],
 "Amazonas_Leticia": [-69.9400, -4.2153],
 "Caqueta_Florencia": [-75.6144, 1.6144],
 "Guaviare_San_Jose": [-72.6400, 2.5700],
 "Atlantico_Barranquilla": [-74.7813, 10.9639],
 "Magdalena_Santa_Marta": [-74.2100, 11.2408],
 "Cesar_Valledupar": [-73.2500, 10.4631],
}


def initialize_earth_engine():
 """Inicializa Google Earth Engine."""
 try:
 ee.Initialize(project=PROJECT_ID)
 logger.info(f"Earth Engine inicializado con proyecto: {PROJECT_ID}")
 return True
 except Exception as e:
 logger.error(f"Error: {e}")
 return False


def download_aster_image(name: str, lon: float, lat: float, label: int, output_dir: Path):
 """
 Descarga una imagen ASTER de una zona específica.
 
 Args:
 name: Nombre de la zona
 lon, lat: Coordenadas
 label: 1=geotérmico, 0=control
 output_dir: Directorio de salida
 """
 try:
 # Crear región de interés (5km aprox)
 point = ee.Geometry.Point([lon, lat])
 region = point.buffer(5000).bounds() # 5km buffer
 
 # Cargar dataset ASTER Global Emissivity (5 bandas térmicas)
 aster = ee.Image('NASA/ASTER_GED/AG100_003')
 
 # Seleccionar bandas térmicas (emisividad)
 thermal_bands = aster.select([
 'emissivity_band10', 'emissivity_band11', 
 'emissivity_band12', 'emissivity_band13', 
 'emissivity_band14'
 ])
 
 # Definir nombre del archivo
 filename = f"{name}.tif"
 filepath = output_dir / filename
 
 # Descargar imagen
 logger.info(f"Descargando: {name} ({lat:.4f}, {lon:.4f}) - Label: {label}")
 
 geemap.ee_export_image(
 thermal_bands,
 filename=str(filepath),
 scale=100, # 100m resolución
 region=region,
 file_per_band=False
 )
 
 # Verificar descarga
 if filepath.exists():
 size_mb = filepath.stat().st_size / (1024 * 1024)
 logger.info(f"Guardado: {filename} ({size_mb:.2f} MB)")
 return True
 else:
 logger.warning(f"Archivo no creado: {filename}")
 return False
 
 except Exception as e:
 logger.error(f"Error descargando {name}: {e}")
 return False


def create_labels_csv(output_dir: Path):
 """Crea archivo CSV con las etiquetas."""
 data = []
 
 for name, coords in GEOTHERMAL_ZONES.items():
 data.append({
 'filename': f"{name}.tif",
 'label': 1,
 'zone_name': name,
 'latitude': coords[1],
 'longitude': coords[0],
 'zone_type': 'geothermal'
 })
 
 for name, coords in CONTROL_ZONES.items():
 data.append({
 'filename': f"{name}.tif",
 'label': 0,
 'zone_name': name,
 'latitude': coords[1],
 'longitude': coords[0],
 'zone_type': 'control'
 })
 
 df = pd.DataFrame(data)
 labels_path = output_dir.parent / 'labels_mini.csv'
 df.to_csv(labels_path, index=False)
 logger.info(f"Labels guardados en: {labels_path}")
 return df


def main():
 """Función principal para descargar mini-dataset."""
 print("=" * 60)
 print("DESCARGA MINI-DATASET GEOTÉRMICO")
 print("=" * 60)
 print(f"• 10 zonas geotérmicas (volcanes)")
 print(f"• 10 zonas control (llanos/amazonia)")
 print(f"• Total: 20 imágenes ASTER")
 print("=" * 60)
 
 # Inicializar Earth Engine
 if not initialize_earth_engine():
 return
 
 # Crear directorio de salida
 OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 logger.info(f"Directorio de salida: {OUTPUT_DIR}")
 
 # Crear labels CSV primero
 df_labels = create_labels_csv(OUTPUT_DIR)
 
 # Contadores
 success = 0
 failed = 0
 
 # Descargar zonas geotérmicas
 print("\nDescargando zonas GEOTÉRMICAS (label=1)...")
 for name, coords in GEOTHERMAL_ZONES.items():
 if download_aster_image(name, coords[0], coords[1], 1, OUTPUT_DIR):
 success += 1
 else:
 failed += 1
 time.sleep(1) # Evitar rate limiting
 
 # Descargar zonas control
 print("\nDescargando zonas CONTROL (label=0)...")
 for name, coords in CONTROL_ZONES.items():
 if download_aster_image(name, coords[0], coords[1], 0, OUTPUT_DIR):
 success += 1
 else:
 failed += 1
 time.sleep(1)
 
 # Resumen
 print("\n" + "=" * 60)
 print("RESUMEN DE DESCARGA")
 print("=" * 60)
 print(f"Exitosas: {success}")
 print(f"Fallidas: {failed}")
 print(f"Ubicación: {OUTPUT_DIR}")
 print("=" * 60)
 
 if success > 0:
 print("\n ¡Mini-dataset listo!")
 print("Siguiente paso: python scripts/prepare_dataset.py")


if __name__ == "__main__":
 main()
