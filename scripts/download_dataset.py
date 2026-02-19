"""
Script para Descargar Dataset Completo de Imágenes ASTER para Detección de Potencial Geotérmico
========================================================================================

Este script descarga un conjunto balanceado de imágenes térmicas ASTER de Colombia:
- Zonas CON potencial geotérmico (label=1): volcanes y zonas geotérmicas conocidas
- Zonas SIN potencial geotérmico (label=0): zonas de control sin actividad volcánica

Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
         Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martín
Asesor: Prof. Yeison Eduardo Conejo Sandoval
Universidad de San Buenaventura - Bogotá
Fecha: Noviembre 2025
"""

import os
import json
import time
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import ee
import geemap
import pandas as pd
from pathlib import Path

# v2: Usar PROJECT_ROOT para logs (BUG 21)
PROJECT_ROOT = Path(__file__).parent.parent
os.makedirs(PROJECT_ROOT / 'logs', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler(PROJECT_ROOT / 'logs' / 'download_dataset.log'),
    logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GeotermalDatasetDownloader:
    """Clase para descargar dataset de imágenes ASTER para análisis geotérmico."""

    def __init__(self, output_dir: str = None):
        """
        Inicializar descargador de dataset.

        Args:
        output_dir: Directorio donde guardar las imágenes descargadas.
        Si es None, usa la ruta de config.py (soporta disco externo).
        """
        # Importar configuración centralizada
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import cfg

        self.output_dir = Path(output_dir) if output_dir else cfg.raw_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Crear subdirectorios para imágenes positivas y negativas
        self.positive_dir = self.output_dir / "positive"
        self.negative_dir = self.output_dir / "negative"
        self.positive_dir.mkdir(exist_ok=True)
        self.negative_dir.mkdir(exist_ok=True)

        # Inicializar Earth Engine con el proyecto configurado
        try:
            # v2: Usar GEE_PROJECT de config.py en vez de hardcodeado (BUG 10)
            try:
                ee.Initialize(project=cfg.GEE_PROJECT)
                logger.info(f"Google Earth Engine inicializado con proyecto: {cfg.GEE_PROJECT}")
            except Exception:  # v2: bare except → except Exception (BUG 9)
                # Si falla, intentar sin especificar proyecto
                ee.Initialize()
                logger.info("Google Earth Engine inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando Earth Engine: {e}")
            logger.error("")
            logger.error("SOLUCIÓN:")
            logger.error("1. Verifica que el proyecto 'alpine-air-469115-f0' esté activo")
            logger.error("2. Habilita la API de Earth Engine en: https://console.cloud.google.com/")
            logger.error("3. Ejecuta: python -c \"import ee; ee.Authenticate()\" y luego python setup.py")
            raise

        # Cargar dataset ASTER
        self.aster_dataset = ee.Image('NASA/ASTER_GED/AG100_003')

        # Definir zonas geotérmicas de Colombia (CON potencial - label=1)
        self.geothermal_zones = {
            # ── Nevado del Ruiz (volcán activo, campo geotérmico principal) ──
            "Nevado_del_Ruiz_center": [-75.3222, 4.8951],
            "Nevado_del_Ruiz_north": [-75.3150, 4.9100],
            "Nevado_del_Ruiz_south": [-75.3300, 4.8800],
            "Nevado_del_Ruiz_east": [-75.3000, 4.8950],
            "Nevado_del_Ruiz_west": [-75.3450, 4.8950],
            "Nevado_del_Ruiz_NE": [-75.3000, 4.9100],
            "Nevado_del_Ruiz_SW": [-75.3450, 4.8800],

            # ── Volcán Puracé (campo geotérmico confirmado por SGC) ──
            "Volcan_Purace_center": [-76.4036, 2.3206],
            "Volcan_Purace_north": [-76.4000, 2.3350],
            "Volcan_Purace_south": [-76.4070, 2.3050],
            "Volcan_Purace_east": [-76.3850, 2.3200],
            "Volcan_Purace_west": [-76.4200, 2.3200],
            "Volcan_Purace_NE": [-76.3850, 2.3350],
            "Volcan_Purace_SW": [-76.4200, 2.3050],

            # ── Volcán Galeras (actividad volcánica permanente) ──
            "Volcan_Galeras_center": [-77.3600, 1.2200],
            "Volcan_Galeras_north": [-77.3550, 1.2350],
            "Volcan_Galeras_south": [-77.3650, 1.2050],
            "Volcan_Galeras_east": [-77.3400, 1.2200],
            "Volcan_Galeras_west": [-77.3800, 1.2200],
            "Volcan_Galeras_NE": [-77.3400, 1.2350],
            "Volcan_Galeras_SW": [-77.3800, 1.2050],

            # ── Paipa-Iza (campo geotérmico de baja entalpía) ──
            "Paipa_Iza_center": [-73.1124, 5.7781],
            "Paipa_Iza_north": [-73.1100, 5.7900],
            "Paipa_Iza_south": [-73.1150, 5.7650],
            "Paipa_Iza_east": [-73.0950, 5.7780],
            "Paipa_Iza_west": [-73.1300, 5.7780],
            "Paipa_Iza_NE": [-73.0950, 5.7900],
            "Paipa_Iza_SW": [-73.1300, 5.7650],

            # ── Nevado del Tolima ──
            "Nevado_del_Tolima_center": [-75.3300, 4.6600],
            "Nevado_del_Tolima_north": [-75.3250, 4.6750],
            "Nevado_del_Tolima_south": [-75.3350, 4.6450],
            "Nevado_del_Tolima_east": [-75.3100, 4.6600],
            "Nevado_del_Tolima_west": [-75.3500, 4.6600],

            # ── Volcán Cumbal ──
            "Volcan_Cumbal_center": [-77.8800, 0.9500],
            "Volcan_Cumbal_north": [-77.8750, 0.9650],
            "Volcan_Cumbal_south": [-77.8850, 0.9350],
            "Volcan_Cumbal_east": [-77.8600, 0.9500],
            "Volcan_Cumbal_west": [-77.9000, 0.9500],

            # ── Volcán Sotará ──
            "Volcan_Sotara_center": [-76.5900, 2.1100],
            "Volcan_Sotara_north": [-76.5850, 2.1250],
            "Volcan_Sotara_south": [-76.5950, 2.0950],
            "Volcan_Sotara_east": [-76.5700, 2.1100],
            "Volcan_Sotara_west": [-76.6100, 2.1100],

            # ── Volcán Azufral ──
            "Volcan_Azufral_center": [-77.6800, 1.0850],
            "Volcan_Azufral_north": [-77.6750, 1.1000],
            "Volcan_Azufral_south": [-77.6850, 1.0700],
            "Volcan_Azufral_east": [-77.6600, 1.0850],
            "Volcan_Azufral_west": [-77.7000, 1.0850],

            # ── Volcán Cerro Machín (alto riesgo, actividad hidrotermal) ──
            "Cerro_Machin_center": [-75.3900, 4.4900],
            "Cerro_Machin_north": [-75.3850, 4.5050],
            "Cerro_Machin_south": [-75.3950, 4.4750],
            "Cerro_Machin_east": [-75.3700, 4.4900],
            "Cerro_Machin_west": [-75.4100, 4.4900],

            # ── Volcán Doña Juana ──
            "Dona_Juana_center": [-76.9400, 1.5000],
            "Dona_Juana_north": [-76.9350, 1.5150],
            "Dona_Juana_south": [-76.9450, 1.4850],
            "Dona_Juana_east": [-76.9200, 1.5000],
            "Dona_Juana_west": [-76.9600, 1.5000],

            # ── Volcán Las Ánimas / Complejo Pan de Azúcar ──
            "Las_Animas_center": [-76.5800, 2.3700],
            "Las_Animas_north": [-76.5750, 2.3850],
            "Las_Animas_south": [-76.5850, 2.3550],

            # ── Volcán Cerro Bravo ──
            "Cerro_Bravo_center": [-75.3000, 5.0900],
            "Cerro_Bravo_north": [-75.2950, 5.1050],
            "Cerro_Bravo_south": [-75.3050, 5.0750],
            "Cerro_Bravo_east": [-75.2800, 5.0900],

            # ── Volcán Nevado de Santa Isabel ──
            "Santa_Isabel_center": [-75.3700, 4.8150],
            "Santa_Isabel_north": [-75.3650, 4.8300],
            "Santa_Isabel_south": [-75.3750, 4.8000],

            # ── Volcán Nevado del Huila ──
            "Nevado_Huila_center": [-76.0300, 2.9400],
            "Nevado_Huila_north": [-76.0250, 2.9550],
            "Nevado_Huila_south": [-76.0350, 2.9250],
            "Nevado_Huila_east": [-76.0100, 2.9400],

            # ── Volcán Chiles ──
            "Volcan_Chiles_center": [-77.9400, 0.8200],
            "Volcan_Chiles_north": [-77.9350, 0.8350],
            "Volcan_Chiles_south": [-77.9450, 0.8050],

            # ── Volcán Cerro Negro de Mayasquer ──
            "Cerro_Negro_center": [-77.9600, 0.8300],
            "Cerro_Negro_east": [-77.9400, 0.8300],

            # ── Zonas geotérmicas / termales confirmadas ──
            "Manizales_thermal": [-75.5200, 5.0700],
            "Coconuco_thermal": [-76.3850, 2.4300],
            "Santa_Rosa_Cabal": [-75.6200, 4.8700],
            "Herveo_thermal": [-75.1300, 4.8800],
            "Villa_Maria_thermal": [-74.9800, 4.9000],
            "Termales_San_Vicente": [-73.4200, 6.8800],
            "Termales_Tabio": [-74.0900, 4.9200],
            "Termales_Choachi": [-73.9200, 4.5300],
            "Termales_Rivera_Huila": [-75.2600, 2.7800],
            "Termales_Guadalupe_Santander": [-73.4100, 6.2500],
            "San_Agustin_Huila_thermal": [-76.2700, 1.8800],
            "Pitalito_thermal": [-76.0400, 1.8600],

            # ── Volcán Romeral / Complejo volcánico norte ──
            "Romeral_center": [-75.3640, 4.9600],
            "Romeral_north": [-75.3590, 4.9750],
            "Romeral_south": [-75.3690, 4.9450],

            # ── Termas / manantiales geotérmicos adicionales ──
            "Termales_Coconuco_sur": [-76.3900, 2.3000],
            "Termales_Purace_flanco": [-76.3700, 2.3100],
            "Termales_Santa_Rosa_norte": [-75.6100, 4.8900],
            "Termales_Manizales_sur": [-75.5150, 5.0500],
            "Termales_Ruiz_flanco_E": [-75.2800, 4.8900],
            "Termales_Ruiz_flanco_W": [-75.3600, 4.9000],
            "Termales_Galeras_flanco_N": [-77.3450, 1.2400],
            "Termales_Galeras_flanco_S": [-77.3700, 1.2000],

            # ── Cerro Bravo / Santa Isabel flancos ──
            "Cerro_Bravo_west": [-75.3200, 5.0900],
            "Santa_Isabel_east": [-75.3500, 4.8150],

            # ── Volcán Nevado del Huila flancos ──
            "Nevado_Huila_west": [-76.0500, 2.9400],
            "Nevado_Huila_NE": [-76.0100, 2.9550],

            # ── Complejo Cerro Machín flancos ──
            "Cerro_Machin_NE": [-75.3700, 4.5050],
            "Cerro_Machin_SW": [-75.4100, 4.4750],

            # ── Volcán Doña Juana flancos ──
            "Dona_Juana_NE": [-76.9200, 1.5150],
            "Dona_Juana_SW": [-76.9600, 1.4850],

            # ── Volcán Chiles flancos ──
            "Volcan_Chiles_east": [-77.9200, 0.8200],
            "Volcan_Chiles_west": [-77.9600, 0.8200],

            # ── Sierra Nevada del Cocuy (fumarolas reportadas) ──
            "Cocuy_thermal": [-72.3100, 6.4200],
        }

        # Definir zonas de control (SIN potencial - label=0)
        self.control_zones = {
            # ── Llanos Orientales (llanura sedimentaria) ──
            "Casanare_Yopal": [-72.3950, 5.3378],
            "Casanare_Paz_Ariporo": [-71.8800, 5.8800],
            "Casanare_Hato_Corozal": [-71.7600, 6.1700],
            "Casanare_Trinidad": [-71.6600, 5.4300],
            "Arauca_Arauca": [-70.7600, 7.0900],
            "Arauca_Saravena": [-71.8800, 6.9600],
            "Vichada_Puerto_Carreno": [-67.4900, 6.1900],
            "Meta_Villavicencio": [-73.6200, 4.1420],
            "Meta_Puerto_Lopez": [-72.9600, 4.0800],
            "Meta_Puerto_Gaitan": [-72.0800, 4.3200],
            "Meta_Acacias": [-73.7600, 3.9900],
            "Meta_Granada": [-73.7000, 3.5500],
            "Casanare_Aguazul": [-72.5500, 5.1700],
            "Casanare_Villanueva": [-72.9200, 4.9500],

            # ── Amazonas y Orinoquía (selva tropical) ──
            "Amazonas_Leticia": [-69.9400, -4.2153],
            "Amazonas_Puerto_Narino": [-70.3800, -3.7700],
            "Caqueta_Florencia": [-75.6144, 1.6144],
            "Caqueta_San_Vicente": [-74.7700, 0.6400],
            "Putumayo_Mocoa": [-76.6500, 1.1500],
            "Putumayo_Puerto_Asis": [-76.4989, 0.5054],
            "Guaviare_San_Jose": [-72.6400, 2.5700],
            "Vaupes_Mitu": [-70.1700, 1.2500],
            "Guainia_Inirida": [-67.9200, 3.8700],
            "Amazonas_Tarapaca": [-69.7500, -2.8800],

            # ── Costa Caribe (llanura costera) ──
            "Atlantico_Barranquilla": [-74.7813, 10.9639],
            "Magdalena_Santa_Marta": [-74.2100, 11.2408],
            "Cesar_Valledupar": [-73.2500, 10.4631],
            "La_Guajira_Riohacha": [-72.9072, 11.5444],
            "Cordoba_Monteria": [-75.8814, 8.7479],
            "Sucre_Sincelejo": [-75.3978, 9.3047],
            "Bolivar_Cartagena": [-75.5144, 10.3910],
            "Bolivar_Magangue": [-74.7542, 9.2417],
            "La_Guajira_Maicao": [-72.2400, 11.3800],
            "Cesar_Aguachica": [-73.6200, 8.3100],
            "Magdalena_Cienaga": [-74.2500, 11.0100],
            "Atlantico_Sabanalarga": [-74.9200, 10.6300],

            # ── Zona Andina Oriental (sin vulcanismo) ──
            "Santander_Bucaramanga": [-73.1198, 7.1254],
            "Santander_Barrancabermeja": [-73.8542, 7.0653],
            "Norte_Santander_Cucuta": [-72.5047, 7.8939],
            "Boyaca_Tunja": [-73.3678, 5.5353],
            "Boyaca_Sogamoso": [-72.9342, 5.7142],
            "Cundinamarca_Girardot": [-74.8039, 4.3011],
            "Cundinamarca_Zipaquira": [-74.0042, 5.0214],
            "Santander_San_Gil": [-73.1300, 6.5600],
            "Santander_Socorro": [-73.2600, 6.4700],
            "Boyaca_Duitama": [-73.0300, 5.8300],
            "Boyaca_Chiquinquira": [-73.8200, 5.6200],
            "Norte_Santander_Ocana": [-73.3600, 8.2400],

            # ── Valle del Cauca (zona plana) ──
            "Valle_Cali_norte": [-76.5225, 3.5000],
            "Valle_Palmira": [-76.3036, 3.5394],
            "Valle_Tulua": [-76.1953, 4.0864],
            "Valle_Buga": [-76.3000, 3.9014],
            "Valle_Cartago": [-75.9114, 4.7467],
            "Valle_Cali_sur": [-76.5400, 3.3800],
            "Valle_Jamundi": [-76.5400, 3.2600],
            "Cauca_Popayan_control": [-76.6100, 2.4400],

            # ── Chocó / Pacífico ──
            "Choco_Quibdo": [-76.6611, 5.6919],
            "Choco_Bahia_Solano": [-77.4094, 6.1989],
            "Narino_Tumaco": [-78.7700, 1.7900],
            "Choco_Istmina": [-76.6800, 5.1600],

            # ── Eje cafetero (zona baja, no volcánica) ──
            "Risaralda_Pereira_control": [-75.6900, 4.8100],
            "Quindio_Armenia_control": [-75.6810, 4.5339],
            "Caldas_La_Dorada": [-74.6600, 5.4500],

            # ── Antioquia ──
            "Antioquia_Medellin": [-75.5636, 6.2442],
            "Antioquia_Rionegro": [-75.3700, 6.1500],
            "Antioquia_Turbo": [-76.7300, 8.0900],

            # ── Sabana de Bogotá (control urbano/altiplano) ──
            "Bogota_centro": [-74.0721, 4.7110],
            "Bogota_norte": [-74.0300, 4.7600],
            "Cundinamarca_Chia": [-74.0600, 4.8600],
            "Cundinamarca_Facatativa": [-74.3600, 4.8100],
            "Cundinamarca_Soacha": [-74.2200, 4.5900],

            # ── Llanos Orientales adicionales ──
            "Meta_San_Martin": [-73.6900, 3.6900],
            "Casanare_Monterrey": [-72.8900, 4.8800],
            "Arauca_Tame": [-71.7300, 6.4600],
            "Vichada_La_Primavera": [-70.4100, 5.4900],

            # ── Costa Caribe adicionales ──
            "La_Guajira_Uribia": [-72.2700, 11.7100],
            "Cesar_La_Paz": [-73.1700, 10.3800],
            "Magdalena_Plato": [-74.7900, 9.7900],
            "Cordoba_Planeta_Rica": [-75.5900, 8.4100],
            "Sucre_Ovejas": [-75.2300, 9.5300],

            # ── Amazonia adicionales ──
            "Caqueta_Belen_Andaquies": [-75.8700, 1.5900],
            "Putumayo_Puerto_Leguizamo": [-74.7800, -0.1900],
            "Guaviare_El_Retorno": [-72.6200, 2.3300],

            # ── Antioquia / interior norte ──
            "Antioquia_Santa_Fe": [-75.8300, 7.8700],
            "Antioquia_Apartado": [-76.6300, 7.8800],

            # ── Sabana de Bogotá adicionales ──
            "Cundinamarca_Madrid": [-74.2600, 4.7300],
            "Cundinamarca_Funza": [-74.2100, 4.7200],

            # ── Tolima / Huila (zona baja, lejos de volcanes) ──
            "Tolima_Ibague_control": [-75.2300, 4.4380],
            "Huila_Neiva_control": [-75.2800, 2.9270],
        }

        # Metadata del dataset
        self.metadata = {
            'dataset_name': 'Colombia_Geothermal_ASTER',
            'download_date': datetime.now().isoformat(),
            'total_images': 0,
            'positive_images': 0,
            'negative_images': 0,
            'image_details': []
        }

    def download_image(
        self, 
        name: str, 
        coords: List[float], 
        label: int,
        buffer_size: int = 5000,
        scale: int = 90,
        max_retries: int = 3
        ) -> bool:
        """
        Descargar una imagen ASTER de una zona específica.

        Args:
        name: Nombre identificador de la zona
        coords: [longitud, latitud] de la zona
        label: 1 para geotérmica, 0 para control
        buffer_size: Radio del área a descargar (metros)
        scale: Resolución espacial (metros/pixel)
        max_retries: Número máximo de reintentos (v2: BUG 25)

        Returns:
        True si la descarga fue exitosa, False en caso contrario
        """
        # Verificar si ya existe antes de descargar
        output_subdir = self.positive_dir if label == 1 else self.negative_dir
        output_path = output_subdir / f"{name}.tif"
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Ya existe, saltando: {name}")
            return True

        for attempt in range(1, max_retries + 1):
            try:
                # Crear geometría del punto y buffer
                point = ee.Geometry.Point(coords)
                roi = point.buffer(buffer_size)

                # Seleccionar 7 bandas: 5 emisividad TIR + temperatura + NDVI
                # Las bandas TIR detectan anomalías térmicas, la temperatura
                # confirma el gradiente superficial y el NDVI indica alteración
                # hidrotermal (menor vegetación en zonas alteradas)
                aster_bands = [
                    'emissivity_band10', 'emissivity_band11',
                    'emissivity_band12', 'emissivity_band13',
                    'emissivity_band14', 'temperature', 'ndvi'
                ]

                image = self.aster_dataset.select(aster_bands).clip(roi)

                # Descargar imagen usando geemap
                logger.info(f"Descargando: {name} (label={label}, intento {attempt}/{max_retries})...")

                geemap.ee_export_image(
                    image,
                    filename=str(output_path),
                    scale=scale,
                    region=roi,
                    file_per_band=False
                )

                # Verificar que el archivo se descargó correctamente
                if output_path.exists():
                    file_size = output_path.stat().st_size / (1024 * 1024) # MB
                    logger.info(f"Descargado: {name} ({file_size:.2f} MB)")

                    # Guardar metadata
                    self.metadata['image_details'].append({
                        'name': name,
                        'filename': output_path.name,
                        'label': label,
                        'coords': coords,
                        'buffer_size': buffer_size,
                        'scale': scale,
                        'file_size_mb': round(file_size, 2),
                        'bands': aster_bands
                    })

                    return True
                else:
                    logger.warning(f"Archivo no encontrado después de descarga: {name}")
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        logger.info(f"Reintentando en {wait}s...")
                        time.sleep(wait)

            except Exception as e:
                logger.error(f"Error descargando {name} (intento {attempt}): {e}")
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(f"Reintentando en {wait}s...")
                    time.sleep(wait)

        logger.error(f"Fallo definitivo descargando {name} tras {max_retries} intentos")
        return False

    def download_all_zones(
        self, 
        max_positive: int = 50,
        max_negative: int = 50,
        delay: float = 2.0
        ) -> Tuple[int, int]:
        """
        Descargar todas las zonas geotérmicas y de control.

        Args:
        max_positive: Número máximo de imágenes positivas
        max_negative: Número máximo de imágenes negativas
        delay: Tiempo de espera entre descargas (segundos)

        Returns:
        Tupla (num_positivas, num_negativas) descargadas exitosamente
        """
        logger.info("="*80)
        logger.info("INICIANDO DESCARGA DE DATASET COMPLETO")
        logger.info("="*80)

        positive_count = 0
        negative_count = 0

        # Descargar zonas geotérmicas (positivas)
        logger.info(f"\nDescargando zonas CON potencial geotérmico (máximo {max_positive})...")
        for name, coords in list(self.geothermal_zones.items())[:max_positive]:
            if self.download_image(name, coords, label=1):
                positive_count += 1
            time.sleep(delay) # Evitar sobrecargar la API

        logger.info(f"\nZonas geotérmicas descargadas: {positive_count}/{max_positive}")

        # Descargar zonas de control (negativas)
        logger.info(f"\nDescargando zonas SIN potencial geotérmico (máximo {max_negative})...")
        for name, coords in list(self.control_zones.items())[:max_negative]:
            if self.download_image(name, coords, label=0):
                negative_count += 1
            time.sleep(delay) # Evitar sobrecargar la API

        logger.info(f"\nZonas de control descargadas: {negative_count}/{max_negative}")

        # Actualizar metadata
        self.metadata['positive_images'] = positive_count
        self.metadata['negative_images'] = negative_count
        self.metadata['total_images'] = positive_count + negative_count

        return positive_count, negative_count

    def save_metadata(self):
        """Guardar metadata del dataset en formato JSON y CSV."""
        # Guardar JSON
        json_path = self.output_dir / "dataset_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata guardada en: {json_path}")

        # Guardar CSV con detalles de imágenes
        if self.metadata['image_details']:
            df = pd.DataFrame(self.metadata['image_details'])
            csv_path = self.output_dir / "dataset_images.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Lista de imágenes guardada en: {csv_path}")

            # Mostrar estadísticas
            logger.info("\nESTADÍSTICAS DEL DATASET:")
            logger.info(f"Total de imágenes: {len(df)}")
            logger.info(f"Imágenes positivas (geotérmicas): {len(df[df['label']==1])}")
            logger.info(f"Imágenes negativas (control): {len(df[df['label']==0])}")
            logger.info(f"Tamaño total: {df['file_size_mb'].sum():.2f} MB")
            logger.info(f"Balance: {len(df[df['label']==1])/len(df)*100:.1f}% positivas")

    def create_labels_file(self):
        """Crear archivo de etiquetas para entrenamiento."""
        labels_data = []

        for detail in self.metadata['image_details']:
            labels_data.append({
                'filename': detail['filename'],
                'label': detail['label'],
                'zone_name': detail['name'],
                'latitude': detail['coords'][1],
                'longitude': detail['coords'][0]
            })

        df = pd.DataFrame(labels_data)
        labels_path = self.output_dir / "labels.csv"
        df.to_csv(labels_path, index=False, encoding='utf-8')
        logger.info(f"Archivo de etiquetas guardado en: {labels_path}")

        return labels_path


def main():
    """Función principal para ejecutar la descarga del dataset."""
    # Crear directorio de logs
    os.makedirs('logs', exist_ok=True)

    logger.info("\n" + "="*80)
    logger.info("DESCARGA DE DATASET PARA DETECCIÓN DE POTENCIAL GEOTÉRMICO")
    logger.info("Universidad de San Buenaventura - Bogotá")
    logger.info("="*80 + "\n")

    # Importar configuración centralizada (soporta disco externo)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import cfg
    logger.info(f"Fuente de datos: {cfg.source}")
    logger.info(f"Data root: {cfg.data_root}")
    cfg.ensure_dirs()

    # Crear descargador (usa ruta de config.py)
    downloader = GeotermalDatasetDownloader()

    # Configurar cantidad de imágenes a descargar
    MAX_POSITIVE = 111 # Zonas geotérmicas
    MAX_NEGATIVE = 89  # Zonas de control

    logger.info(f"Configuración:")
    logger.info(f" - Imágenes positivas objetivo: {MAX_POSITIVE}")
    logger.info(f" - Imágenes negativas objetivo: {MAX_NEGATIVE}")
    logger.info(f" - Total objetivo: {MAX_POSITIVE + MAX_NEGATIVE}")
    logger.info(f" - Resolución espacial: 90 metros/pixel")
    logger.info(f" - Área por imagen: ~5km de radio\n")

    # Confirmar antes de iniciar
    response = input("¿Desea iniciar la descarga? (s/n): ")
    if response.lower() != 's':
        logger.info("Descarga cancelada por el usuario")
        return

    # Descargar dataset
    start_time = time.time()
    pos_count, neg_count = downloader.download_all_zones(
        max_positive=MAX_POSITIVE,
        max_negative=MAX_NEGATIVE,
        delay=2.0
    )
    end_time = time.time()

    # Guardar metadata y crear archivo de etiquetas
    downloader.save_metadata()
    downloader.create_labels_file()

    # Resumen final
    total_time = end_time - start_time
    logger.info("\n" + "="*80)
    logger.info("DESCARGA COMPLETADA")
    logger.info("="*80)
    logger.info(f"Imágenes positivas descargadas: {pos_count}")
    logger.info(f"Imágenes negativas descargadas: {neg_count}")
    logger.info(f"Total descargado: {pos_count + neg_count}")
    logger.info(f"Tiempo total: {total_time/60:.2f} minutos")
    logger.info(f"Ubicación: data/raw/")
    logger.info("="*80)

    # Verificar balance del dataset
    if pos_count > 0 and neg_count > 0:
        balance = min(pos_count, neg_count) / max(pos_count, neg_count) * 100
        if balance >= 80:
            logger.info(f"Dataset bien balanceado ({balance:.1f}%)")
        else:
            logger.warning(f"Dataset desbalanceado ({balance:.1f}%). Considera descargar más imágenes de la clase minoritaria.")


if __name__ == "__main__":
    main()
