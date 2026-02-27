# Copyright (c) 2025-2026 Vega Sánchez · Arévalo Rubiano · Espitia Ayala · Rivera Martín
# Universidad de San Buenaventura — Bogotá | github.com/crisveg24/geotermia-colombia-cnn
"""
Script para Descargar Dataset Completo de Imágenes ASTER para Detección de Potencial Geotérmico
========================================================================================

Este script descarga un conjunto balanceado de imágenes térmicas ASTER de la Región Andina
(Colombia, Ecuador, Perú y Chile):
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

        # Definir zonas geotérmicas de la Región Andina (CON potencial - label=1)
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

            # ════════════════════════════════════════════════════════════════
            # ECUADOR — Zona Volcánica / Campos Geotérmicos
            # ════════════════════════════════════════════════════════════════

            # ── Volcán Cotopaxi (estratovolcán activo, 5,897 m) ──
            "EC_Cotopaxi_center": [-78.4360, -0.6770],
            "EC_Cotopaxi_north": [-78.4360, -0.6620],
            "EC_Cotopaxi_south": [-78.4360, -0.6920],
            "EC_Cotopaxi_east": [-78.4160, -0.6770],
            "EC_Cotopaxi_west": [-78.4560, -0.6770],
            "EC_Cotopaxi_NE": [-78.4160, -0.6620],
            "EC_Cotopaxi_SW": [-78.4560, -0.6920],

            # ── Volcán Tungurahua (erupción frecuente) ──
            "EC_Tungurahua_center": [-78.4420, -1.4670],
            "EC_Tungurahua_north": [-78.4420, -1.4520],
            "EC_Tungurahua_south": [-78.4420, -1.4820],
            "EC_Tungurahua_east": [-78.4220, -1.4670],
            "EC_Tungurahua_west": [-78.4620, -1.4670],
            "EC_Tungurahua_NE": [-78.4220, -1.4520],
            "EC_Tungurahua_SW": [-78.4620, -1.4820],

            # ── Volcán Reventador (muy activo, erupción constante) ──
            "EC_Reventador_center": [-77.6500, -0.0770],
            "EC_Reventador_north": [-77.6500, -0.0620],
            "EC_Reventador_south": [-77.6500, -0.0920],
            "EC_Reventador_east": [-77.6300, -0.0770],
            "EC_Reventador_west": [-77.6700, -0.0770],
            "EC_Reventador_NE": [-77.6300, -0.0620],
            "EC_Reventador_SW": [-77.6700, -0.0920],

            # ── Volcán Sangay (actividad permanente) ──
            "EC_Sangay_center": [-78.3400, -2.0300],
            "EC_Sangay_north": [-78.3400, -2.0150],
            "EC_Sangay_south": [-78.3400, -2.0450],
            "EC_Sangay_east": [-78.3200, -2.0300],
            "EC_Sangay_west": [-78.3600, -2.0300],
            "EC_Sangay_NE": [-78.3200, -2.0150],
            "EC_Sangay_SW": [-78.3600, -2.0450],

            # ── Guagua Pichincha (fumarolas activas) ──
            "EC_Pichincha_center": [-78.5980, -0.1710],
            "EC_Pichincha_north": [-78.5980, -0.1560],
            "EC_Pichincha_south": [-78.5980, -0.1860],
            "EC_Pichincha_east": [-78.5780, -0.1710],
            "EC_Pichincha_west": [-78.6180, -0.1710],

            # ── Volcán Cayambe (fumarolas en cumbre) ──
            "EC_Cayambe_center": [-77.9860, 0.0290],
            "EC_Cayambe_north": [-77.9860, 0.0440],
            "EC_Cayambe_south": [-77.9860, 0.0140],
            "EC_Cayambe_east": [-77.9660, 0.0290],
            "EC_Cayambe_west": [-78.0060, 0.0290],

            # ── Volcán Antisana (fumarolas) ──
            "EC_Antisana_center": [-78.1400, -0.4810],
            "EC_Antisana_north": [-78.1400, -0.4660],
            "EC_Antisana_south": [-78.1400, -0.4960],
            "EC_Antisana_east": [-78.1200, -0.4810],
            "EC_Antisana_west": [-78.1600, -0.4810],

            # ── Volcán Chimborazo (6,263 m, más alto de Ecuador) ──
            "EC_Chimborazo_center": [-78.8170, -1.4690],
            "EC_Chimborazo_north": [-78.8170, -1.4540],
            "EC_Chimborazo_south": [-78.8170, -1.4840],
            "EC_Chimborazo_east": [-78.7970, -1.4690],
            "EC_Chimborazo_west": [-78.8370, -1.4690],

            # ── Caldera Chalupas (potencial geotérmico de alta entalpía) ──
            "EC_Chalupas_center": [-78.3400, -0.7800],
            "EC_Chalupas_north": [-78.3400, -0.7650],
            "EC_Chalupas_south": [-78.3400, -0.7950],
            "EC_Chalupas_east": [-78.3200, -0.7800],
            "EC_Chalupas_west": [-78.3600, -0.7800],

            # ── Quilotoa (caldera con lago, actividad termal) ──
            "EC_Quilotoa_center": [-78.9000, -0.8500],
            "EC_Quilotoa_north": [-78.9000, -0.8350],
            "EC_Quilotoa_south": [-78.9000, -0.8650],

            # ── Volcán Illiniza (fumarolas) ──
            "EC_Illiniza_center": [-78.7140, -0.6590],
            "EC_Illiniza_north": [-78.7140, -0.6440],
            "EC_Illiniza_south": [-78.7140, -0.6740],

            # ── Chachimbiro (sistema hidrotermal en exploración) ──
            "EC_Chachimbiro_center": [-78.2900, 0.4600],
            "EC_Chachimbiro_north": [-78.2900, 0.4750],
            "EC_Chachimbiro_south": [-78.2900, 0.4450],

            # ── Cuicocha (caldera con lago y fumarolas) ──
            "EC_Cuicocha_center": [-78.3640, 0.3080],
            "EC_Cuicocha_north": [-78.3640, 0.3230],
            "EC_Cuicocha_south": [-78.3640, 0.2930],

            # ── Pululagua (caldera con fumarolas) ──
            "EC_Pululagua_center": [-78.4630, 0.0380],
            "EC_Pululagua_north": [-78.4630, 0.0530],
            "EC_Pululagua_south": [-78.4630, 0.0230],

            # ── Atacazo (sistema volcánico con termalismo) ──
            "EC_Atacazo_center": [-78.6170, -0.3530],
            "EC_Atacazo_north": [-78.6170, -0.3380],
            "EC_Atacazo_south": [-78.6170, -0.3680],

            # ── Volcán Cotacachi ──
            "EC_Cotacachi_center": [-78.3490, 0.3610],
            "EC_Cotacachi_north": [-78.3490, 0.3760],
            "EC_Cotacachi_south": [-78.3490, 0.3460],

            # ── Volcán Mojanda (complejo de calderas) ──
            "EC_Mojanda_center": [-78.2700, 0.1300],
            "EC_Mojanda_north": [-78.2700, 0.1450],
            "EC_Mojanda_south": [-78.2700, 0.1150],

            # ── Volcán Imbabura ──
            "EC_Imbabura_center": [-78.1800, 0.2600],
            "EC_Imbabura_north": [-78.1800, 0.2750],
            "EC_Imbabura_south": [-78.1800, 0.2450],

            # ── Zonas termales Ecuador ──
            "EC_Banos_Agua_Santa": [-78.4200, -1.4000],
            "EC_Papallacta_thermal": [-78.1400, -0.3700],
            "EC_Oyacachi_thermal": [-78.0700, -0.2100],
            "EC_Nangulvi_thermal": [-78.5000, 0.3300],
            "EC_Guapan_thermal": [-78.9000, -2.7300],
            "EC_San_Vicente_thermal": [-80.2300, -2.2200],

            # ════════════════════════════════════════════════════════════════
            # PERÚ — Zona Volcánica Central / Campos Geotérmicos
            # ════════════════════════════════════════════════════════════════

            # ── Volcán Misti (estratovolcán, símbolo de Arequipa) ──
            "PE_Misti_center": [-71.4090, -16.2940],
            "PE_Misti_north": [-71.4090, -16.2790],
            "PE_Misti_south": [-71.4090, -16.3090],
            "PE_Misti_east": [-71.3890, -16.2940],
            "PE_Misti_west": [-71.4290, -16.2940],
            "PE_Misti_NE": [-71.3890, -16.2790],
            "PE_Misti_SW": [-71.4290, -16.3090],

            # ── Volcán Ubinas (más activo del Perú) ──
            "PE_Ubinas_center": [-70.9030, -16.3550],
            "PE_Ubinas_north": [-70.9030, -16.3400],
            "PE_Ubinas_south": [-70.9030, -16.3700],
            "PE_Ubinas_east": [-70.8830, -16.3550],
            "PE_Ubinas_west": [-70.9230, -16.3550],
            "PE_Ubinas_NE": [-70.8830, -16.3400],
            "PE_Ubinas_SW": [-70.9230, -16.3700],

            # ── Volcán Sabancaya (erupción intermitente desde 2017) ──
            "PE_Sabancaya_center": [-71.8500, -15.7800],
            "PE_Sabancaya_north": [-71.8500, -15.7650],
            "PE_Sabancaya_south": [-71.8500, -15.7950],
            "PE_Sabancaya_east": [-71.8300, -15.7800],
            "PE_Sabancaya_west": [-71.8700, -15.7800],
            "PE_Sabancaya_NE": [-71.8300, -15.7650],
            "PE_Sabancaya_SW": [-71.8700, -15.7950],

            # ── Volcán Chachani (fumarolas de cumbre) ──
            "PE_Chachani_center": [-71.5300, -16.1910],
            "PE_Chachani_north": [-71.5300, -16.1760],
            "PE_Chachani_south": [-71.5300, -16.2060],
            "PE_Chachani_east": [-71.5100, -16.1910],
            "PE_Chachani_west": [-71.5500, -16.1910],

            # ── Volcán Huaynaputina (gran erupción de 1600) ──
            "PE_Huaynaputina_center": [-70.8500, -16.6080],
            "PE_Huaynaputina_north": [-70.8500, -16.5930],
            "PE_Huaynaputina_south": [-70.8500, -16.6230],
            "PE_Huaynaputina_east": [-70.8300, -16.6080],
            "PE_Huaynaputina_west": [-70.8700, -16.6080],

            # ── Volcán Tutupaca (fumarolas activas) ──
            "PE_Tutupaca_center": [-70.3580, -17.0250],
            "PE_Tutupaca_north": [-70.3580, -17.0100],
            "PE_Tutupaca_south": [-70.3580, -17.0400],
            "PE_Tutupaca_east": [-70.3380, -17.0250],
            "PE_Tutupaca_west": [-70.3780, -17.0250],

            # ── Volcán Ticsani (fumarolas y termalismo) ──
            "PE_Ticsani_center": [-70.5950, -16.7550],
            "PE_Ticsani_north": [-70.5950, -16.7400],
            "PE_Ticsani_south": [-70.5950, -16.7700],
            "PE_Ticsani_east": [-70.5750, -16.7550],
            "PE_Ticsani_west": [-70.6150, -16.7550],

            # ── Volcán Yucamane (actividad fumarólica) ──
            "PE_Yucamane_center": [-70.2000, -17.1800],
            "PE_Yucamane_north": [-70.2000, -17.1650],
            "PE_Yucamane_south": [-70.2000, -17.1950],
            "PE_Yucamane_east": [-70.1800, -17.1800],
            "PE_Yucamane_west": [-70.2200, -17.1800],

            # ── Volcán Coropuna (nevado con termalismo) ──
            "PE_Coropuna_center": [-72.6500, -15.5200],
            "PE_Coropuna_north": [-72.6500, -15.5050],
            "PE_Coropuna_south": [-72.6500, -15.5350],
            "PE_Coropuna_east": [-72.6300, -15.5200],
            "PE_Coropuna_west": [-72.6700, -15.5200],

            # ── Volcán Casiri (fumarolas) ──
            "PE_Casiri_center": [-69.8130, -17.4700],
            "PE_Casiri_north": [-69.8130, -17.4550],
            "PE_Casiri_south": [-69.8130, -17.4850],

            # ── Volcán Ampato (fumarolas) ──
            "PE_Ampato_center": [-71.8830, -15.8170],
            "PE_Ampato_north": [-71.8830, -15.8020],
            "PE_Ampato_south": [-71.8830, -15.8320],

            # ── Volcán Sara Sara (fumarolas y termalismo) ──
            "PE_Sara_Sara_center": [-73.4500, -15.3300],
            "PE_Sara_Sara_north": [-73.4500, -15.3150],
            "PE_Sara_Sara_south": [-73.4500, -15.3450],

            # ── Campo geotérmico Calientes (Tacna, alta entalpía) ──
            "PE_Calientes_center": [-69.9000, -17.5500],
            "PE_Calientes_north": [-69.9000, -17.5350],
            "PE_Calientes_south": [-69.9000, -17.5650],

            # ── Campo geotérmico Borateras (Tacna) ──
            "PE_Borateras_center": [-69.8300, -17.4800],
            "PE_Borateras_north": [-69.8300, -17.4650],
            "PE_Borateras_south": [-69.8300, -17.4950],

            # ── Zonas termales Perú ──
            "PE_Salinas_Chivay_thermal": [-71.6000, -15.6300],
            "PE_Banos_del_Inca_thermal": [-78.4700, -7.1600],
            "PE_Churin_thermal": [-76.8700, -10.8200],
            "PE_Yura_thermal": [-71.6800, -16.2500],
            "PE_Jesus_thermal": [-71.4600, -16.4800],
            "PE_Calientes_Candarave": [-70.2600, -17.2900],
            "PE_La_Calera_Chivay": [-71.5900, -15.6400],
            "PE_Rio_Jesus_thermal": [-70.2000, -16.3500],

            # ════════════════════════════════════════════════════════════════
            # CHILE — Zona Volcánica Central y Sur / Campos Geotérmicos
            # ════════════════════════════════════════════════════════════════

            # ── El Tatio (mayor campo de géiseres del hemisferio sur) ──
            "CL_El_Tatio_center": [-68.0100, -22.3320],
            "CL_El_Tatio_north": [-68.0100, -22.3170],
            "CL_El_Tatio_south": [-68.0100, -22.3470],
            "CL_El_Tatio_east": [-67.9900, -22.3320],
            "CL_El_Tatio_west": [-68.0300, -22.3320],
            "CL_El_Tatio_NE": [-67.9900, -22.3170],
            "CL_El_Tatio_SW": [-68.0300, -22.3470],

            # ── Cerro Pabellón (planta geotérmica 48 MW, única en Sudamérica) ──
            "CL_Cerro_Pabellon_center": [-68.1500, -21.8500],
            "CL_Cerro_Pabellon_north": [-68.1500, -21.8350],
            "CL_Cerro_Pabellon_south": [-68.1500, -21.8650],
            "CL_Cerro_Pabellon_east": [-68.1300, -21.8500],
            "CL_Cerro_Pabellon_west": [-68.1700, -21.8500],
            "CL_Cerro_Pabellon_NE": [-68.1300, -21.8350],
            "CL_Cerro_Pabellon_SW": [-68.1700, -21.8650],

            # ── Puchuldiza-Tuja (géiseres, fumarolas, 33 MW térmicos) ──
            "CL_Puchuldiza_center": [-69.0000, -19.4170],
            "CL_Puchuldiza_north": [-69.0000, -19.4020],
            "CL_Puchuldiza_south": [-69.0000, -19.4320],
            "CL_Puchuldiza_east": [-68.9800, -19.4170],
            "CL_Puchuldiza_west": [-69.0200, -19.4170],
            "CL_Puchuldiza_NE": [-68.9800, -19.4020],
            "CL_Puchuldiza_SW": [-69.0200, -19.4320],

            # ── Nevados de Chillán (volcán activo, campo geotérmico) ──
            "CL_Chillan_center": [-71.3700, -36.8600],
            "CL_Chillan_north": [-71.3700, -36.8450],
            "CL_Chillan_south": [-71.3700, -36.8750],
            "CL_Chillan_east": [-71.3500, -36.8600],
            "CL_Chillan_west": [-71.3900, -36.8600],
            "CL_Chillan_NE": [-71.3500, -36.8450],
            "CL_Chillan_SW": [-71.3900, -36.8750],

            # ── Volcán Láscar (más activo del norte de Chile) ──
            "CL_Lascar_center": [-67.7300, -23.3700],
            "CL_Lascar_north": [-67.7300, -23.3550],
            "CL_Lascar_south": [-67.7300, -23.3850],
            "CL_Lascar_east": [-67.7100, -23.3700],
            "CL_Lascar_west": [-67.7500, -23.3700],

            # ── Volcán Guallatiri (fumarolas intensas) ──
            "CL_Guallatiri_center": [-69.0900, -18.4200],
            "CL_Guallatiri_north": [-69.0900, -18.4050],
            "CL_Guallatiri_south": [-69.0900, -18.4350],
            "CL_Guallatiri_east": [-69.0700, -18.4200],
            "CL_Guallatiri_west": [-69.1100, -18.4200],

            # ── Calabozos (caldera con potencial de alta entalpía) ──
            "CL_Calabozos_center": [-70.5000, -35.5500],
            "CL_Calabozos_north": [-70.5000, -35.5350],
            "CL_Calabozos_south": [-70.5000, -35.5650],
            "CL_Calabozos_east": [-70.4800, -35.5500],
            "CL_Calabozos_west": [-70.5200, -35.5500],

            # ── Tolhuaca (campo geotérmico en exploración) ──
            "CL_Tolhuaca_center": [-71.6500, -38.3200],
            "CL_Tolhuaca_north": [-71.6500, -38.3050],
            "CL_Tolhuaca_south": [-71.6500, -38.3350],
            "CL_Tolhuaca_east": [-71.6300, -38.3200],
            "CL_Tolhuaca_west": [-71.6700, -38.3200],

            # ── Cordón Caulle (fisura volcánica, potencial geotérmico) ──
            "CL_Cordon_Caulle_center": [-71.7500, -40.5200],
            "CL_Cordon_Caulle_north": [-71.7500, -40.5050],
            "CL_Cordon_Caulle_south": [-71.7500, -40.5350],
            "CL_Cordon_Caulle_east": [-71.7300, -40.5200],
            "CL_Cordon_Caulle_west": [-71.7700, -40.5200],

            # ── Volcán Villarrica (volcán muy activo, lago de lava) ──
            "CL_Villarrica_center": [-71.9300, -39.4200],
            "CL_Villarrica_north": [-71.9300, -39.4050],
            "CL_Villarrica_south": [-71.9300, -39.4350],
            "CL_Villarrica_east": [-71.9100, -39.4200],
            "CL_Villarrica_west": [-71.9500, -39.4200],

            # ── Volcán Putana (fumarolas permanentes) ──
            "CL_Putana_center": [-67.8500, -22.5700],
            "CL_Putana_north": [-67.8500, -22.5550],
            "CL_Putana_south": [-67.8500, -22.5850],

            # ── Volcán Irruputuncu (fumarolas activas) ──
            "CL_Irruputuncu_center": [-68.5500, -20.7300],
            "CL_Irruputuncu_north": [-68.5500, -20.7150],
            "CL_Irruputuncu_south": [-68.5500, -20.7450],

            # ── Volcán Isluga (fumarolas) ──
            "CL_Isluga_center": [-68.8300, -19.1500],
            "CL_Isluga_north": [-68.8300, -19.1350],
            "CL_Isluga_south": [-68.8300, -19.1650],

            # ── Volcán Ollagüe (fumarolas activas) ──
            "CL_Ollague_center": [-68.1800, -21.3000],
            "CL_Ollague_north": [-68.1800, -21.2850],
            "CL_Ollague_south": [-68.1800, -21.3150],

            # ── Volcán San Pedro ──
            "CL_San_Pedro_center": [-68.4000, -21.8800],
            "CL_San_Pedro_north": [-68.4000, -21.8650],
            "CL_San_Pedro_south": [-68.4000, -21.8950],

            # ── Volcán Tacora (fumarolas) ──
            "CL_Tacora_center": [-69.7700, -17.7200],
            "CL_Tacora_north": [-69.7700, -17.7050],
            "CL_Tacora_south": [-69.7700, -17.7350],

            # ── Volcán Licancabur ──
            "CL_Licancabur_center": [-67.8830, -22.8330],
            "CL_Licancabur_north": [-67.8830, -22.8180],
            "CL_Licancabur_south": [-67.8830, -22.8480],

            # ── Surire (campo geotérmico en salar) ──
            "CL_Surire_center": [-69.0500, -18.8500],
            "CL_Surire_north": [-69.0500, -18.8350],
            "CL_Surire_south": [-69.0500, -18.8650],

            # ── Volcán Parinacota ──
            "CL_Parinacota_center": [-69.1400, -18.1700],
            "CL_Parinacota_north": [-69.1400, -18.1550],
            "CL_Parinacota_south": [-69.1400, -18.1850],

            # ── Volcán Llaima (activo) ──
            "CL_Llaima_center": [-71.7300, -38.6900],
            "CL_Llaima_north": [-71.7300, -38.6750],
            "CL_Llaima_south": [-71.7300, -38.7050],

            # ── Volcán Copahue (fumarolas, aguas termales) ──
            "CL_Copahue_center": [-71.1700, -37.8500],
            "CL_Copahue_north": [-71.1700, -37.8350],
            "CL_Copahue_south": [-71.1700, -37.8650],

            # ── Volcán Antuco ──
            "CL_Antuco_center": [-71.3490, -37.4060],
            "CL_Antuco_north": [-71.3490, -37.3910],
            "CL_Antuco_south": [-71.3490, -37.4210],

            # ── Volcán Callaqui (fumarolas activas) ──
            "CL_Callaqui_center": [-71.4490, -37.9230],
            "CL_Callaqui_north": [-71.4490, -37.9080],
            "CL_Callaqui_south": [-71.4490, -37.9380],

            # ── Zonas termales Chile ──
            "CL_Polloquere_thermal": [-69.0600, -18.8550],
            "CL_Jurasi_thermal": [-69.5100, -18.1800],
            "CL_Colina_thermal": [-70.3000, -33.3000],
            "CL_Huife_thermal": [-71.7500, -39.3500],
            "CL_Geometricas_thermal": [-71.9000, -39.6000],
            "CL_Flaco_thermal": [-70.4200, -34.9700],
            "CL_Menetue_thermal": [-71.8500, -39.3300],
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

            # ════════════════════════════════════════════════════════════════
            # ECUADOR — Zonas de control (sin vulcanismo)
            # ════════════════════════════════════════════════════════════════
            "EC_Guayaquil_control": [-79.8970, -2.1700],
            "EC_Machala_control": [-79.9600, -3.2600],
            "EC_Esmeraldas_control": [-79.6500, 0.9600],
            "EC_Santo_Domingo_control": [-79.1700, -0.2500],
            "EC_Quevedo_control": [-79.4600, -1.0200],
            "EC_Manta_control": [-80.7330, -0.9500],
            "EC_Portoviejo_control": [-80.4500, -1.0500],
            "EC_Lago_Agrio_control": [-76.8780, 0.0840],
            "EC_Puyo_control": [-77.9900, -1.4900],
            "EC_Tena_control": [-77.8100, -1.0000],
            "EC_Zamora_control": [-78.9500, -4.0700],
            "EC_Loja_sur_control": [-79.2200, -4.0500],
            "EC_Santa_Elena_control": [-80.8600, -2.2300],
            "EC_Duran_control": [-79.8300, -2.1700],
            "EC_Milagro_control": [-79.5900, -2.1300],
            "EC_Babahoyo_control": [-79.5300, -1.8000],
            "EC_Vinces_control": [-79.7500, -1.5600],
            "EC_Cuenca_control": [-79.0200, -2.9200],
            "EC_Nueva_Loja_control": [-76.8800, 0.0900],
            "EC_Daule_control": [-79.9800, -1.8600],

            # ════════════════════════════════════════════════════════════════
            # PERÚ — Zonas de control (sin vulcanismo)
            # ════════════════════════════════════════════════════════════════
            "PE_Lima_control": [-77.0280, -12.0460],
            "PE_Piura_control": [-80.6300, -5.1900],
            "PE_Chiclayo_control": [-79.8400, -6.7700],
            "PE_Trujillo_control": [-79.0300, -8.1100],
            "PE_Ica_control": [-75.7300, -14.0600],
            "PE_Nasca_control": [-75.1200, -14.8300],
            "PE_Chimbote_control": [-78.5300, -9.0700],
            "PE_Huancayo_control": [-75.2100, -12.0700],
            "PE_Cusco_control": [-71.9700, -13.5200],
            "PE_Puno_control": [-70.0200, -15.8400],
            "PE_Juliaca_control": [-70.1300, -15.5000],
            "PE_Iquitos_control": [-73.2500, -3.7500],
            "PE_Pucallpa_control": [-74.5300, -8.3800],
            "PE_Tarapoto_control": [-76.3700, -6.4900],
            "PE_Tumbes_control": [-80.4500, -3.5700],
            "PE_Ayacucho_control": [-74.2200, -13.1600],
            "PE_Huaraz_control": [-77.5280, -9.5300],
            "PE_Cajamarca_control": [-78.5200, -7.1600],
            "PE_Tacna_control": [-70.2500, -18.0100],
            "PE_Arequipa_control": [-71.5400, -16.4090],

            # ════════════════════════════════════════════════════════════════
            # CHILE — Zonas de control (sin vulcanismo)
            # ════════════════════════════════════════════════════════════════
            "CL_Santiago_control": [-70.6500, -33.4480],
            "CL_Valparaiso_control": [-71.6200, -33.0470],
            "CL_Concepcion_control": [-73.0500, -36.8300],
            "CL_La_Serena_control": [-71.2500, -29.9000],
            "CL_Copiapo_control": [-70.3300, -27.3700],
            "CL_Antofagasta_control": [-70.4000, -23.6500],
            "CL_Arica_control": [-70.3110, -18.4750],
            "CL_Temuco_control": [-72.6400, -38.7400],
            "CL_Osorno_control": [-73.1500, -40.5730],
            "CL_Valdivia_control": [-73.2400, -39.8100],
            "CL_Rancagua_control": [-70.7400, -34.1700],
            "CL_Talca_control": [-71.6500, -35.4300],
            "CL_Chillan_ciudad_control": [-72.1000, -36.6000],
            "CL_Punta_Arenas_control": [-70.9170, -53.1500],
            "CL_Puerto_Montt_control": [-72.9400, -41.4700],
            "CL_Iquique_control": [-70.1300, -20.2100],
            "CL_Calama_control": [-68.9200, -22.4500],
            "CL_San_Fernando_control": [-71.0000, -34.5800],
            "CL_Curico_control": [-71.2400, -34.9800],
            "CL_Los_Angeles_control": [-72.3500, -37.4700],
        }

        # ═══════════════════════════════════════════════════════════════
        # EXPANSIÓN AUTOMÁTICA: zonas adicionales → ~2000 imágenes
        # Se generan 9 tiles por cada zona base (center + 8 direcciones)
        # Justificación: el dataset colombiano original (~200 imgs) es
        # insuficiente para entrenar una CNN robusta, por lo que se agregan
        # zonas geotérmicas y de control de la región andina.
        # ═══════════════════════════════════════════════════════════════

        _extra_positive_base = {
            # ── COLOMBIA — Volcanes y zonas termales adicionales ──
            "CO_Sotara": [-76.5900, 2.1100],
            "CO_Dona_Juana": [-76.9300, 1.4900],
            "CO_Cerro_Negro_Mayasquer": [-77.1600, 1.0800],
            "CO_Cerro_Bravo": [-75.3000, 5.0900],
            "CO_Santa_Isabel": [-75.3700, 4.8200],
            "CO_Tolima": [-75.3300, 4.6600],
            "CO_Huila": [-75.5800, 2.9300],
            "CO_Chiles": [-77.9400, 0.8200],
            "CO_Cerro_Machin": [-75.3900, 4.4850],
            "CO_Romeral": [-75.3640, 4.9370],
            "CO_Cisne": [-75.3370, 4.8370],
            "CO_Santa_Rosa_Termas": [-75.6200, 4.8700],
            "CO_San_Vicente_Termas": [-73.4100, 6.0800],
            "CO_Paipa_Termas": [-73.1100, 5.7800],
            "CO_Tabio_Termas": [-74.0900, 4.9200],
            "CO_Coconucos": [-76.4660, 2.2700],
            "CO_Dona_Juana_E": [-76.9000, 1.5200],
            "CO_Azufral_Mayasquer": [-77.7300, 0.9800],

            # ── ECUADOR — Volcanes y zonas termales adicionales ──
            "EC_Pichincha_V": [-78.5980, -0.1720],
            "EC_Atacazo": [-78.6170, -0.3530],
            "EC_Sumaco": [-77.6300, -0.5400],
            "EC_Sangay": [-78.3410, -2.0050],
            "EC_El_Altar": [-78.4200, -1.6800],
            "EC_Cayambe": [-77.9860, 0.0290],
            "EC_Illiniza": [-78.7140, -0.6590],
            "EC_Corazon": [-78.6600, -0.5400],
            "EC_Imbabura_V": [-78.1830, 0.2580],
            "EC_Mojanda": [-78.2700, 0.1300],
            "EC_Carihuairazo": [-78.7500, -1.4000],
            "EC_Guagua_Pichincha": [-78.6000, -0.1710],
            "EC_Cerro_Hermoso": [-77.5000, -0.4800],
            "EC_Sarahurco": [-78.2300, -0.3300],
            "EC_Cuicocha": [-78.3640, 0.3080],
            "EC_Pululahua": [-78.4640, -0.0380],
            "EC_Papallacta_Termas": [-78.1400, -0.3730],
            "EC_Banos_Cuenca_Termas": [-78.9200, -2.9300],

            # ── PERÚ — Volcanes y zonas termales adicionales ──
            "PE_Ampato": [-71.8900, -15.8200],
            "PE_Huaynaputina": [-70.8500, -16.6080],
            "PE_Ticsani": [-70.5950, -16.7550],
            "PE_Tutupaca": [-70.3600, -17.0250],
            "PE_Yucamane": [-70.2020, -17.1800],
            "PE_Coropuna": [-72.6530, -15.5200],
            "PE_Sara_Sara": [-73.4500, -15.3260],
            "PE_Solimana": [-72.8900, -15.4100],
            "PE_Chachani": [-71.5320, -16.1910],
            "PE_Pichu_Pichu": [-71.2300, -16.4400],
            "PE_Calientes_Tacna": [-70.0800, -17.7100],
            "PE_Churin_Termas": [-76.8800, -10.8200],
            "PE_La_Calera_Termas": [-71.5900, -15.6600],
            "PE_Chivay_Termas": [-71.6000, -15.6400],
            "PE_Jesus_Termas": [-71.4200, -15.9700],
            "PE_Cayani": [-70.3000, -16.8500],
            "PE_Salinas_Laguna": [-71.1330, -16.2560],
            "PE_Nicholson": [-70.7000, -16.7500],

            # ── CHILE — Volcanes y zonas termales adicionales ──
            "CL_Irruputuncu": [-68.5600, -20.7400],
            "CL_Olague": [-68.1800, -21.3000],
            "CL_San_Pedro_V": [-68.3960, -21.8800],
            "CL_Putana": [-67.8560, -22.5680],
            "CL_Guallatiri": [-69.0920, -18.4230],
            "CL_Parinacota": [-69.1420, -18.1700],
            "CL_Isluga": [-68.8310, -19.1500],
            "CL_Tacora": [-69.7710, -17.7200],
            "CL_Cerro_Pabellon": [-68.1500, -22.3100],
            "CL_Apacheta": [-68.1800, -22.2200],
            "CL_Pampa_Lirima": [-69.3300, -19.8200],
            "CL_Chaiten": [-72.6500, -42.8300],
            "CL_Melimoyu": [-72.8600, -44.0700],
            "CL_Corcovado": [-72.8000, -43.1900],
            "CL_Lonquimay": [-71.5850, -38.3790],
            "CL_Sollipulli": [-71.5200, -38.9700],
            "CL_Tolhuaca": [-71.6450, -38.3100],
            "CL_Mocho_Choshuenco": [-72.0270, -39.9270],
        }

        _extra_negative_base = {
            # ── COLOMBIA — Zonas sin vulcanismo adicionales ──
            "CO_Leticia_Ctrl": [-69.9400, -4.2150],
            "CO_Puerto_Inirida_Ctrl": [-67.9200, 3.8650],
            "CO_Mitu_Ctrl": [-70.2400, 1.2530],
            "CO_San_Jose_Guaviare_Ctrl": [-72.6400, 2.5700],
            "CO_Florencia_Ctrl": [-75.6100, 1.6100],
            "CO_Mocoa_Ctrl": [-76.6500, 1.1500],
            "CO_Sincelejo_Ctrl": [-75.3900, 9.3000],
            "CO_Monteria_Ctrl": [-75.8900, 8.7500],
            "CO_Turbo_Ctrl": [-76.7300, 8.0900],
            "CO_Quibdo_Ctrl": [-76.6500, 5.6900],
            "CO_Riohacha_Ctrl": [-72.9070, 11.5440],
            "CO_Valledupar_Ctrl": [-73.2500, 10.4730],
            "CO_Santa_Marta_Ctrl": [-74.2000, 11.2400],
            "CO_Barranquilla_Ctrl": [-74.7960, 10.9630],
            "CO_Cartagena_Ctrl": [-75.5140, 10.3910],
            "CO_Bucaramanga_Ctrl": [-73.1260, 7.1190],
            "CO_Cucuta_Ctrl": [-72.5070, 7.8940],
            "CO_Tunja_Ctrl": [-73.3610, 5.5350],
            "CO_Pasto_Ciudad_Ctrl": [-77.2790, 1.2140],
            "CO_Popayan_Ciudad_Ctrl": [-76.6060, 2.4410],
            "CO_Neiva_Ctrl": [-75.2920, 2.9270],
            "CO_Ibague_Ctrl": [-75.2320, 4.4380],
            "CO_Armenia_Ctrl": [-75.6810, 4.5340],
            "CO_Pereira_Ctrl": [-75.6960, 4.8130],
            "CO_Manizales_Ctrl": [-75.5170, 5.0690],

            # ── ECUADOR — Zonas sin vulcanismo adicionales ──
            "EC_Guayaquil_Ctrl": [-79.8890, -2.1870],
            "EC_Quito_Ciudad_Ctrl": [-78.5250, -0.2300],
            "EC_Cuenca_East_Ctrl": [-78.9500, -2.9000],
            "EC_Machala_Ctrl": [-79.9600, -3.2600],
            "EC_Esmeraldas_Ctrl": [-79.6500, 0.9600],
            "EC_Manta_Ctrl": [-80.7330, -0.9500],
            "EC_Ambato_Ctrl": [-78.6300, -1.2400],
            "EC_Riobamba_Ctrl": [-78.6500, -1.6700],
            "EC_Ibarra_Ctrl": [-78.1200, 0.3500],
            "EC_Tulcan_Ctrl": [-77.7200, 0.8100],
            "EC_Quevedo_Ctrl": [-79.4700, -1.0200],
            "EC_Santo_Domingo_Ctrl": [-79.1740, -0.2530],
            "EC_Coca_Ctrl": [-76.9700, -0.4700],
            "EC_Macas_Ctrl": [-78.1200, -2.3100],
            "EC_Guaranda_Ctrl": [-78.9900, -1.5900],
            "EC_Azogues_Ctrl": [-78.8500, -2.7400],
            "EC_Salinas_Ciudad_Ctrl": [-80.9500, -2.2100],
            "EC_Playas_Ctrl": [-80.3900, -2.6300],
            "EC_Jipijapa_Ctrl": [-80.5800, -1.3500],
            "EC_Chone_Ctrl": [-80.0900, -0.6900],
            "EC_Pedernales_Ctrl": [-80.0500, 0.0700],
            "EC_Muisne_Ctrl": [-80.0200, 0.6100],

            # ── PERÚ — Zonas sin vulcanismo adicionales ──
            "PE_Lima_North_Ctrl": [-77.0800, -11.9500],
            "PE_Sullana_Ctrl": [-80.6900, -4.9000],
            "PE_Paita_Ctrl": [-81.1100, -5.0900],
            "PE_Lambayeque_Ctrl": [-79.9100, -6.7100],
            "PE_Chepen_Ctrl": [-79.4300, -7.2300],
            "PE_Huacho_Ctrl": [-77.6100, -11.1100],
            "PE_Chincha_Ctrl": [-76.1300, -13.4600],
            "PE_Pisco_Ctrl": [-76.2200, -13.7100],
            "PE_Camana_Ctrl": [-72.7100, -16.6200],
            "PE_Mollendo_Ctrl": [-72.0100, -17.0200],
            "PE_Ilo_Ctrl": [-71.3400, -17.6400],
            "PE_Moquegua_Ctrl": [-70.9300, -17.1900],
            "PE_Abancay_Ctrl": [-72.8800, -13.6400],
            "PE_Huanuco_Ctrl": [-76.2300, -9.9300],
            "PE_Cerro_Pasco_Ctrl": [-76.2100, -10.6900],
            "PE_Tingo_Maria_Ctrl": [-76.0000, -9.2900],
            "PE_Moyobamba_Ctrl": [-76.9700, -6.0400],
            "PE_Chachapoyas_Ctrl": [-77.8700, -6.2300],
            "PE_Bagua_Ctrl": [-78.5300, -5.6400],
            "PE_Jaen_Ctrl": [-78.8100, -5.7100],
            "PE_Atalaya_Ctrl": [-73.7500, -10.7300],
            "PE_Nauta_Ctrl": [-73.5800, -4.5100],
            "PE_Puerto_Maldonado_Ctrl": [-69.1900, -12.6000],
            "PE_Huancavelica_Ctrl": [-75.0200, -12.7800],
            "PE_Oxapampa_Ctrl": [-75.4000, -10.5800],

            # ── CHILE — Zonas sin vulcanismo adicionales ──
            "CL_Coquimbo_Ctrl": [-71.3400, -29.9500],
            "CL_Ovalle_Ctrl": [-71.2000, -30.6000],
            "CL_San_Antonio_Ctrl": [-71.6100, -33.5900],
            "CL_Quillota_Ctrl": [-71.2500, -32.8800],
            "CL_San_Felipe_Ctrl": [-70.7300, -32.7500],
            "CL_Linares_Ctrl": [-71.6000, -35.8500],
            "CL_Cauquenes_Ctrl": [-72.3200, -35.9700],
            "CL_Constitucion_Ctrl": [-72.4100, -35.3300],
            "CL_Angol_Ctrl": [-72.7100, -37.7900],
            "CL_Victoria_Ctrl": [-72.3300, -38.2300],
            "CL_Villarrica_Ciudad_Ctrl": [-72.2200, -39.2700],
            "CL_Ancud_Ctrl": [-73.8300, -41.8700],
            "CL_Castro_Ctrl": [-73.7600, -42.4800],
            "CL_Coyhaique_Ctrl": [-72.0700, -45.5700],
            "CL_Natales_Ctrl": [-72.5100, -51.7200],
            "CL_Porvenir_Ctrl": [-70.3700, -53.2900],
            "CL_Salamanca_Ctrl": [-70.9700, -31.7700],
            "CL_Illapel_Ctrl": [-71.1700, -31.6300],
            "CL_Tocopilla_Ctrl": [-70.2000, -22.0900],
            "CL_Mejillones_Ctrl": [-70.4500, -23.1000],
            "CL_Taltal_Ctrl": [-70.4800, -25.4000],
            "CL_Chanaral_Ctrl": [-70.6200, -26.3500],
            "CL_Huasco_Ctrl": [-71.2100, -28.4700],
            "CL_Lebu_Ctrl": [-73.6500, -37.6100],
            "CL_Coronel_Ctrl": [-73.1500, -37.0300],
        }

        # Expandir zonas base con grilla de 9 puntos y agregar
        self.geothermal_zones.update(
            self._generate_grid(_extra_positive_base)
        )
        self.control_zones.update(
            self._generate_grid(_extra_negative_base)
        )

        logger.info(
            f"Zonas cargadas: {len(self.geothermal_zones)} positivas, "
            f"{len(self.control_zones)} negativas "
            f"(total: {len(self.geothermal_zones) + len(self.control_zones)})"
        )

        # Metadata del dataset
        self.metadata = {
            'dataset_name': 'Andean_Region_Geothermal_ASTER',
            'download_date': datetime.now().isoformat(),
            'total_images': 0,
            'positive_images': 0,
            'negative_images': 0,
            'image_details': []
        }

    @staticmethod
    def _generate_grid(base_zones: dict, delta: float = 0.013) -> dict:
        """
        Generar grilla de 9 puntos alrededor de cada zona base.

        Cada zona base produce 9 tiles: center, N, S, E, W, NE, NW, SE, SW
        con desplazamientos de ~1.4 km (0.013°). Esto garantiza tiles
        distintos con mínima superposición en imágenes ASTER (buffer 5 km).

        Args:
            base_zones: Dict {nombre: [lon, lat]} con coordenadas base.
            delta: Desplazamiento en grados (~1.4 km por 0.013°).

        Returns:
            Dict expandido con 9 variantes por cada zona base.
        """
        offsets = {
            'center': (0, 0),
            'N': (0, delta),
            'S': (0, -delta),
            'E': (delta, 0),
            'W': (-delta, 0),
            'NE': (delta, delta),
            'NW': (-delta, delta),
            'SE': (delta, -delta),
            'SW': (-delta, -delta),
        }
        expanded = {}
        for name, coords in base_zones.items():
            for suffix, (dx, dy) in offsets.items():
                expanded[f"{name}_{suffix}"] = [
                    round(coords[0] + dx, 4),
                    round(coords[1] + dy, 4)
                ]
        return expanded

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
    logger.info("DESCARGA DE DATASET ANDINO PARA DETECCIÓN DE POTENCIAL GEOTÉRMICO")
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
    # Las zonas se calculan dinámicamente: zonas manuales + expansión por grilla
    MAX_POSITIVE = len(downloader.geothermal_zones)
    MAX_NEGATIVE = len(downloader.control_zones)

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
