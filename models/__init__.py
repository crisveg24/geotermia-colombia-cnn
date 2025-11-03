"""
Modelos de Deep Learning para Identificación de Potencial Geotérmico
Universidad de San Buenaventura - Bogotá
"""

from .cnn_geotermia import GeotermiaCNN, create_geotermia_model

__all__ = ['GeotermiaCNN', 'create_geotermia_model']
