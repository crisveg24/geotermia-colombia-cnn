"""
En el siguiente enlace del servicio geologico se encuentran 21 puntos geolocalizados con potencial geotermic: https://sgcolombiano.maps.arcgis.com/apps/dashboards/0186f2c2b6e74866b849025b0bf6fd90
se pueden usar para obtenerlos y crear el modleo de deep learning
Se usan las imagenes de AG100: ASTER Global Emissivity Dataset 100-meter V003, proporcionadas por 
google earth engine, se descargan la imagenes y se deben etiquetar, zonas cercanas a volcanes activos, termales
se clasifican como 1, potencial geotermico, mientras que zonas como llanos orientales, desiertos, sabanas se clas
fican como 0no potencial.  
para entrar a google earth se da en abrir en el editor:
https://developers.google.com/earth-engine/datasets/catalog/NASA_ASTER_GED_AG100_003?hl=es-419
Pasos:
1. se instala la libreria pip install earthengine-api geemap, para trabajar con google earth, antes, nos debemos
autenticar en google earth.
2. en la terminal se configura el proyecto: earthengine set_project id_proyecto_estaengogoleearth
3. se instala geemap y notebook: pip install notebook geemap
4. se abre notebook en la terminal se escribe: jupyter notebook
"""
import ee
import geemap

ee.Initialize()

# Carga la imagen
aster = ee.Image("NASA/ASTER_GED/AG100_003")

# Define una región de interés (por ejemplo, cerca de Galeras)
roi = ee.Geometry.Point([-77.36, 1.22]).buffer(5000)

# Visualiza una banda (por ejemplo, "emissivity_band13")
Map = geemap.Map()
Map.centerObject(roi, 10)
Map.addLayer(aster.select('emissivity_band13'), {'min': 0.8, 'max': 1.0}, 'Emisividad B13')
Map
