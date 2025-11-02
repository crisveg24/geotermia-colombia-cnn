# 🌋 Proyecto Geotermia - Análisis de Potencial Geotérmico con Google Earth Engine

## 📋 Descripción

Proyecto de análisis geotérmico utilizando imágenes satelitales ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) para identificar zonas con potencial geotérmico en Colombia mediante técnicas de Deep Learning y procesamiento de imágenes geoespaciales.

Este proyecto utiliza Google Earth Engine para descargar y procesar datos de emisividad térmica de zonas volcánicas y geotérmicas de Colombia, preparando datasets etiquetados para entrenar modelos de Machine Learning.

## 🎯 Objetivo

Desarrollar un sistema automatizado para:
- Descargar imágenes satelitales de zonas geotérmicas colombianas
- Procesar y etiquetar datos de emisividad térmica
- Crear datasets para modelos de clasificación de potencial geotérmico
- Identificar áreas con características geotérmicas favorables

## 🗺️ Zonas de Estudio

El proyecto se enfoca en tres zonas geotérmicas principales de Colombia:

1. **Nevado del Ruiz** (-75.3222, 4.8951)
   - Volcán activo en el Eje Cafetero
   - Alta actividad geotérmica

2. **Volcán Purácé** (-76.4036, 2.3206)
   - Volcán activo en el Cauca
   - Conocido por sus aguas termales

3. **Paipa-Iza** (-73.1124, 5.7781)
   - Sistema geotérmico de Boyacá
   - Zona de aguas termales

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Google Earth Engine** - Procesamiento de imágenes satelitales
- **geemap** - Interface Python para Earth Engine
- **rasterio** - Procesamiento de datos geoespaciales
- **matplotlib** - Visualización de imágenes
- **Jupyter Notebook** - Desarrollo interactivo

## 📦 Dataset Utilizado

**ASTER Global Emissivity Dataset 100-meter V003 (AG100)**
- Proveedor: NASA/METI/AIST/Japan Spacesystems
- Resolución: 100 metros
- Bandas de emisividad térmica (bandas 10-14)
- Fuente: [Google Earth Engine Catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_ASTER_GED_AG100_003?hl=es-419)

## 🚀 Instalación

### Prerrequisitos

1. **Cuenta de Google Earth Engine**
   - Regístrate en: https://earthengine.google.com/
   - Crea un proyecto en Google Cloud Platform

2. **Python 3.8 o superior**

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tuusuario/g_earth_geotermia-proyect.git
cd g_earth_geotermia-proyect
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Autenticación en Google Earth Engine**

   **Opción A: Autenticación via CLI**
   ```bash
   earthengine authenticate
   ```

   **Opción B: Autenticación via Python (recomendado)**
   ```python
   import ee
   ee.Authenticate()
   ```
   
   Esto abrirá un navegador para autorizar el acceso a tu cuenta de Google Earth Engine.

5. **Configurar proyecto de Earth Engine**
```bash
earthengine set_project tu-proyecto-id
```

⚠️ **Nota importante**: Necesitas tener una cuenta de Google Earth Engine activa y un proyecto creado en Google Cloud Platform. Si no tienes uno:
- Visita: https://earthengine.google.com/
- Regístrate para obtener acceso
- Crea un proyecto en: https://console.cloud.google.com/

## 💻 Uso

### 1. Modo Interactivo (Jupyter Notebook)

```bash
jupyter notebook descargarimagenes.ipynb
```

El notebook contiene:
- Inicialización de Earth Engine
- Visualización interactiva de mapas
- Descarga de imágenes geotérmicas
- Visualización de datos descargados

### 2. Modo Script (Python)

```bash
python main.py
```

Este script proporciona una visualización básica de una zona geotérmica.

## 📁 Estructura del Proyecto

```
g_earth_geotermia-proyect/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias del proyecto
├── main.py                       # Script principal de visualización
├── descargarimagenes.ipynb       # Notebook interactivo
├── etiquetas_imagenesgeotermia.xlsx  # Etiquetas de clasificación
├── geotermia_imagenes/           # Directorio de imágenes descargadas
│   ├── Nevado_del_Ruiz.tif
│   ├── Volcan_Purace.tif
│   └── Paipa_Iza.tif
└── .ipynb_checkpoints/           # Checkpoints de Jupyter
```

## 🔧 Configuración

### Parámetros de Descarga

En el notebook, puedes modificar:

```python
# Buffer alrededor del punto (en metros)
roi = geom.buffer(5000)  # 5 km de radio

# Banda de emisividad a utilizar
band = dataset.select('emissivity_band10')

# Escala de exportación
scale=100  # 100 metros de resolución
```

### Zonas Personalizadas

Puedes agregar nuevas zonas editando el diccionario en el notebook:

```python
zones = {
    "Tu_Zona": ee.Geometry.Point([longitud, latitud]),
}
```

## 📊 Etiquetado de Datos

Las imágenes se clasifican según su potencial geotérmico:

- **Clase 1 (Potencial Alto)**: Zonas cercanas a volcanes activos, fuentes termales
- **Clase 0 (Sin Potencial)**: Llanos orientales, desiertos, sabanas

El archivo `etiquetas_imagenesgeotermia.xlsx` contiene las etiquetas de entrenamiento.

## 🌐 Recursos Adicionales

- **Mapa de puntos geotérmicos**: [SGC Dashboard](https://sgcolombiano.maps.arcgis.com/apps/dashboards/0186f2c2b6e74866b849025b0bf6fd90)
- **Documentación Earth Engine**: https://developers.google.com/earth-engine
- **Catálogo de Datos**: https://developers.google.com/earth-engine/datasets

## 🔬 Desarrollo Futuro

- [ ] Implementar modelo CNN para clasificación automática
- [ ] Expandir dataset con más zonas geotérmicas
- [ ] Integrar datos de temperatura superficial
- [ ] Crear API para predicciones en tiempo real
- [ ] Visualización web interactiva de resultados

## 📝 Notas Técnicas

- Las imágenes ASTER tienen una resolución espacial de 100m
- La banda 10 de emisividad es útil para detectar anomalías térmicas
- Se recomienda un buffer de 5-10 km alrededor de puntos de interés
- Los archivos .tif son imágenes geoespaciales en formato GeoTIFF

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👥 Autores

- **Cristian Vega** - Desarrollo inicial

## 🙏 Agradecimientos

- NASA/METI/AIST/Japan Spacesystems por el dataset ASTER
- Google Earth Engine por la plataforma de procesamiento
- Servicio Geológico Colombiano por los datos de referencia

## 📧 Contacto

Para preguntas o colaboraciones, por favor abre un issue en el repositorio.

---

⭐ Si este proyecto te resulta útil, ¡no olvides darle una estrella en GitHub!
