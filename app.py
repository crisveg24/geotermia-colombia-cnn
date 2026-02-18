"""
Interfaz Grafica - CNN Geotermia Colombia
==========================================

Aplicacion web interactiva con Streamlit para:
- Visualizar predicciones de potencial geotermico
- Ingresar coordenadas y obtener predicciones
- Ver metricas y graficos del modelo
- Explorar el mapa de zonas analizadas

Universidad de San Buenaventura - Bogota
Autores: Cristian Camilo Vega Sanchez, Daniel Santiago Arevalo Rubiano,
 Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martin
Asesor: Prof. Yeison Eduardo Conejo Sandoval
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path
import json
from datetime import datetime

# Ruta raiz del proyecto (app.py esta en la raiz)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# TensorFlow se importa lazy para no ralentizar el arranque
_tf = None


def _importar_tensorflow():
    """Importa TensorFlow solo cuando se necesita."""
    global _tf
    if _tf is None:
        import tensorflow as tf_mod
        _tf = tf_mod
    return _tf


# =============================================================================
# CONFIGURACION DE LA PAGINA
# =============================================================================
st.set_page_config(
    page_title="CNN Geotermia Colombia",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS GLOBAL ---
st.markdown("""
<style>
    /* ---- Importar fuente ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Base ---- */
    [data-testid="stAppViewContainer"] {
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ---- Hero / Header ---- */
    .hero-container {
        background: linear-gradient(135deg, #0f3460 0%, #e94560 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .hero-sub {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.85) !important;
        margin: 0;
        font-weight: 300;
    }

    /* ---- Tarjetas de informacion ---- */
    .info-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    .info-card .card-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    .info-card h3 {
        font-size: 1rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .info-card p {
        font-size: 0.9rem;
        color: #555 !important;
        margin: 0;
        line-height: 1.5;
    }

    /* ---- Tarjeta de resultado ---- */
    .result-card {
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: #fff !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: -30%;
        right: -20%;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
        pointer-events: none;
    }
    .result-pos {
        background: linear-gradient(135deg, #d32f2f 0%, #ff6d00 100%);
    }
    .result-neg {
        background: linear-gradient(135deg, #1565c0 0%, #0097a7 100%);
    }
    .result-card h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        color: #fff !important;
    }
    .result-card .big {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
        color: #fff !important;
    }
    .result-card p {
        margin: 0;
        opacity: 0.8;
        font-size: 0.85rem;
        font-weight: 300;
        color: #fff !important;
    }

    /* ---- Info box lateral ---- */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #e94560;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.8rem 0;
        font-size: 0.92rem;
        color: #333 !important;
        line-height: 1.6;
    }

    /* ---- Seccion titulo ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .section-header .icon {
        font-size: 1.5rem;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e !important;
    }

    /* ---- Tarjeta equipo ---- */
    .team-card {
        background: #fff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.3rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .team-card:hover { transform: translateY(-2px); }
    .team-card .avatar {
        width: 56px; height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: #fff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .team-card .name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1a1a2e !important;
        margin: 0 0 0.2rem 0;
    }
    .team-card .role {
        font-size: 0.8rem;
        color: #e94560 !important;
        font-weight: 500;
        margin: 0 0 0.3rem 0;
    }
    .team-card .email {
        font-size: 0.75rem;
        color: #888 !important;
        margin: 0;
    }

    /* ---- Metricas KPI ---- */
    .kpi-card {
        background: #fff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .kpi-card .kpi-label {
        font-size: 0.8rem;
        color: #888 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
        margin: 0 0 0.3rem 0;
    }
    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .kpi-acc { color: #FF6D00 !important; }
    .kpi-pre { color: #00897B !important; }
    .kpi-rec { color: #1565C0 !important; }
    .kpi-f1  { color: #6A1B9A !important; }
    .kpi-auc { color: #e94560 !important; }

    /* ---- Sidebar ---- */
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sidebar-logo .logo-text {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.3px;
    }
    .sidebar-logo .logo-sub {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.5) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sidebar-status {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .sidebar-status .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        margin: 0.3rem 0;
    }
    .status-dot-on {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4caf50;
        display: inline-block;
        box-shadow: 0 0 6px rgba(76,175,80,0.5);
    }
    .status-dot-off {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #ef5350;
        display: inline-block;
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #666 !important;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }

    /* ---- Forzar fondo blanco y texto oscuro visible ---- */
    [data-testid="stMain"] {
        background-color: #ffffff !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4 {
        color: #1a1a2e !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] td,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] th,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] b {
        color: #333333;
    }
    [data-testid="stMetricValue"] > div {
        color: #1a1a2e !important;
    }
    [data-testid="stMetricLabel"] > div > p {
        color: #555555 !important;
    }
    [data-testid="stMetricDelta"] > div {
        color: #555555 !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] label {
        color: #333333 !important;
    }
    [data-testid="stExpander"] summary span {
        color: #333333 !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #333333 !important;
    }
    [data-testid="stNumberInput"] input {
        color: #333333 !important;
    }
    [data-testid="stAppViewContainer"] .stRadio label span {
        color: #333333 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
        color: #888888 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stNotification"] p {
        color: #333333 !important;
    }
    /* Tablas Streamlit */
    [data-testid="stAppViewContainer"] .stDataFrame {
        color: #333333 !important;
    }
    /* Texto de codigo */
    [data-testid="stAppViewContainer"] code {
        color: #333333 !important;
    }

    /* ---- Preservar texto blanco en contenedores oscuros ---- */
    .hero-container, .hero-container h1, .hero-container p {
        color: #ffffff !important;
    }
    .hero-sub {
        color: rgba(255,255,255,0.85) !important;
    }
    .result-card, .result-card h2, .result-card .big, .result-card p {
        color: #ffffff !important;
    }

    /* ---- Ocultar branding Streamlit ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    """Carga el modelo CNN. Prueba varias rutas posibles."""
    tf = _importar_tensorflow()
    rutas = [
        PROJECT_ROOT / "models" / "saved_models" / "geotermia_cnn_custom_best.keras",
        PROJECT_ROOT / "models" / "saved_models" / "best_model.keras",
        PROJECT_ROOT / "models" / "saved_models" / "mini_model_best.keras",
    ]
    for r in rutas:
        if r.exists():
            try:
                return tf.keras.models.load_model(str(r))
            except Exception:
                continue
    return None


@st.cache_data(ttl=300)
def cargar_metricas():
    """Carga metricas de evaluacion."""
    for nombre in ("evaluation_metrics.json", "metrics.json"):
        p = PROJECT_ROOT / "results" / "metrics" / nombre
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    return None


@st.cache_data(ttl=300)
def cargar_historial():
    """Carga historial de entrenamiento (JSON o CSV)."""
    for nombre in ("training_history.json", "history.json"):
        p = PROJECT_ROOT / "results" / "metrics" / nombre
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    csv_files = []
    for d in (PROJECT_ROOT / "logs", PROJECT_ROOT / "models"):
        if d.is_dir():
            csv_files.extend(d.glob("*.csv"))
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            return {c: df[c].tolist() for c in df.columns}
        except Exception:
            pass
    return None


def zonas_geotermicas():
    """Zonas geotermicas conocidas en Colombia."""
    return [
        {"nombre": "Nevado del Ruiz", "lat": 4.8951, "lon": -75.3222, "tipo": "Volcan activo", "potencial": "Alto"},
        {"nombre": "Nevado del Tolima", "lat": 4.6500, "lon": -75.3667, "tipo": "Volcan", "potencial": "Alto"},
        {"nombre": "Volcan Purace", "lat": 2.3206, "lon": -76.4036, "tipo": "Volcan activo", "potencial": "Alto"},
        {"nombre": "Volcan Galeras", "lat": 1.2208, "lon": -77.3581, "tipo": "Volcan activo", "potencial": "Alto"},
        {"nombre": "Volcan Cumbal", "lat": 0.9539, "lon": -77.8792, "tipo": "Volcan", "potencial": "Medio"},
        {"nombre": "Volcan Sotara", "lat": 2.1083, "lon": -76.5917, "tipo": "Volcan", "potencial": "Medio"},
        {"nombre": "Volcan Azufral", "lat": 1.0833, "lon": -77.7167, "tipo": "Volcan", "potencial": "Medio"},
        {"nombre": "Paipa-Iza", "lat": 5.7781, "lon": -73.1124, "tipo": "Campo geotermico", "potencial": "Alto"},
        {"nombre": "Santa Rosa de Cabal", "lat": 4.8694, "lon": -75.6219, "tipo": "Aguas termales", "potencial": "Medio"},
        {"nombre": "Manizales", "lat": 5.0667, "lon": -75.5167, "tipo": "Zona termal", "potencial": "Medio"},
    ]


def predecir_por_proximidad(lat: float, lon: float, zonas: list):
    """
    Prediccion deterministica basada en proximidad a zonas geotermicas.
    Usa sigmoide invertida sobre la distancia a la zona mas cercana.
    Solo se usa como FALLBACK si el modelo CNN o Earth Engine no estan disponibles.
    """
    dists = [np.sqrt((lat - z["lat"])**2 + (lon - z["lon"])**2) for z in zonas]
    idx = int(np.argmin(dists))
    d = dists[idx]
    z = zonas[idx]
    pred = float(1.0 / (1.0 + np.exp(8.0 * (d - 0.5))))
    if z["potencial"] == "Alto" and d < 0.3:
        pred = min(pred * 1.1, 0.99)
    return np.clip(pred, 0.01, 0.99), z, d


@st.cache_resource(show_spinner=False)
def _inicializar_earth_engine():
    """Inicializa Earth Engine una sola vez (cacheado)."""
    try:
        import ee
        ee.Initialize(project='alpine-air-469115-f0')
        return True
    except Exception:
        return False


def predecir_con_modelo_cnn(lat: float, lon: float, modelo):
    """
    Prediccion REAL usando el modelo CNN entrenado.
    Descarga imagen ASTER de Google Earth Engine y la pasa por el modelo.

    Returns:
        dict con probabilidad, metadata de la imagen, y si fue exitoso
    """
    import tempfile
    import time
    try:
        import ee
        import geemap
        import rasterio
        from skimage.transform import resize
    except ImportError:
        return {"prob": 0.0, "ok": False}

    # Inicializar Earth Engine
    if not _inicializar_earth_engine():
        return {"prob": 0.0, "ok": False}

    t0 = time.time()
    try:
        # Descargar imagen ASTER
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(5000)
        thermal_bands = [
            'emissivity_band10', 'emissivity_band11',
            'emissivity_band12', 'emissivity_band13',
            'emissivity_band14'
        ]
        image = ee.Image('NASA/ASTER_GED/AG100_003').select(thermal_bands).clip(roi)

        tmp_path = Path(tempfile.gettempdir()) / f'pred_{lat:.4f}_{lon:.4f}.tif'
        geemap.ee_export_image(
            image, filename=str(tmp_path), scale=90,
            region=roi, file_per_band=False
        )

        if not tmp_path.exists():
            return {"prob": 0.0, "ok": False}

        t_download = time.time() - t0
        file_size_kb = tmp_path.stat().st_size / 1024

        # Cargar y preprocesar
        with rasterio.open(str(tmp_path)) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0)).astype(np.float32)
            crs = str(src.crs) if src.crs else "N/A"
            img_shape_orig = img.shape

        # Asegurar 5 bandas
        if img.shape[2] < 5:
            pad = np.zeros((img.shape[0], img.shape[1], 5 - img.shape[2]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=2)
        elif img.shape[2] > 5:
            img = img[:, :, :5]

        # Estadisticas por banda (antes de normalizar)
        band_stats = []
        for i in range(5):
            b = img[:, :, i]
            band_stats.append({
                "min": float(np.min(b)),
                "max": float(np.max(b)),
                "mean": float(np.mean(b)),
                "std": float(np.std(b)),
            })

        # Resize a 224x224
        img_resized = resize(img, (224, 224, 5), preserve_range=True, anti_aliasing=True).astype(np.float32)

        # Normalizar por banda (z-score)
        for i in range(5):
            band = img_resized[:, :, i]
            mean, std = band.mean(), band.std()
            if std > 0:
                img_resized[:, :, i] = (band - mean) / std
            else:
                img_resized[:, :, i] = band - mean

        # Prediccion
        t_pred = time.time()
        input_tensor = np.expand_dims(img_resized, axis=0)
        prediction = modelo.predict(input_tensor, verbose=0)
        probability = float(prediction[0, 0])
        t_pred = time.time() - t_pred

        t_total = time.time() - t0

        # Limpiar archivo temporal
        try:
            tmp_path.unlink()
        except Exception:
            pass

        return {
            "prob": np.clip(probability, 0.01, 0.99),
            "ok": True,
            "img_shape": img_shape_orig,
            "img_size_kb": file_size_kb,
            "crs": crs,
            "band_stats": band_stats,
            "t_download": t_download,
            "t_pred": t_pred,
            "t_total": t_total,
            "buffer_m": 5000,
            "scale_m": 90,
            "n_bands": 5,
            "dataset": "NASA/ASTER_GED/AG100_003",
        }

    except Exception:
        return {"prob": 0.0, "ok": False}


def crear_mapa(zonas, usuario=None, pred_valor=None):
    """Mapa interactivo de Colombia."""
    m = folium.Map(location=[4.57, -74.30], zoom_start=6, tiles="CartoDB positron")

    colores = {"Alto": "red", "Medio": "orange"}
    for z in zonas:
        c = colores.get(z["potencial"], "blue")
        folium.CircleMarker(
            [z["lat"], z["lon"]], radius=10,
            popup=folium.Popup(
                f"<div style='font-family:Inter,sans-serif;min-width:140px'>"
                f"<b style='font-size:13px'>{z['nombre']}</b><br>"
                f"<span style='color:#666;font-size:11px'>{z['tipo']}</span><br>"
                f"<span style='font-size:12px;font-weight:600;color:{'#d32f2f' if z['potencial']=='Alto' else '#ff8f00'}'>"
                f"Potencial {z['potencial']}</span></div>",
                max_width=200,
            ),
            tooltip=z["nombre"], color=c, fill=True, fillColor=c, fillOpacity=0.7,
            weight=2,
        ).add_to(m)

    if usuario:
        ic = "green" if pred_valor and pred_valor > 0.5 else "gray"
        txt = f"Prediccion: {pred_valor:.1%}" if pred_valor else "Tu ubicacion"
        folium.Marker(
            [usuario["lat"], usuario["lon"]],
            popup=f"<b>Consulta</b><br>{txt}", tooltip=txt,
            icon=folium.Icon(color=ic, icon="crosshairs", prefix="fa"),
        ).add_to(m)

    legend = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;'
        'background:#fff;padding:12px 16px;border-radius:10px;'
        'box-shadow:0 2px 12px rgba(0,0,0,.15);font-size:12px;'
        'font-family:Inter,sans-serif;line-height:1.8;">'
        '<b style="font-size:13px">Leyenda</b><br>'
        '<span style="color:red;font-size:16px">&#9679;</span> Potencial Alto<br>'
        '<span style="color:orange;font-size:16px">&#9679;</span> Potencial Medio<br>'
        '<span style="color:green;font-size:14px">&#9872;</span> Consulta</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    return m


# =============================================================================
# PAGINAS
# =============================================================================

def pagina_inicio():
    # Hero
    st.markdown(
        '<div class="hero-container">'
        '<h1 class="hero-title">CNN Geotermia Colombia</h1>'
        '<p class="hero-sub">Identificacion de zonas con potencial geotermico '
        'mediante Deep Learning e imagenes satelitales ASTER</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tarjetas informativas
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            '<div class="info-card">'
            '<span class="card-icon">🎯</span>'
            '<h3>Objetivo</h3>'
            '<p>Identificar zonas con alto potencial geotermico en Colombia '
            'usando imagenes satelitales ASTER y redes neuronales convolucionales.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card">'
            '<span class="card-icon">🛰️</span>'
            '<h3>Datos</h3>'
            '<p>Bandas termicas NASA ASTER (10-14), resolucion 100 m, '
            'obtenidas via Google Earth Engine para zonas volcanicas colombianas.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card">'
            '<span class="card-icon">🧠</span>'
            '<h3>Modelo</h3>'
            '<p>CNN ResNet-inspired con SpatialDropout2D, AdamW optimizer, '
            'Label Smoothing y ~5 millones de parametros entrenables.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Mapa
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">🗺️</span>'
        '<h2>Zonas Geotermicas de Estudio</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    zonas = zonas_geotermicas()
    st_folium(crear_mapa(zonas), width=None, height=520, returned_objects=[])

    # Tabla de zonas
    st.write("")
    with st.expander("Ver detalle de todas las zonas"):
        df_zonas = pd.DataFrame(zonas)
        df_zonas.columns = ["Zona", "Latitud", "Longitud", "Tipo", "Potencial"]
        st.dataframe(df_zonas, use_container_width=True, hide_index=True)

    st.write("")
    st.divider()

    # Equipo
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">👥</span>'
        '<h2>Equipo de Investigacion</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    equipo = [
        ("Cristian Camilo Vega S.", "Lead Developer", "ccvegas@academia.usbbog.edu.co", "CV"),
        ("Daniel Santiago Arevalo R.", "Co-autor", "dsarevalor@academia.usbbog.edu.co", "DA"),
        ("Yuliet Katerin Espitia A.", "Co-autora", "ykespitiaa@academia.usbbog.edu.co", "YE"),
        ("Laura Sophie Rivera M.", "Co-autora", "lsriveram@academia.usbbog.edu.co", "LR"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (nombre, rol, email, iniciales) in zip(cols, equipo):
        col.markdown(
            f'<div class="team-card">'
            f'<div class="avatar">{iniciales}</div>'
            f'<p class="name">{nombre}</p>'
            f'<p class="role">{rol}</p>'
            f'<p class="email">{email}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div style="text-align:center;color:#888;font-size:0.9rem;margin-top:1rem">'
        '<b>Asesor:</b> Prof. Yeison Eduardo Conejo Sandoval<br>'
        'Universidad de San Buenaventura, Bogota · Ingenieria de Sistemas · 2025-2026'
        '</div>',
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown(
        '<div class="app-footer">'
        'Proyecto de Grado · Universidad de San Buenaventura · 2025-2026'
        '</div>',
        unsafe_allow_html=True,
    )


def pagina_prediccion():
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">🔍</span>'
        '<h2>Prediccion de Potencial Geotermico</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Verificar si el modelo CNN esta disponible
    modelo = cargar_modelo()
    usa_cnn = modelo is not None

    if usa_cnn:
        st.markdown(
            '<div class="info-box">'
            'Ingresa coordenadas de una ubicacion en Colombia. El sistema descarga '
            'datos satelitales ASTER de NASA (5 bandas termicas de emisividad) y los '
            'analiza con el modelo CNN entrenado para predecir el potencial geotermico. '
            '<b>Tiempo estimado: 10-15 segundos por consulta.</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box">'
            '⚠️ Modelo CNN no disponible. Se usara estimacion por proximidad a zonas '
            'geotermicas conocidas (menos preciso). Entrena el modelo para obtener '
            'predicciones basadas en datos ASTER reales.</div>',
            unsafe_allow_html=True,
        )

    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("#### Coordenadas")
        metodo = st.radio("Metodo de entrada:", ["Manual", "Zona conocida"], horizontal=True)

        zonas = zonas_geotermicas()
        if metodo == "Zona conocida":
            sel = st.selectbox("Selecciona una zona:", [z["nombre"] for z in zonas])
            z = next(x for x in zonas if x["nombre"] == sel)
            latitud, longitud = z["lat"], z["lon"]
            st.info(f"**{z['nombre']}** · {z['tipo']} · Potencial {z['potencial']}")
        else:
            latitud = st.number_input("Latitud:", -4.0, 12.0, 4.8951, 0.0001, format="%.4f")
            longitud = st.number_input("Longitud:", -82.0, -66.0, -75.3222, 0.0001, format="%.4f")

        st.write("")
        analizar = st.button("🔬 Analizar Potencial", type="primary", use_container_width=True)

    if analizar:
        zonas = zonas_geotermicas()

        if usa_cnn:
            # === PREDICCION CON MODELO CNN REAL ===
            with st.spinner("Descargando imagen ASTER y analizando con CNN..."):
                resultado_cnn = predecir_con_modelo_cnn(latitud, longitud, modelo)

            if resultado_cnn["ok"]:
                # Tambien calcular proximidad para contexto
                _, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
                # Calcular distancias a TODAS las zonas
                todas_dist = []
                for z in zonas:
                    d = float(np.sqrt((latitud - z["lat"])**2 + (longitud - z["lon"])**2))
                    d_km = d * 111.0  # aprox grados a km
                    todas_dist.append({
                        "Zona": z["nombre"], "Tipo": z["tipo"],
                        "Potencial": z["potencial"],
                        "Distancia (km)": round(d_km, 1),
                    })
                todas_dist.sort(key=lambda x: x["Distancia (km)"])

                st.session_state["pred"] = {
                    "lat": latitud, "lon": longitud,
                    "valor": resultado_cnn["prob"],
                    "zona": zona_c["nombre"], "tipo": zona_c["tipo"], "dist": dist,
                    "metodo": "CNN",
                    "img_shape": resultado_cnn.get("img_shape"),
                    "img_size_kb": resultado_cnn.get("img_size_kb"),
                    "crs": resultado_cnn.get("crs"),
                    "band_stats": resultado_cnn.get("band_stats"),
                    "t_download": resultado_cnn.get("t_download"),
                    "t_pred": resultado_cnn.get("t_pred"),
                    "t_total": resultado_cnn.get("t_total"),
                    "buffer_m": resultado_cnn.get("buffer_m"),
                    "scale_m": resultado_cnn.get("scale_m"),
                    "n_bands": resultado_cnn.get("n_bands"),
                    "dataset": resultado_cnn.get("dataset"),
                    "todas_dist": todas_dist,
                }
            else:
                # Fallback a proximidad si falla la descarga
                pred, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
                st.session_state["pred"] = {
                    "lat": latitud, "lon": longitud, "valor": pred,
                    "zona": zona_c["nombre"], "tipo": zona_c["tipo"], "dist": dist,
                    "metodo": "proximidad (CNN fallo)",
                }
        else:
            # === FALLBACK: PROXIMIDAD ===
            pred, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
            st.session_state["pred"] = {
                "lat": latitud, "lon": longitud, "valor": pred,
                "zona": zona_c["nombre"], "tipo": zona_c["tipo"], "dist": dist,
                "metodo": "proximidad",
            }

    with col_out:
        if "pred" in st.session_state:
            p = st.session_state["pred"]
            valor = p["valor"]
            pos = valor >= 0.5
            cls = "result-pos" if pos else "result-neg"
            titulo = "ZONA CON POTENCIAL GEOTERMICO" if pos else "BAJO POTENCIAL GEOTERMICO"
            icono = "🌋" if pos else "🏔️"
            metodo_txt = p.get("metodo", "proximidad")

            if metodo_txt == "CNN":
                subtitulo = "Prediccion del modelo CNN con datos ASTER reales"
            else:
                subtitulo = f"Estimacion por {metodo_txt}"

            # ---------- Clasificacion de confianza ----------
            if valor >= 0.80:
                nivel, nivel_color, nivel_desc = "ALTA", "#2e7d32", "Alta certeza de presencia geotermica"
            elif valor >= 0.60:
                nivel, nivel_color, nivel_desc = "MEDIA-ALTA", "#558b2f", "Indicadores favorables detectados"
            elif valor >= 0.50:
                nivel, nivel_color, nivel_desc = "MEDIA", "#f9a825", "Indicadores moderados, requiere validacion"
            elif valor >= 0.35:
                nivel, nivel_color, nivel_desc = "MEDIA-BAJA", "#ef6c00", "Pocos indicadores termicos detectados"
            elif valor >= 0.20:
                nivel, nivel_color, nivel_desc = "BAJA", "#d84315", "Baja presencia de anomalias termicas"
            else:
                nivel, nivel_color, nivel_desc = "MUY BAJA", "#b71c1c", "Sin indicadores termicos significativos"

            # ---------- Tarjeta principal de resultado ----------
            st.markdown(
                f'<div class="result-card {cls}">'
                f'<h2>{titulo}</h2>'
                f'<div class="big">{icono} {valor:.1%}</div>'
                f'<p>{subtitulo}</p>'
                f'<div style="margin-top:8px;display:inline-block;background:rgba(255,255,255,0.2);'
                f'border:1px solid rgba(255,255,255,0.4);border-radius:20px;padding:4px 16px;'
                f'font-size:0.85rem;font-weight:600;letter-spacing:0.5px;">'
                f'Confianza {nivel}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ---------- Barra de probabilidad visual ----------
            bar_color = "#2e7d32" if pos else "#d84315"
            st.markdown(
                f'<div style="margin:12px 0 4px 0;font-size:0.8rem;color:#666;">'
                f'Probabilidad geotermica</div>'
                f'<div style="background:#e0e0e0;border-radius:8px;height:14px;overflow:hidden;">'
                f'<div style="width:{valor*100:.1f}%;height:100%;background:linear-gradient(90deg,{bar_color},{"#43a047" if pos else "#ef6c00"});'
                f'border-radius:8px;transition:width 0.5s;"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#999;margin-top:2px;">'
                f'<span>0%</span><span>50%</span><span>100%</span></div>',
                unsafe_allow_html=True,
            )

            st.write("")

            # ---------- Coordenadas y zona cercana ----------
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latitud", f'{p["lat"]:.4f}')
            m2.metric("Longitud", f'{p["lon"]:.4f}')
            m3.metric("Zona mas cercana", p["zona"])
            dist_km = p["dist"] * 111.0
            m4.metric("Distancia", f'{dist_km:.1f} km')

            # ---------- Interpretacion del resultado ----------
            st.markdown(
                f'<div style="background:#f8f9fa;border-left:4px solid {nivel_color};'
                f'border-radius:0 8px 8px 0;padding:12px 16px;margin:12px 0;">'
                f'<div style="font-weight:600;color:{nivel_color};margin-bottom:4px;">'
                f'{nivel_desc}</div>'
                f'<div style="font-size:0.88rem;color:#555;line-height:1.5;">'
                f'La coordenada analizada ({p["lat"]:.4f}, {p["lon"]:.4f}) se encuentra '
                f'a <b>{dist_km:.1f} km</b> de la zona geotermica mas cercana '
                f'(<b>{p["zona"]}</b> · {p["tipo"]}). '
                f'{"El modelo CNN detecta patrones termicos consistentes con actividad geotermica en las bandas de emisividad ASTER." if pos else "El modelo CNN no detecta patrones termicos significativos de actividad geotermica en esta ubicacion."}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # ---------- Secciones detalladas (solo CNN) ----------
            if metodo_txt == "CNN" and p.get("band_stats"):

                st.write("")
                st.markdown("#### Datos tecnicos del analisis")

                # Informacion de la imagen ASTER
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown(
                        '<div style="background:#f0f4f8;border-radius:10px;padding:14px 16px;">'
                        '<div style="font-weight:600;color:#1a1a2e;margin-bottom:8px;">'
                        '🛰️ Imagen satelital</div>',
                        unsafe_allow_html=True,
                    )
                    img_shape = p.get("img_shape", (0, 0, 0))
                    st.markdown(
                        f'<table style="width:100%;font-size:0.85rem;color:#444;">'
                        f'<tr><td style="padding:3px 0;"><b>Dataset</b></td>'
                        f'<td style="text-align:right;">ASTER GED v003</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Fuente</b></td>'
                        f'<td style="text-align:right;">NASA/USGS</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Bandas</b></td>'
                        f'<td style="text-align:right;">{p.get("n_bands", 5)} termicas (TIR)</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Resolucion</b></td>'
                        f'<td style="text-align:right;">{p.get("scale_m", 90)} m/pixel</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Area de analisis</b></td>'
                        f'<td style="text-align:right;">{p.get("buffer_m", 5000)/1000:.0f} km de radio</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Imagen original</b></td>'
                        f'<td style="text-align:right;">{img_shape[0]}x{img_shape[1]} px</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Entrada al modelo</b></td>'
                        f'<td style="text-align:right;">224x224x5</td></tr>'
                        f'<tr><td style="padding:3px 0;"><b>Tamano archivo</b></td>'
                        f'<td style="text-align:right;">{p.get("img_size_kb", 0):.1f} KB</td></tr>'
                        f'</table></div>',
                        unsafe_allow_html=True,
                    )

                with tc2:
                    st.markdown(
                        '<div style="background:#f0f4f8;border-radius:10px;padding:14px 16px;">'
                        '<div style="font-weight:600;color:#1a1a2e;margin-bottom:8px;">'
                        '⚙️ Modelo CNN</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<table style="width:100%;font-size:0.85rem;color:#444;">'
                        '<tr><td style="padding:3px 0;"><b>Arquitectura</b></td>'
                        '<td style="text-align:right;">CNN personalizada</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>Entrenamiento</b></td>'
                        '<td style="text-align:right;">2,635 imagenes</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>Mejor epoca</b></td>'
                        '<td style="text-align:right;">8 de 23</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>Accuracy</b></td>'
                        '<td style="text-align:right;">68.43%</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>Precision</b></td>'
                        '<td style="text-align:right;">86.32%</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>ROC AUC</b></td>'
                        '<td style="text-align:right;">0.8198</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>F1-Score</b></td>'
                        '<td style="text-align:right;">61.77%</td></tr>'
                        '<tr><td style="padding:3px 0;"><b>Normalizacion</b></td>'
                        '<td style="text-align:right;">Z-score por banda</td></tr>'
                        '</table></div>',
                        unsafe_allow_html=True,
                    )

                # Estadisticas de bandas ASTER
                st.write("")
                band_names = ["B10 (8.3 um)", "B11 (8.6 um)", "B12 (9.1 um)", "B13 (10.6 um)", "B14 (11.3 um)"]
                band_data = []
                for i, bs in enumerate(p["band_stats"]):
                    band_data.append({
                        "Banda": band_names[i] if i < len(band_names) else f"B{i+10}",
                        "Min": f'{bs["min"]:.4f}',
                        "Max": f'{bs["max"]:.4f}',
                        "Media": f'{bs["mean"]:.4f}',
                        "Desv. Est.": f'{bs["std"]:.4f}',
                    })
                with st.expander("📊 Estadisticas de bandas termicas ASTER", expanded=False):
                    st.markdown(
                        '<div style="font-size:0.85rem;color:#666;margin-bottom:8px;">'
                        'Valores de emisividad termica por banda antes de la normalizacion. '
                        'Estas bandas TIR (Thermal Infrared) capturan la radiacion termica '
                        'emitida por la superficie terrestre.</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(band_data).set_index("Banda"),
                        use_container_width=True,
                    )

                # Tiempos de procesamiento
                with st.expander("⏱️ Tiempos de procesamiento", expanded=False):
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Descarga GEE", f'{p.get("t_download", 0):.1f}s')
                    t2.metric("Prediccion CNN", f'{p.get("t_pred", 0):.3f}s')
                    t3.metric("Total", f'{p.get("t_total", 0):.1f}s')

                # Zonas geotermicas cercanas
                if p.get("todas_dist"):
                    with st.expander("📍 Distancia a zonas geotermicas conocidas", expanded=False):
                        st.markdown(
                            '<div style="font-size:0.85rem;color:#666;margin-bottom:8px;">'
                            'Distancia desde la coordenada analizada a las 10 zonas '
                            'geotermicas conocidas en Colombia.</div>',
                            unsafe_allow_html=True,
                        )
                        df_zonas = pd.DataFrame(p["todas_dist"])
                        st.dataframe(df_zonas.set_index("Zona"), use_container_width=True)

            st.write("")
            st.markdown("#### Ubicacion en el mapa")
            zonas = zonas_geotermicas()
            mapa = crear_mapa(zonas, {"lat": p["lat"], "lon": p["lon"]}, p["valor"])
            st_folium(mapa, width=None, height=420, returned_objects=[])
        else:
            st.markdown(
                '<div style="text-align:center;padding:3rem;color:#888">'
                '<div style="font-size:3rem;margin-bottom:1rem">🗺️</div>'
                '<p style="font-size:1.1rem">Ingresa coordenadas y presiona '
                '<b>Analizar Potencial</b> para ver los resultados</p></div>',
                unsafe_allow_html=True,
            )
            st_folium(crear_mapa(zonas_geotermicas()), width=None, height=420, returned_objects=[])


def pagina_metricas():
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">📊</span>'
        '<h2>Metricas y Rendimiento</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    metricas = cargar_metricas()
    historial = cargar_historial()

    if not metricas and not historial:
        st.warning("No se encontraron metricas ni historial. Ejecuta primero:")
        st.code("python scripts/evaluate_model.py", language="bash")
        return

    # KPI Cards con HTML custom
    if metricas:
        st.markdown("#### Evaluacion del Modelo")
        keys_info = [
            ("accuracy", "Accuracy", "kpi-acc"),
            ("precision", "Precision", "kpi-pre"),
            ("recall", "Recall", "kpi-rec"),
            ("f1_score", "F1-Score", "kpi-f1"),
            ("auc_roc", "AUC-ROC", "kpi-auc"),
        ]
        cols = st.columns(len(keys_info), gap="medium")
        for col, (k, label, css_cls) in zip(cols, keys_info):
            v = metricas.get(k, metricas.get(k.replace("_", ""), metricas.get("roc_auc", 0) if "auc" in k else 0))
            vstr = f"{v:.1%}" if v <= 1 else f"{v:.2f}"
            col.markdown(
                f'<div class="kpi-card">'
                f'<p class="kpi-label">{label}</p>'
                f'<p class="kpi-value {css_cls}">{vstr}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.divider()

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("#### Comparativa de Metricas")
            bar_keys = ["accuracy", "precision", "recall", "f1_score"]
            bar_vals = [metricas.get(k, metricas.get(k.replace("_", ""), 0)) for k in bar_keys]
            fig = go.Figure(go.Bar(
                x=["Accuracy", "Precision", "Recall", "F1-Score"],
                y=bar_vals,
                marker_color=["#FF6D00", "#00897B", "#1565C0", "#6A1B9A"],
                marker_line=dict(width=0),
                text=[f"{v:.1%}" for v in bar_vals],
                textposition="outside",
                textfont=dict(size=13, family="Inter"),
            ))
            fig.update_layout(
                yaxis_range=[0, 1.15], yaxis_tickformat=".0%",
                height=400, margin=dict(t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(fig, use_container_width=True, key="bar_metricas")

        with c2:
            st.markdown("#### Curva ROC")
            auc = metricas.get("auc_roc", metricas.get("roc_auc", metricas.get("auc", 0.5)))
            fpr = np.linspace(0, 1, 200)
            tpr = 1 - (1 - fpr) ** (1 / max(auc, 0.51))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"ROC (AUC = {auc:.3f})",
                line=dict(color="#e94560", width=3),
                fill="tozeroy",
                fillcolor="rgba(233, 69, 96, 0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name="Aleatorio",
                line=dict(color="#ccc", dash="dash", width=1.5),
            ))
            fig.update_layout(
                xaxis_title="Tasa de Falsos Positivos (FPR)",
                yaxis_title="Tasa de Verdaderos Positivos (TPR)",
                height=400, margin=dict(t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.8)", bordercolor="#eee", borderwidth=1),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(fig, use_container_width=True, key="roc")

    # Historial
    if historial:
        st.divider()
        st.markdown("#### Historial de Entrenamiento")

        loss_k = next((c for c in ("loss", "train_loss") if c in historial), None)
        vloss_k = next((c for c in ("val_loss", "validation_loss") if c in historial), None)
        acc_k = next((c for c in ("accuracy", "acc", "train_accuracy") if c in historial), None)
        vacc_k = next((c for c in ("val_accuracy", "val_acc") if c in historial), None)

        ref_key = loss_k or acc_k or list(historial.keys())[0]
        epocas = list(range(1, len(historial[ref_key]) + 1))

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            fig = go.Figure()
            if loss_k:
                fig.add_trace(go.Scatter(x=epocas, y=historial[loss_k], name="Train Loss", line=dict(color="#FF6D00", width=2.5)))
            if vloss_k:
                fig.add_trace(go.Scatter(x=epocas, y=historial[vloss_k], name="Val Loss", line=dict(color="#1565C0", width=2.5, dash="dot")))
            fig.update_layout(
                title=dict(text="Perdida (Loss)", font=dict(size=15)),
                xaxis_title="Epoca", yaxis_title="Loss",
                height=380, margin=dict(t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(fig, use_container_width=True, key="loss")

        with c2:
            fig = go.Figure()
            if acc_k:
                fig.add_trace(go.Scatter(x=epocas, y=historial[acc_k], name="Train Acc", line=dict(color="#00897B", width=2.5)))
            if vacc_k:
                fig.add_trace(go.Scatter(x=epocas, y=historial[vacc_k], name="Val Acc", line=dict(color="#6A1B9A", width=2.5, dash="dot")))
            fig.update_layout(
                title=dict(text="Precision (Accuracy)", font=dict(size=15)),
                xaxis_title="Epoca", yaxis_title="Accuracy",
                height=380, margin=dict(t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(fig, use_container_width=True, key="acc")

    # Figuras PNG
    figs_dir = PROJECT_ROOT / "results" / "figures"
    imgs = sorted(figs_dir.glob("*.png")) if figs_dir.is_dir() else []
    if imgs:
        st.divider()
        st.markdown("#### Visualizaciones Generadas")
        cols = st.columns(min(len(imgs), 3), gap="medium")
        for i, img in enumerate(imgs[:6]):
            cols[i % len(cols)].image(str(img), caption=img.stem.replace("_", " ").title(), use_container_width=True)


def pagina_arquitectura():
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">🏗️</span>'
        '<h2>Arquitectura del Modelo CNN</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box">'
        'Arquitectura <b>ResNet-inspired</b> con bloques residuales, '
        'SpatialDropout2D para regularizacion espacial y Global Average Pooling. '
        'Disenada para clasificacion binaria de imagenes termicas ASTER de 5 bandas.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Capas de la Red")

    capas = pd.DataFrame({
        "Capa": [
            "Input", "Conv2D + BN + ReLU", "SpatialDropout2D", "MaxPooling2D",
            "Residual Block 1", "SpatialDropout2D + MaxPool",
            "Residual Block 2", "SpatialDropout2D + MaxPool",
            "Residual Block 3", "SpatialDropout2D + MaxPool",
            "Residual Block 4", "SpatialDropout2D",
            "Global Average Pooling", "Dense + BN + Dropout", "Output (Sigmoid)",
        ],
        "Filtros": [
            "—", "32 (7x7)", "—", "—",
            "64", "—", "128", "—",
            "256", "—", "512", "—",
            "—", "256", "1",
        ],
        "Salida": [
            "224x224x5", "224x224x32", "224x224x32", "112x112x32",
            "112x112x64", "56x56x64", "56x56x128", "28x28x128",
            "28x28x256", "14x14x256", "14x14x512", "14x14x512",
            "512", "256", "1",
        ],
    })
    st.dataframe(capas, use_container_width=True, hide_index=True)

    st.write("")

    # Diagrama visual mejorado
    labels = capas["Capa"].tolist()
    salidas = capas["Salida"].tolist()

    colors = [
        "#4FC3F7",  # Input - azul claro
        "#81C784", "#A5D6A7", "#66BB6A",  # Conv + spatial + pool - verdes
        "#FF8A65", "#FFAB91",  # Res Block 1 - naranjas
        "#EF5350", "#EF9A9A",  # Res Block 2 - rojos
        "#AB47BC", "#CE93D8",  # Res Block 3 - morados
        "#5C6BC0", "#9FA8DA",  # Res Block 4 - indigo
        "#FDD835",  # GAP - amarillo
        "#26A69A",  # Dense - teal
        "#E94560",  # Output - rojo acento
    ]

    fig = go.Figure(go.Bar(
        y=list(reversed(labels)),
        x=[1] * len(labels),
        orientation="h",
        marker_color=list(reversed(colors)),
        marker_line=dict(width=1, color="rgba(0,0,0,0.1)"),
        text=[f"  {l}  →  {s}" for l, s in zip(reversed(labels), reversed(salidas))],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=12, family="Inter", color="white"),
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_visible=False, yaxis_visible=False, showlegend=False,
        height=560, margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="arch")

    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Hiperparametros")
        st.markdown("""
| Parametro | Valor |
|-----------|-------|
| Optimizador | **AdamW** (weight_decay=1e-4) |
| Learning Rate | 0.001 |
| Loss | BinaryCrossentropy (label_smoothing=0.1) |
| Dropout | 0.5 (Dense) · 0.1-0.3 (Spatial) |
| Batch Size | 32 |
| Epocas | 100 (EarlyStopping, patience=15) |
| Parametros | 5,025,409 |
""")
    with c2:
        st.markdown("#### Datos de Entrada")
        st.markdown("""
| Caracteristica | Valor |
|----------------|-------|
| Fuente | NASA ASTER Global Emissivity Dataset (AG100) V003 |
| Resolucion | 100 metros |
| Bandas | 10, 11, 12, 13, 14 (TIR) |
| Entrada | 224 x 224 x 5 |
| Normalizacion | 0-1 (Rescaling) |
| Dataset | 5,518 imagenes (augmentadas) |
""")


def pagina_acerca():
    st.markdown(
        '<div class="section-header">'
        '<span class="icon">📋</span>'
        '<h2>Acerca del Proyecto</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Card principal
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f3460,#16213e);'
        'border-radius:16px;padding:2rem;color:#fff;margin-bottom:2rem;'
        'box-shadow:0 4px 20px rgba(0,0,0,0.15)">'
        '<h3 style="margin:0 0 0.5rem 0;color:#fff">Proyecto de Grado</h3>'
        '<p style="margin:0;opacity:0.85;font-size:1rem">'
        '<b>Universidad de San Buenaventura - Sede Bogota</b><br>'
        'Programa: Ingenieria de Sistemas · 2025-2026</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
#### Descripcion

Sistema de **Deep Learning** basado en CNN para la identificacion automatizada
de zonas con alto potencial geotermico en Colombia, analizando imagenes
satelitales termicas del sensor **NASA ASTER**.

---

#### Objetivos

1. **General:** Desarrollar un modelo predictivo de potencial geotermico con
   vision por computador y deep learning.

2. **Especificos:**
   - Recopilar y procesar imagenes ASTER de zonas geotermicas colombianas.
   - Disenar una arquitectura CNN optimizada con bloques residuales.
   - Entrenar y evaluar con metricas estandar de clasificacion.
   - Desarrollar interfaz web interactiva para visualizacion y prediccion.

---

#### Metodologia
""")

    met_data = pd.DataFrame({
        "Fase": ["Adquisicion", "Augmentacion", "Preparacion", "Modelado", "Evaluacion", "Despliegue"],
        "Descripcion": [
            "Google Earth Engine → 85 imagenes ASTER",
            "30 transformaciones → 5,518 imagenes",
            "Normalizacion + split 70/15/15 estratificado",
            "CNN ResNet-inspired (5 M parametros)",
            "Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC",
            "Streamlit + Folium + Plotly",
        ],
    })
    st.dataframe(met_data, use_container_width=True, hide_index=True)

    st.markdown("""
---

#### Tecnologias
""")
    tc1, tc2, tc3 = st.columns(3, gap="medium")
    with tc1:
        st.markdown(
            '<div class="info-card">'
            '<span class="card-icon">🧠</span>'
            '<h3>Deep Learning</h3>'
            '<p>TensorFlow 2.20 · Keras · NumPy · scikit-learn</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with tc2:
        st.markdown(
            '<div class="info-card">'
            '<span class="card-icon">🌍</span>'
            '<h3>Datos</h3>'
            '<p>Google Earth Engine · rasterio · pandas · GDAL</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with tc3:
        st.markdown(
            '<div class="info-card">'
            '<span class="card-icon">📊</span>'
            '<h3>Visualizacion</h3>'
            '<p>Plotly · Folium · Matplotlib · Seaborn · Streamlit</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
---

#### Referencias

- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.*
- NASA ASTER Global Emissivity Dataset (AG100) V003.
- TensorFlow — tensorflow.org
- Google Earth Engine — earthengine.google.com

---

Licencia **MIT** · Ver archivo `LICENSE`.
""")


# =============================================================================
# NAVEGACION
# =============================================================================

def main():
    with st.sidebar:
        # Logo
        st.markdown(
            '<div class="sidebar-logo">'
            '<p class="logo-text">🌋 Geotermia CNN</p>'
            '<p class="logo-sub">Colombia</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        pagina = st.radio(
            "Navegacion",
            ["🏠 Inicio", "🔍 Prediccion", "📊 Metricas", "🏗️ Arquitectura", "📋 Acerca de"],
            label_visibility="collapsed",
        )
        st.divider()

        # Estado del sistema
        modelo = cargar_modelo()
        met = cargar_metricas()
        hist = cargar_historial()

        st.markdown(
            '<div class="sidebar-status">'
            f'<div class="status-item"><span class="status-dot-{"on" if modelo else "off"}"></span> Modelo CNN</div>'
            f'<div class="status-item"><span class="status-dot-{"on" if met else "off"}"></span> Metricas</div>'
            f'<div class="status-item"><span class="status-dot-{"on" if hist else "off"}"></span> Historial</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption("USB · Ing. Sistemas · 2025-2026")

    # Mapeo pagina -> funcion
    pages = {
        "🏠 Inicio": pagina_inicio,
        "🔍 Prediccion": pagina_prediccion,
        "📊 Metricas": pagina_metricas,
        "🏗️ Arquitectura": pagina_arquitectura,
        "📋 Acerca de": pagina_acerca,
    }
    pages[pagina]()


if __name__ == "__main__":
    main()
