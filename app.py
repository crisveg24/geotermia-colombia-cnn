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
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import sys
from pathlib import Path
import json
import io
import logging
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
                model = tf.keras.models.load_model(str(r))
                # v2: Registrar qué modelo se cargó (BUG 8)
                st.session_state["modelo_cargado_nombre"] = r.name
                logging.info(f"Modelo cargado: {r.name}")
                return model
            except Exception as e:
                logging.warning(f"No se pudo cargar {r.name}: {e}")
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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia en km entre dos puntos usando fórmula de Haversine.
    v2: Reemplaza distancia Euclidea en grados × 111 (BUG 13).
    """
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def predecir_por_proximidad(lat: float, lon: float, zonas: list):
    """
    Prediccion deterministica basada en proximidad a zonas geotermicas.
    Usa sigmoide invertida sobre la distancia a la zona mas cercana.
    Solo se usa como FALLBACK si el modelo CNN o Earth Engine no estan disponibles.
    """
    # v2: Usar Haversine en vez de Euclidea en grados (BUG 13)
    dists_km = [_haversine_km(lat, lon, z["lat"], z["lon"]) for z in zonas]
    idx = int(np.argmin(dists_km))
    d_km = dists_km[idx]
    z = zonas[idx]
    # Convertir km a "grados equivalentes" para compatibilidad con sigmoide existente
    d = d_km / 111.0
    pred = float(1.0 / (1.0 + np.exp(8.0 * (d - 0.5))))
    if z["potencial"] == "Alto" and d < 0.3:
        pred = min(pred * 1.1, 0.99)
    return np.clip(pred, 0.01, 0.99), z, d


@st.cache_resource(show_spinner=False)
def _inicializar_earth_engine():
    """Inicializa Earth Engine una sola vez (cacheado)."""
    try:
        import ee
        # v2: Usar GEE_PROJECT centralizado desde config.py (BUG 10)
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from config import cfg
        ee.Initialize(project=cfg.GEE_PROJECT)
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
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import cfg
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
        aster_bands = [
            'emissivity_band10', 'emissivity_band11',
            'emissivity_band12', 'emissivity_band13',
            'emissivity_band14', 'temperature', 'ndvi'
        ]
        image = ee.Image('NASA/ASTER_GED/AG100_003').select(aster_bands).clip(roi)

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

        # Asegurar 7 bandas (5 emisividad + temperature + NDVI)
        n_bands = cfg.NUM_BANDS  # 7
        if img.shape[2] < n_bands:
            pad = np.zeros((img.shape[0], img.shape[1], n_bands - img.shape[2]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=2)
        elif img.shape[2] > n_bands:
            img = img[:, :, :n_bands]

        # Estadisticas por banda (antes de normalizar)
        band_stats = []
        for i in range(n_bands):
            b = img[:, :, i]
            # v2: Filtrar NoData (-9999) de estadísticas y normalización
            valid = b[b > -9999]
            if len(valid) > 0:
                band_stats.append({
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "nodata_pct": float((b <= -9999).mean() * 100),
                })
            else:
                band_stats.append({"min": 0, "max": 0, "mean": 0, "std": 0, "nodata_pct": 100.0})

        # v2: Reemplazar NoData con mediana por banda antes de resize
        for i in range(n_bands):
            band = img[:, :, i]
            valid = band[band > -9999]
            if len(valid) > 0:
                band[band <= -9999] = np.median(valid)
            else:
                band[band <= -9999] = 0

        # 1. No hacemos resize directo a 224x224 para Sliding Window
        # Mantenemos las proporciones
        img_resized = img.astype(np.float32)

        # Normalizar por banda (z-score)
        for i in range(n_bands):
            band = img_resized[:, :, i]
            mean, std = band.mean(), band.std()
            if std > 0:
                img_resized[:, :, i] = (band - mean) / std
            else:
                img_resized[:, :, i] = band - mean

        # Prediccion con Sliding Window si es mas grande que 224x224
        t_pred = time.time()
        
        h_img, w_img, c = img_resized.shape
        h_win, w_win = (224, 224)
        stride = 112
        
        if h_img <= h_win and w_img <= w_win:
            # Padding si es mas pequeña
            padded = np.zeros((h_win, w_win, c), dtype=np.float32)
            padded[:h_img, :w_img, :] = img_resized
            input_tensor = np.expand_dims(padded, axis=0)
            prediction = modelo.predict(input_tensor, verbose=0)
            probability = float(prediction[0, 0])
        else:
            # Sliding window real
            windows = []
            for y in range(0, h_img - h_win + 1, stride):
                for x in range(0, w_img - w_win + 1, stride):
                    windows.append(img_resized[y:y+h_win, x:x+w_win, :])
            
            # Asegurar bordes
            if (h_img - h_win) % stride != 0:
                for x in range(0, w_img - w_win + 1, stride):
                    windows.append(img_resized[-h_win:, x:x+w_win, :])
            if (w_img - w_win) % stride != 0:
                for y in range(0, h_img - h_win + 1, stride):
                    windows.append(img_resized[y:y+h_win, -w_win:, :])
            if (h_img - h_win) % stride != 0 and (w_img - w_win) % stride != 0:
                windows.append(img_resized[-h_win:, -w_win:, :])
                
            if not windows:
                padded = np.zeros((h_win, w_win, c), dtype=np.float32)
                padded[:h_img, :w_img, :] = img_resized
                windows.append(padded)
                
            batch = np.array(windows)
            predictions = modelo.predict(batch, batch_size=32, verbose=0)
            
            if predictions.shape[1] == 1:
                probability = float(np.max(predictions))
            else:
                probability = float(np.max(predictions[:, 1]))
                
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
            "n_bands": n_bands,
            "dataset": "NASA/ASTER_GED/AG100_003",
        }

    except Exception as e:
        # v2: Registrar error en vez de silenciarlo (BUG 7)
        import traceback
        logging.error(f"Error en predicción CNN: {e}\n{traceback.format_exc()}")
        return {"prob": 0.0, "ok": False, "error": str(e)}


def generar_reporte_texto(p: dict) -> str:
    """Genera reporte de prediccion en formato texto para descarga."""
    valor = p["valor"]
    pos = valor >= 0.5
    dist_km = p["dist"] * 111.0

    if valor >= 0.80:
        nivel = "ALTA"
    elif valor >= 0.60:
        nivel = "MEDIA-ALTA"
    elif valor >= 0.50:
        nivel = "MEDIA"
    elif valor >= 0.35:
        nivel = "MEDIA-BAJA"
    elif valor >= 0.20:
        nivel = "BAJA"
    else:
        nivel = "MUY BAJA"

    lines = [
        "=" * 60,
        "   REPORTE DE PREDICCION GEOTERMICA",
        "   CNN Geotermia Colombia — Universidad de San Buenaventura",
        "=" * 60,
        "",
        f"Fecha:            {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Metodo:           {p.get('metodo', 'N/A')}",
        "",
        "--- RESULTADO ---",
        f"Prediccion:       {'CON POTENCIAL GEOTERMICO' if pos else 'BAJO POTENCIAL GEOTERMICO'}",
        f"Probabilidad:     {valor:.1%}",
        f"Confianza:        {nivel}",
        "",
        "--- UBICACION ---",
        f"Latitud:          {p['lat']:.4f}",
        f"Longitud:         {p['lon']:.4f}",
        f"Zona mas cercana: {p['zona']} ({p['tipo']})",
        f"Distancia:        {dist_km:.1f} km",
    ]

    if p.get("band_stats"):
        lines += [
            "",
            "--- IMAGEN SATELITAL ---",
            f"Dataset:          {p.get('dataset', 'ASTER GED v003')}",
            f"Bandas:           {p.get('n_bands', 7)} (5 TIR + temperature + NDVI)",
            f"Resolucion:       {p.get('scale_m', 90)} m/pixel",
            f"Area analizada:   {p.get('buffer_m', 5000)/1000:.0f} km de radio",
            f"Imagen original:  {p.get('img_shape', (0,0,0))[0]}x{p.get('img_shape', (0,0,0))[1]} px",
            f"Entrada modelo:   224x224x7",
            f"Tamano archivo:   {p.get('img_size_kb', 0):.1f} KB",
            "",
            "--- ESTADISTICAS DE BANDAS ---",
            f"{'Banda':<16} {'Min':>10} {'Max':>10} {'Media':>10} {'Desv.Est':>10}",
        ]
        band_names = ["B10 (8.3um)", "B11 (8.6um)", "B12 (9.1um)", "B13 (10.6um)", "B14 (11.3um)"]
        for i, bs in enumerate(p["band_stats"]):
            bn = band_names[i] if i < len(band_names) else f"B{i+10}"
            lines.append(
                f"{bn:<16} {bs['min']:>10.4f} {bs['max']:>10.4f} {bs['mean']:>10.4f} {bs['std']:>10.4f}"
            )

        lines += [
            "",
            "--- TIEMPOS ---",
            f"Descarga GEE:     {p.get('t_download', 0):.1f}s",
            f"Prediccion CNN:   {p.get('t_pred', 0):.3f}s",
            f"Total:            {p.get('t_total', 0):.1f}s",
        ]

    if p.get("todas_dist"):
        lines += ["", "--- DISTANCIA A ZONAS CONOCIDAS ---"]
        lines.append(f"{'Zona':<25} {'Tipo':<18} {'Potencial':<10} {'Dist(km)':>10}")
        for zd in p["todas_dist"]:
            lines.append(
                f"{zd['Zona']:<25} {zd['Tipo']:<18} {zd['Potencial']:<10} {zd['Distancia (km)']:>10.1f}"
            )

    # v2: Leer métricas dinámicamente si están disponibles (BUG 15)
    met = cargar_metricas()
    if met:
        acc_str = f"{met.get('accuracy', 0)*100:.2f}%"
        prec_str = f"{met.get('precision', 0)*100:.2f}%"
        roc_str = f"{met.get('roc_auc', 0):.4f}"
        f1_str = f"{met.get('f1_score', 0)*100:.2f}%"
        epoch_str = "ver historial"
    else:
        acc_str = "N/A (sin métricas)"
        prec_str = "N/A"
        roc_str = "N/A"
        f1_str = "N/A"
        epoch_str = "N/A"

    lines += [
        "",
        "--- MODELO ---",
        "Arquitectura:     CNN personalizada",
        f"Accuracy:         {acc_str}",
        f"Precision:        {prec_str}",
        f"ROC AUC:          {roc_str}",
        f"F1-Score:         {f1_str}",
        "Normalizacion:    Z-score por banda",
        "",
        "=" * 60,
        "Autores: C.Vega, D.Arevalo, Y.Espitia, L.Rivera",
        "Asesor: Prof. Yeison Eduardo Conejo Sandoval",
        "Universidad de San Buenaventura — Bogota — 2025-2026",
        "=" * 60,
    ]
    return "\n".join(lines)


def crear_mapa_heatmap(zonas, predicciones_hist=None):
    """Mapa con capa de calor basada en predicciones realizadas."""
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

    # Capa de calor con predicciones
    if predicciones_hist:
        heat_data = []
        for h in predicciones_hist:
            heat_data.append([h["lat"], h["lon"], h["valor"]])
            # Marcador por cada prediccion
            es_pos = h["valor"] >= 0.5
            folium.CircleMarker(
                [h["lat"], h["lon"]], radius=6,
                color="#2e7d32" if es_pos else "#1565c0",
                fill=True,
                fillColor="#4caf50" if es_pos else "#42a5f5",
                fillOpacity=0.8, weight=1.5,
                tooltip=f'{h["valor"]:.1%} — {h.get("zona", "")}',
                popup=folium.Popup(
                    f"<div style='font-family:Inter,sans-serif;min-width:120px'>"
                    f"<b>{h['valor']:.1%}</b><br>"
                    f"<span style='font-size:11px'>({h['lat']:.4f}, {h['lon']:.4f})</span><br>"
                    f"<span style='font-size:11px;color:#666'>{h.get('zona', '')}</span></div>",
                    max_width=180,
                ),
            ).add_to(m)

        if len(heat_data) >= 2:
            HeatMap(
                heat_data,
                min_opacity=0.3, max_val=1.0,
                radius=30, blur=25,
                gradient={0.2: '#1565c0', 0.4: '#42a5f5', 0.6: '#ffca28', 0.8: '#ff6d00', 1.0: '#d32f2f'},
            ).add_to(m)

    legend = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;'
        'background:#fff;padding:12px 16px;border-radius:10px;'
        'box-shadow:0 2px 12px rgba(0,0,0,.15);font-size:12px;'
        'font-family:Inter,sans-serif;line-height:1.8;">'
        '<b style="font-size:13px">Leyenda</b><br>'
        '<span style="color:red;font-size:16px">&#9679;</span> Potencial Alto (conocido)<br>'
        '<span style="color:orange;font-size:16px">&#9679;</span> Potencial Medio (conocido)<br>'
        '<span style="color:#4caf50;font-size:14px">&#9679;</span> Prediccion positiva<br>'
        '<span style="color:#42a5f5;font-size:14px">&#9679;</span> Prediccion negativa<br>'
        '🔥 Mapa de calor (intensidad de prediccion)</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    return m


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
        st.dataframe(df_zonas, width="stretch", hide_index=True)

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
    zonas = zonas_geotermicas()

    if usa_cnn:
        st.markdown(
            '<div class="info-box">'
            'Selecciona una ubicacion en Colombia haciendo <b>clic en el mapa</b>, '
            'ingresando coordenadas manualmente, o eligiendo una zona conocida. '
            'El sistema descarga datos ASTER de NASA y los analiza con el modelo CNN. '
            '<b>Tiempo estimado: 10-15 segundos.</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box">'
            'Modelo CNN no disponible. Se usara estimacion por proximidad a zonas '
            'geotermicas conocidas (menos preciso). Entrena el modelo para obtener '
            'predicciones basadas en datos ASTER reales.</div>',
            unsafe_allow_html=True,
        )

    # =====================================================================
    # SECCION DE ENTRADA — Seleccion de ubicacion
    # =====================================================================
    st.markdown("#### Selecciona una ubicacion")
    metodo = st.radio(
        "Metodo de entrada:",
        ["🗺️ Clic en mapa", "📝 Coordenadas", "📍 Zona conocida"],
        horizontal=True, label_visibility="collapsed",
    )

    # Inicializar session state para coordenadas
    if "pred_lat" not in st.session_state:
        st.session_state["pred_lat"] = 4.8951
        st.session_state["pred_lon"] = -75.3222

    analizar = False
    latitud = st.session_state["pred_lat"]
    longitud = st.session_state["pred_lon"]

    if metodo == "🗺️ Clic en mapa":
        st.caption("Haz clic en cualquier punto del mapa para seleccionar coordenadas.")

        # Crear mapa selector interactivo
        mapa_sel = folium.Map(
            location=[4.57, -74.30], zoom_start=6,
            tiles="CartoDB positron",
        )

        # Agregar zonas geotermicas como referencia
        colores_z = {"Alto": "red", "Medio": "orange"}
        for z in zonas:
            c = colores_z.get(z["potencial"], "blue")
            folium.CircleMarker(
                [z["lat"], z["lon"]], radius=8,
                popup=f'{z["nombre"]} ({z["potencial"]})',
                tooltip=z["nombre"],
                color=c, fill=True, fillColor=c, fillOpacity=0.6, weight=1.5,
            ).add_to(mapa_sel)

        # Marcador de seleccion actual
        folium.Marker(
            [st.session_state["pred_lat"], st.session_state["pred_lon"]],
            tooltip=f'Seleccion: {st.session_state["pred_lat"]:.4f}, {st.session_state["pred_lon"]:.4f}',
            icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        ).add_to(mapa_sel)

        # Leyenda compacta
        leyenda_sel = (
            '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;'
            'background:#fff;padding:10px 14px;border-radius:8px;'
            'box-shadow:0 2px 8px rgba(0,0,0,.12);font-size:11px;'
            'font-family:Inter,sans-serif;line-height:1.7;">'
            '<b>Leyenda</b><br>'
            '<span style="color:red;font-size:14px">&#9679;</span> Potencial Alto<br>'
            '<span style="color:orange;font-size:14px">&#9679;</span> Potencial Medio<br>'
            '<span style="color:#1565c0;font-size:12px">📍</span> Tu seleccion</div>'
        )
        mapa_sel.get_root().html.add_child(folium.Element(leyenda_sel))

        map_data = st_folium(
            mapa_sel, height=380, width=None,
            returned_objects=["last_clicked"],
        )

        # Actualizar coordenadas si hubo clic
        if map_data and map_data.get("last_clicked"):
            st.session_state["pred_lat"] = round(map_data["last_clicked"]["lat"], 4)
            st.session_state["pred_lon"] = round(map_data["last_clicked"]["lng"], 4)

        latitud = st.session_state["pred_lat"]
        longitud = st.session_state["pred_lon"]

        # Coordenadas seleccionadas + boton
        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            st.markdown(
                f'<div style="background:#f0f4f8;border-radius:8px;padding:10px 14px;text-align:center;">'
                f'<div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Latitud</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#1a1a2e;">{latitud:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                f'<div style="background:#f0f4f8;border-radius:8px;padding:10px 14px;text-align:center;">'
                f'<div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Longitud</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#1a1a2e;">{longitud:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with sc3:
            st.write("")
            analizar = st.button(
                "🔬 Analizar Potencial", type="primary", width="stretch",
                key="btn_mapa",
            )

    elif metodo == "📝 Coordenadas":
        mc1, mc2, mc3 = st.columns([1, 1, 1])
        with mc1:
            latitud = st.number_input(
                "Latitud:", -4.0, 12.0,
                st.session_state["pred_lat"], 0.0001, format="%.4f",
            )
        with mc2:
            longitud = st.number_input(
                "Longitud:", -82.0, -66.0,
                st.session_state["pred_lon"], 0.0001, format="%.4f",
            )
        with mc3:
            st.write("")
            st.write("")
            analizar = st.button(
                "🔬 Analizar Potencial", type="primary", width="stretch",
                key="btn_manual",
            )
        st.session_state["pred_lat"] = latitud
        st.session_state["pred_lon"] = longitud

    else:  # Zona conocida
        zc1, zc2 = st.columns([2, 1])
        with zc1:
            sel = st.selectbox("Selecciona una zona:", [z["nombre"] for z in zonas])
            z = next(x for x in zonas if x["nombre"] == sel)
            latitud, longitud = z["lat"], z["lon"]
            st.markdown(
                f'<div style="background:#f0f4f8;border-radius:8px;padding:10px 14px;'
                f'border-left:3px solid #e94560;">'
                f'<b>{z["nombre"]}</b> · {z["tipo"]} · '
                f'<span style="color:{"#d32f2f" if z["potencial"]=="Alto" else "#ff8f00"};'
                f'font-weight:600;">Potencial {z["potencial"]}</span><br>'
                f'<span style="font-size:0.85rem;color:#666;">'
                f'Lat: {latitud:.4f} · Lon: {longitud:.4f}</span></div>',
                unsafe_allow_html=True,
            )
        with zc2:
            st.write("")
            st.write("")
            analizar = st.button(
                "🔬 Analizar Potencial", type="primary", width="stretch",
                key="btn_zona",
            )
        st.session_state["pred_lat"] = latitud
        st.session_state["pred_lon"] = longitud

    # =====================================================================
    # LOGICA DE PREDICCION
    # =====================================================================
    if analizar:
        if usa_cnn:
            with st.spinner("Descargando imagen ASTER y analizando con CNN..."):
                resultado_cnn = predecir_con_modelo_cnn(latitud, longitud, modelo)

            if resultado_cnn["ok"]:
                _, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
                todas_dist = []
                for z in zonas:
                    d_km = _haversine_km(latitud, longitud, z["lat"], z["lon"])
                    todas_dist.append({
                        "Zona": z["nombre"], "Tipo": z["tipo"],
                        "Potencial": z["potencial"],
                        "Distancia (km)": round(d_km, 1),
                    })
                todas_dist.sort(key=lambda x: x["Distancia (km)"])

                pred_data = {
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
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state["pred"] = pred_data

                # Guardar en historial
                if "pred_historial" not in st.session_state:
                    st.session_state["pred_historial"] = []
                st.session_state["pred_historial"].insert(0, pred_data)
                if len(st.session_state["pred_historial"]) > 10:
                    st.session_state["pred_historial"] = st.session_state["pred_historial"][:10]
            else:
                pred, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
                st.session_state["pred"] = {
                    "lat": latitud, "lon": longitud, "valor": pred,
                    "zona": zona_c["nombre"], "tipo": zona_c["tipo"], "dist": dist,
                    "metodo": "proximidad (CNN fallo)",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
        else:
            pred, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
            st.session_state["pred"] = {
                "lat": latitud, "lon": longitud, "valor": pred,
                "zona": zona_c["nombre"], "tipo": zona_c["tipo"], "dist": dist,
                "metodo": "proximidad",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }

    # =====================================================================
    # SECCION DE RESULTADOS — Full width
    # =====================================================================
    if "pred" in st.session_state:
        st.divider()
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

        # Clasificacion de confianza
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

        # ---- Tarjeta principal de resultado ----
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

        # ---- Barra de probabilidad ----
        bar_color = "#2e7d32" if pos else "#d84315"
        bar_end = "#43a047" if pos else "#ef6c00"
        st.markdown(
            f'<div style="margin:14px 0 4px 0;font-size:0.8rem;color:#666;">'
            f'Probabilidad geotermica</div>'
            f'<div style="background:#e0e0e0;border-radius:8px;height:16px;overflow:hidden;">'
            f'<div style="width:{valor*100:.1f}%;height:100%;'
            f'background:linear-gradient(90deg,{bar_color},{bar_end});'
            f'border-radius:8px;transition:width 0.5s;"></div></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#999;margin-top:2px;">'
            f'<span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>',
            unsafe_allow_html=True,
        )

        st.write("")

        # ---- Metricas clave ----
        dist_km = p["dist"] * 111.0
        mk1, mk2, mk3, mk4, mk5 = st.columns(5)
        mk1.metric("Latitud", f'{p["lat"]:.4f}')
        mk2.metric("Longitud", f'{p["lon"]:.4f}')
        mk3.metric("Zona mas cercana", p["zona"])
        mk4.metric("Distancia", f'{dist_km:.1f} km')
        mk5.metric("Confianza", nivel)

        # ---- Interpretacion ----
        st.markdown(
            f'<div style="background:#f8f9fa;border-left:4px solid {nivel_color};'
            f'border-radius:0 8px 8px 0;padding:14px 18px;margin:14px 0;">'
            f'<div style="font-weight:600;color:{nivel_color};margin-bottom:4px;font-size:0.95rem;">'
            f'{nivel_desc}</div>'
            f'<div style="font-size:0.88rem;color:#555;line-height:1.6;">'
            f'La coordenada analizada (<b>{p["lat"]:.4f}, {p["lon"]:.4f}</b>) se encuentra '
            f'a <b>{dist_km:.1f} km</b> de la zona geotermica mas cercana '
            f'(<b>{p["zona"]}</b> · {p["tipo"]}). '
            f'{"El modelo CNN detecta patrones termicos consistentes con actividad geotermica en las bandas de emisividad ASTER." if pos else "El modelo CNN no detecta patrones termicos significativos de actividad geotermica en esta ubicacion."}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ---- Mapa de resultado + Datos tecnicos lado a lado ----
        st.write("")
        col_mapa, col_tech = st.columns([3, 2], gap="medium")

        with col_mapa:
            st.markdown("##### Ubicacion en el mapa")
            mapa = crear_mapa(zonas, {"lat": p["lat"], "lon": p["lon"]}, p["valor"])
            st_folium(mapa, width=None, height=400, returned_objects=[])

        with col_tech:
            if metodo_txt == "CNN" and p.get("band_stats"):
                st.markdown("##### Datos del analisis")
                # Imagen ASTER
                img_shape = p.get("img_shape", (0, 0, 0))
                st.markdown(
                    f'<div style="background:#f0f4f8;border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
                    f'<div style="font-weight:600;color:#1a1a2e;margin-bottom:6px;font-size:0.9rem;">'
                    f'🛰️ Imagen ASTER</div>'
                    f'<table style="width:100%;font-size:0.82rem;color:#444;">'
                    f'<tr><td style="padding:2px 0;"><b>Dataset</b></td>'
                    f'<td style="text-align:right;">ASTER GED v003</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Fuente</b></td>'
                    f'<td style="text-align:right;">NASA/USGS via GEE</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Bandas</b></td>'
                    f'<td style="text-align:right;">{p.get("n_bands", 7)} (5 TIR + temp + NDVI)</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Resolucion</b></td>'
                    f'<td style="text-align:right;">{p.get("scale_m", 90)} m/pixel</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Area</b></td>'
                    f'<td style="text-align:right;">{p.get("buffer_m", 5000)/1000:.0f} km radio</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Imagen</b></td>'
                    f'<td style="text-align:right;">{img_shape[0]}x{img_shape[1]} → 224x224</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Tamano</b></td>'
                    f'<td style="text-align:right;">{p.get("img_size_kb", 0):.1f} KB</td></tr>'
                    f'</table></div>',
                    unsafe_allow_html=True,
                )
                # Modelo CNN
                st.markdown(
                    '<div style="background:#f0f4f8;border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
                    '<div style="font-weight:600;color:#1a1a2e;margin-bottom:6px;font-size:0.9rem;">'
                    '⚙️ Modelo CNN</div>'
                    '<table style="width:100%;font-size:0.82rem;color:#444;">'
                    '<tr><td style="padding:2px 0;"><b>Arquitectura</b></td>'
                    '<td style="text-align:right;">CNN personalizada</td></tr>'
                    '<tr><td style="padding:2px 0;"><b>Dataset</b></td>'
                    '<td style="text-align:right;">6,200 imagenes</td></tr>'
                    '<tr><td style="padding:2px 0;"><b>Mejor epoca</b></td>'
                    '<td style="text-align:right;">8 / 22</td></tr>'
                    '<tr><td style="padding:2px 0;"><b>Accuracy</b></td>'
                    '<td style="text-align:right;">91.45%</td></tr>'
                    '<tr><td style="padding:2px 0;"><b>Precision</b></td>'
                    '<td style="text-align:right;">97.94%</td></tr>'
                    '<tr><td style="padding:2px 0;"><b>ROC AUC</b></td>'
                    '<td style="text-align:right;">0.983</td></tr>'
                    '<tr><td style="padding:2px 0;"><b>F1-Score</b></td>'
                    '<td style="text-align:right;">91.61%</td></tr>'
                    '</table></div>',
                    unsafe_allow_html=True,
                )
                # Tiempos
                st.markdown(
                    f'<div style="background:#f0f4f8;border-radius:10px;padding:12px 14px;">'
                    f'<div style="font-weight:600;color:#1a1a2e;margin-bottom:6px;font-size:0.9rem;">'
                    f'⏱️ Tiempos</div>'
                    f'<table style="width:100%;font-size:0.82rem;color:#444;">'
                    f'<tr><td style="padding:2px 0;"><b>Descarga GEE</b></td>'
                    f'<td style="text-align:right;">{p.get("t_download", 0):.1f}s</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Prediccion CNN</b></td>'
                    f'<td style="text-align:right;">{p.get("t_pred", 0):.3f}s</td></tr>'
                    f'<tr><td style="padding:2px 0;"><b>Total</b></td>'
                    f'<td style="text-align:right;">{p.get("t_total", 0):.1f}s</td></tr>'
                    f'</table></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("##### Informacion")
                st.markdown(
                    '<div style="background:#f0f4f8;border-radius:10px;padding:14px 16px;">'
                    '<div style="font-size:0.88rem;color:#555;line-height:1.6;">'
                    'Los datos tecnicos detallados (estadisticas de bandas, '
                    'tiempos de procesamiento, informacion del modelo) se muestran '
                    'cuando la prediccion se realiza con el modelo CNN.</div></div>',
                    unsafe_allow_html=True,
                )

        # ---- Expanders con datos adicionales (solo CNN) ----
        if metodo_txt == "CNN" and p.get("band_stats"):
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
            with st.expander("📊 Estadisticas de bandas termicas ASTER"):
                st.markdown(
                    '<div style="font-size:0.85rem;color:#666;margin-bottom:8px;">'
                    'Valores de emisividad termica por banda antes de la normalizacion. '
                    'Estas bandas TIR (Thermal Infrared) capturan la radiacion termica '
                    'emitida por la superficie terrestre.</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(band_data).set_index("Banda"),
                    width="stretch",
                )

            if p.get("todas_dist"):
                with st.expander("📍 Distancia a zonas geotermicas conocidas"):
                    st.markdown(
                        '<div style="font-size:0.85rem;color:#666;margin-bottom:8px;">'
                        'Distancia desde la coordenada analizada a las 10 zonas '
                        'geotermicas conocidas en Colombia.</div>',
                        unsafe_allow_html=True,
                    )
                    df_zonas = pd.DataFrame(p["todas_dist"])
                    st.dataframe(df_zonas.set_index("Zona"), width="stretch")

        # ---- Historial de predicciones ----
        if st.session_state.get("pred_historial") and len(st.session_state["pred_historial"]) > 1:
            with st.expander(f"🕐 Historial de predicciones ({len(st.session_state['pred_historial'])})"):
                hist_data = []
                for h in st.session_state["pred_historial"]:
                    hist_data.append({
                        "Hora": h.get("timestamp", "-"),
                        "Latitud": f'{h["lat"]:.4f}',
                        "Longitud": f'{h["lon"]:.4f}',
                        "Probabilidad": f'{h["valor"]:.1%}',
                        "Resultado": "Con potencial" if h["valor"] >= 0.5 else "Bajo potencial",
                        "Zona cercana": h.get("zona", "-"),
                    })
                st.dataframe(pd.DataFrame(hist_data), width="stretch", hide_index=True)

        # ---- Exportar reporte ----
        st.write("")
        exp1, exp2, _ = st.columns([1, 1, 2])
        with exp1:
            reporte_txt = generar_reporte_texto(p)
            st.download_button(
                "📄 Descargar reporte (.txt)",
                data=reporte_txt,
                file_name=f"reporte_geotermico_{p['lat']:.4f}_{p['lon']:.4f}.txt",
                mime="text/plain",
                width="stretch",
            )
        with exp2:
            if st.session_state.get("pred_historial"):
                csv_rows = []
                for h in st.session_state["pred_historial"]:
                    csv_rows.append({
                        "timestamp": h.get("timestamp", ""),
                        "latitud": h["lat"],
                        "longitud": h["lon"],
                        "probabilidad": round(h["valor"], 4),
                        "resultado": "positivo" if h["valor"] >= 0.5 else "negativo",
                        "confianza": (
                            "alta" if h["valor"] >= 0.8 else
                            "media-alta" if h["valor"] >= 0.6 else
                            "media" if h["valor"] >= 0.5 else
                            "media-baja" if h["valor"] >= 0.35 else
                            "baja" if h["valor"] >= 0.2 else "muy_baja"
                        ),
                        "zona_cercana": h.get("zona", ""),
                        "distancia_km": round(h.get("dist", 0) * 111.0, 1),
                        "metodo": h.get("metodo", ""),
                    })
                csv_df = pd.DataFrame(csv_rows)
                st.download_button(
                    "📊 Descargar historial (.csv)",
                    data=csv_df.to_csv(index=False),
                    file_name="historial_predicciones.csv",
                    mime="text/csv",
                    width="stretch",
                )

        # ---- Mapa de calor (heatmap) ----
        if st.session_state.get("pred_historial") and len(st.session_state["pred_historial"]) >= 2:
            st.write("")
            with st.expander("🔥 Mapa de calor de predicciones realizadas"):
                st.markdown(
                    '<div style="font-size:0.85rem;color:#666;margin-bottom:8px;">'
                    'Mapa de calor generado a partir de todas las predicciones realizadas '
                    'en esta sesion. La intensidad del color representa la probabilidad '
                    'geotermica predicha por el modelo CNN.</div>',
                    unsafe_allow_html=True,
                )
                mapa_heat = crear_mapa_heatmap(zonas, st.session_state["pred_historial"])
                st_folium(mapa_heat, width=None, height=450, returned_objects=[])

        # ---- Modo comparacion ----
        st.write("")
        with st.expander("⚖️ Comparar dos ubicaciones"):
            st.markdown(
                '<div style="font-size:0.85rem;color:#666;margin-bottom:12px;">'
                'Ingresa dos pares de coordenadas para comparar las predicciones '
                'del modelo CNN lado a lado.</div>',
                unsafe_allow_html=True,
            )
            cmp_c1, cmp_c2 = st.columns(2, gap="medium")

            with cmp_c1:
                st.markdown("**Ubicacion A**")
                cmp_lat_a = st.number_input("Latitud A:", -4.0, 12.0, 4.8951, 0.0001, format="%.4f", key="cmp_lat_a")
                cmp_lon_a = st.number_input("Longitud A:", -82.0, -66.0, -75.3222, 0.0001, format="%.4f", key="cmp_lon_a")

            with cmp_c2:
                st.markdown("**Ubicacion B**")
                cmp_lat_b = st.number_input("Latitud B:", -4.0, 12.0, 5.7781, 0.0001, format="%.4f", key="cmp_lat_b")
                cmp_lon_b = st.number_input("Longitud B:", -82.0, -66.0, -73.1124, 0.0001, format="%.4f", key="cmp_lon_b")

            cmp_btn = st.button("🔬 Comparar ambas ubicaciones", type="primary", width="stretch", key="btn_cmp")

            if cmp_btn and usa_cnn and modelo is not None:
                cmp_r1, cmp_r2 = st.columns(2, gap="medium")

                with st.spinner("Analizando ubicacion A..."):
                    res_a = predecir_con_modelo_cnn(cmp_lat_a, cmp_lon_a, modelo)
                with st.spinner("Analizando ubicacion B..."):
                    res_b = predecir_con_modelo_cnn(cmp_lat_b, cmp_lon_b, modelo)

                for col_cmp, res, lat, lon, label in [
                    (cmp_r1, res_a, cmp_lat_a, cmp_lon_a, "A"),
                    (cmp_r2, res_b, cmp_lat_b, cmp_lon_b, "B"),
                ]:
                    with col_cmp:
                        if res["ok"]:
                            v = res["prob"]
                            es_pos = v >= 0.5
                            _, zc, dd = predecir_por_proximidad(lat, lon, zonas)
                            cls_c = "result-pos" if es_pos else "result-neg"
                            st.markdown(
                                f'<div class="result-card {cls_c}" style="padding:1.2rem 1rem;">'
                                f'<h2 style="font-size:0.85rem;">UBICACION {label}</h2>'
                                f'<div class="big" style="font-size:2.2rem;">{"🌋" if es_pos else "🏔️"} {v:.1%}</div>'
                                f'<p style="font-size:0.8rem;">{lat:.4f}, {lon:.4f}</p>'
                                f'<p style="font-size:0.75rem;margin-top:4px;">Zona: {zc["nombre"]} · {dd*111:.1f} km</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Guardar en historial
                            if "pred_historial" not in st.session_state:
                                st.session_state["pred_historial"] = []
                            st.session_state["pred_historial"].insert(0, {
                                "lat": lat, "lon": lon, "valor": v,
                                "zona": zc["nombre"], "tipo": zc["tipo"], "dist": dd,
                                "metodo": "CNN", "timestamp": datetime.now().strftime("%H:%M:%S"),
                            })
                        else:
                            st.error(f"No se pudo analizar la ubicacion {label}")

                if res_a["ok"] and res_b["ok"]:
                    diff = abs(res_a["prob"] - res_b["prob"])
                    mejor = "A" if res_a["prob"] > res_b["prob"] else "B"
                    st.markdown(
                        f'<div style="background:#f8f9fa;border-radius:10px;padding:14px;'
                        f'text-align:center;margin-top:12px;">'
                        f'<span style="font-size:0.9rem;color:#555;">'
                        f'Diferencia: <b>{diff:.1%}</b> · '
                        f'Mayor potencial: <b>Ubicacion {mejor}</b></span></div>',
                        unsafe_allow_html=True,
                    )

            elif cmp_btn and not usa_cnn:
                st.warning("El modo comparacion requiere el modelo CNN cargado.")

    else:
        # Estado vacio — mostrar mapa de zonas como referencia
        st.divider()
        st.markdown(
            '<div style="text-align:center;padding:1.5rem 0;color:#888">'
            '<div style="font-size:2.5rem;margin-bottom:0.8rem">🗺️</div>'
            '<p style="font-size:1.05rem;margin:0">Selecciona una ubicacion y presiona '
            '<b>Analizar Potencial</b> para obtener la prediccion</p></div>',
            unsafe_allow_html=True,
        )
        st_folium(crear_mapa(zonas), width=None, height=380, returned_objects=[])


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
            st.plotly_chart(fig, width="stretch", key="bar_metricas")

        with c2:
            st.markdown("#### Curva ROC")
            auc = metricas.get("auc_roc", metricas.get("roc_auc", metricas.get("auc", 0.5)))
            # v2: Usar curva ROC real del evaluate_model si está disponible (BUG 14)
            roc_data = metricas.get("roc_curve")
            if roc_data and "fpr" in roc_data and "tpr" in roc_data:
                fpr = np.array(roc_data["fpr"])
                tpr = np.array(roc_data["tpr"])
            else:
                # Fallback: aproximación matemática
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
            st.plotly_chart(fig, width="stretch", key="roc")

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
            st.plotly_chart(fig, width="stretch", key="loss")

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
            st.plotly_chart(fig, width="stretch", key="acc")

    # Figuras PNG
    figs_dir = PROJECT_ROOT / "results" / "figures"
    imgs = sorted(figs_dir.glob("*.png")) if figs_dir.is_dir() else []
    if imgs:
        st.divider()
        st.markdown("#### Visualizaciones Generadas")
        cols = st.columns(min(len(imgs), 3), gap="medium")
        for i, img in enumerate(imgs[:6]):
            cols[i % len(cols)].image(str(img), caption=img.stem.replace("_", " ").title(), width="stretch")


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
        'Disenada para clasificacion binaria de imagenes ASTER de 7 bandas (5 TIR + temperatura + NDVI).'
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
            "224x224x7", "224x224x32", "224x224x32", "112x112x32",
            "112x112x64", "56x56x64", "56x56x128", "28x28x128",
            "28x28x256", "14x14x256", "14x14x512", "14x14x512",
            "512", "256", "1",
        ],
    })
    st.dataframe(capas, width="stretch", hide_index=True)

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
    st.plotly_chart(fig, width="stretch", key="arch")

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
| Bandas | 10, 11, 12, 13, 14 (TIR) + temperature + NDVI |
| Entrada | 224 x 224 x 7 |
| Normalizacion | Z-score por banda |
| Dataset | ~6,200 imagenes (augmentadas) |
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
            "Google Earth Engine -> 200 imagenes ASTER (7 bandas)",
            "31 transformaciones -> ~6,200 imagenes",
            "Normalizacion + split 70/15/15 estratificado",
            "CNN ResNet-inspired (5 M parametros)",
            "Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC",
            "Streamlit + Folium + Plotly",
        ],
    })
    st.dataframe(met_data, width="stretch", hide_index=True)

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
