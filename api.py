"""
API REST para el frontend React de Geotermia CNN Colombia.
Integra el modelo CNN real (EfficientNetB0 + Channel Adapter) con
fallback a predicción por proximidad. Cumple OWASP Top 10 2025.
"""

import gc
import os
import sys
import json
import logging
import tempfile
import time
import concurrent.futures
from pathlib import Path

# Thread pool persistente para inferencia (máx 1 worker para proteger los 512MB RAM de Render)
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

import numpy as np
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
import re

PROJECT_ROOT = Path(__file__).resolve().parent
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# Patrón para aceptar previews de Vercel (subdominios dinámicos)
_VERCEL_RE = re.compile(r"^https://[a-z0-9\-]+\.vercel\.app$")

# Límites geográficos de Colombia (con margen)
LAT_MIN, LAT_MAX = -5.0, 14.0
LON_MIN, LON_MAX = -82.0, -66.0

app = Flask(__name__)


def _is_allowed_origin(origin: str) -> bool:
    if not origin:
        return False
    if origin in ALLOWED_ORIGINS:
        return True
    if origin.startswith("https://") and origin.endswith(".vercel.app"):
        return True
    return bool(_VERCEL_RE.match(origin))


@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.before_request
def _handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "")
        if _is_allowed_origin(origin):
            resp = app.make_default_options_response()
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return resp


@app.errorhandler(Exception)
def _handle_exception(exc):
    if isinstance(exc, HTTPException):
        code = exc.code
        desc = exc.description
    else:
        logging.exception("Error interno no capturado: %s", exc)
        code = 500
        desc = "Error interno del servidor"

    origin = request.headers.get("Origin", "")
    resp = jsonify({"error": desc, "status": code})
    resp.status_code = code
    if _is_allowed_origin(origin):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# Cache en memoria de predicciones recientes para evitar re-computo y OOM
_PRED_CACHE = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Rate Limiting sencillo (en memoria, sin dependencias externas)
# ---------------------------------------------------------------------------
_rate_store: dict[str, list[float]] = {}
RATE_LIMIT = 30          # máx peticiones
RATE_WINDOW = 60          # por ventana de 60 seg

def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    hits = _rate_store.get(ip, [])
    hits = [t for t in hits if now - t < RATE_WINDOW]
    if len(hits) >= RATE_LIMIT:
        _rate_store[ip] = hits
        return False
    hits.append(now)
    _rate_store[ip] = hits
    return True


# ---------------------------------------------------------------------------
# Security Headers (OWASP A05 – Security Misconfiguration)
# ---------------------------------------------------------------------------
@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["X-XSS-Protection"] = "0"  # Desactivado; CSP lo reemplaza
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# TensorFlow + Modelo CNN
# ---------------------------------------------------------------------------
_tf = None
_modelo = None
_modelo_nombre = None
_ee_initialized = False


def _importar_tensorflow():
    """Import TF con monkey-patch para compatibilidad Keras 3.12."""
    global _tf
    if _tf is not None:
        return _tf

    # Reducir uso de memoria de TF en entornos limitados (Render free = 512MB)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import tensorflow as tf_mod
    _tf = tf_mod

    _orig_dense_init = tf_mod.keras.layers.Dense.__init__
    def _dense_compat_init(self, *args, quantization_config=None, **kwargs):
        _orig_dense_init(self, *args, **kwargs)
    tf_mod.keras.layers.Dense.__init__ = _dense_compat_init

    tf_mod.keras.mixed_precision.set_global_policy("mixed_float16")
    return _tf


def _cargar_modelo():
    """Carga el modelo CNN una sola vez."""
    global _modelo, _modelo_nombre
    if _modelo is not None:
        return _modelo

    tf = _importar_tensorflow()
    rutas = [
        PROJECT_ROOT / "models" / "saved_models" / "geotermia_v7_phase2_best.keras",
        PROJECT_ROOT / "models" / "saved_models" / "geotermia_v7_final.keras",
        PROJECT_ROOT / "models" / "saved_models" / "geotermia_cnn_custom_final.keras",
        PROJECT_ROOT / "models" / "saved_models" / "mini_model_best.keras",
    ]
    for r in rutas:
        if r.exists():
            try:
                _modelo = tf.keras.models.load_model(str(r), compile=False, safe_mode=False)
                _modelo_nombre = r.name
                logging.info("Modelo CNN cargado: %s", r.name)
                return _modelo
            except Exception as exc:
                logging.warning("No se pudo cargar %s: %s", r.name, exc)
    logging.warning("Ningún modelo CNN disponible — se usará fallback por proximidad.")
    return None


def _inicializar_ee():
    """Inicializa Google Earth Engine una sola vez.
    
    En local: usa credenciales ADC del usuario (ee.Authenticate previo).
    En producción: usa Service Account vía la variable GEE_SERVICE_ACCOUNT_KEY
    (JSON de la service account como string).
    """
    global _ee_initialized
    if _ee_initialized:
        return True
    try:
        import ee
        project = os.environ.get("GEE_PROJECT", "alpine-air-469115-f0")

        sa_key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
        if sa_key:
            import json as _json
            key_data = _json.loads(sa_key)
            credentials = ee.ServiceAccountCredentials(
                key_data["client_email"], key_data=sa_key
            )
            ee.Initialize(credentials=credentials, project=project)
        elif os.environ.get("RENDER") or os.environ.get("FLASK_ENV") == "production":
            logging.warning("GEE_SERVICE_ACCOUNT_KEY no configurada en Render. Se usará fallback por proximidad.")
            return False
        else:
            ee.Initialize(project=project)

        _ee_initialized = True
        logging.info("Google Earth Engine inicializado (proyecto: %s)", project)
        return True
    except Exception as exc:
        logging.error("GEE init falló: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Predicción CNN real
# ---------------------------------------------------------------------------
def predecir_cnn(lat: float, lon: float, modelo):
    """Descarga ASTER desde GEE y pasa la imagen por el modelo CNN."""
    try:
        import ee
        import geemap
        import rasterio
    except ImportError:
        return {"prob": 0.0, "ok": False, "error": "Dependencias GEE no disponibles"}

    if not _inicializar_ee():
        return {"prob": 0.0, "ok": False, "error": "GEE no inicializado"}

    from config import cfg

    t0 = time.time()
    try:
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(5000)
    except Exception as exc:
        logging.error("Error creando geometría GEE: %s", exc)
        return {"prob": 0.0, "ok": False, "error": f"GEE geometry error: {exc}"}

    aster_bands = [
        "emissivity_band10", "emissivity_band11",
        "emissivity_band12", "emissivity_band13",
        "emissivity_band14", "temperature", "ndvi",
    ]
    image = ee.Image("NASA/ASTER_GED/AG100_003").select(aster_bands).clip(roi)

    tmp_path = Path(tempfile.gettempdir()) / f"pred_{lat:.4f}_{lon:.4f}_{int(time.time()*1000)}.tif"
    try:
        geemap.ee_export_image(image, filename=str(tmp_path), scale=90, region=roi, file_per_band=False, timeout=25)
    except Exception as exc:
        logging.error("Error descargando imagen ASTER: %s", exc)
        return {"prob": 0.0, "ok": False, "error": f"ASTER download error: {exc}"}

    if not tmp_path.exists():
        return {"prob": 0.0, "ok": False, "error": "No se descargó imagen ASTER"}

    t_download = time.time() - t0

    with rasterio.open(str(tmp_path)) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        crs = str(src.crs) if src.crs else "N/A"
        img_shape_orig = list(img.shape)

    file_size_kb = round(tmp_path.stat().st_size / 1024, 1)

    n_bands = cfg.NUM_BANDS
    if img.shape[2] < n_bands:
        pad = np.zeros((img.shape[0], img.shape[1], n_bands - img.shape[2]), dtype=np.float32)
        img = np.concatenate([img, pad], axis=2)
    elif img.shape[2] > n_bands:
        img = img[:, :, :n_bands]

    # Recoger estadísticas por banda ANTES de normalizar
    band_names = ["emissivity_band10", "emissivity_band11", "emissivity_band12",
                  "emissivity_band13", "emissivity_band14", "temperature", "ndvi"]
    band_info = []
    for i in range(n_bands):
        b = img[:, :, i]
        valid = b[b > -9999]
        if len(valid) > 0:
            band_info.append({
                "nombre": band_names[i] if i < len(band_names) else f"band_{i}",
                "min": round(float(np.min(valid)), 4),
                "max": round(float(np.max(valid)), 4),
                "mean": round(float(np.mean(valid)), 4),
                "std": round(float(np.std(valid)), 4),
                "nodata_pct": round(float((b <= -9999).mean() * 100), 1),
            })
        else:
            band_info.append({"nombre": band_names[i] if i < len(band_names) else f"band_{i}",
                              "min": 0, "max": 0, "mean": 0, "std": 0, "nodata_pct": 100.0})

    # Reemplazar NoData (-9999) con mediana por banda
    for i in range(n_bands):
        band = img[:, :, i]
        valid = band[band > -9999]
        if len(valid) > 0:
            band[band <= -9999] = np.median(valid)
        else:
            band[band <= -9999] = 0

    # Normalizar con estadísticas globales del dataset
    stats_path = PROJECT_ROOT / "data" / "processed" / "band_stats_v3.json"
    if not stats_path.exists():
        stats_path = PROJECT_ROOT / "data" / "processed" / "band_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        means = np.array(stats["band_means"], dtype=np.float32)
        stds = np.array(stats["band_stds"], dtype=np.float32)
        for i in range(n_bands):
            img[:, :, i] = (img[:, :, i] - means[i]) / stds[i]
    else:
        for i in range(n_bands):
            m, s = img[:, :, i].mean(), img[:, :, i].std()
            img[:, :, i] = (img[:, :, i] - m) / (s if s > 0 else 1)

    # Inferencia optimizada de parche central (parche centrado de 224x224 con llamada directa al tensor)
    # Evita fugas de memoria de Keras .predict() y reduce consumo a <15MB RAM
    t_pred = time.time()
    h, w, c = img.shape
    hw, ww = 224, 224

    if h >= hw and w >= ww:
        sy = (h - hw) // 2
        sx = (w - ww) // 2
        patch = img[sy:sy+hw, sx:sx+ww, :]
    else:
        patch = np.zeros((hw, ww, c), dtype=np.float32)
        patch[:min(h, hw), :min(w, ww), :] = img[:min(h, hw), :min(w, ww), :]
    del img

    tf = _importar_tensorflow()
    input_tensor = tf.constant(np.expand_dims(patch, 0))
    del patch
    pred_tensor = modelo(input_tensor, training=False)
    prob = float(pred_tensor[0, 0])
    del input_tensor, pred_tensor
    gc.collect()

    t_pred = time.time() - t_pred

    try:
        tmp_path.unlink()
    except Exception:
        pass
    gc.collect()

    return {
        "prob": float(np.clip(prob, 0.01, 0.99)),
        "ok": True,
        "t_download": round(t_download, 2),
        "t_pred": round(t_pred, 2),
        "t_total": round(time.time() - t0, 2),
        "satelite": {
            "dataset": "NASA/ASTER_GED/AG100_003",
            "sensor": "ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer)",
            "resolucion_m": 90,
            "buffer_m": 5000,
            "crs": crs,
            "imagen_shape": img_shape_orig,
            "archivo_kb": file_size_kb,
        },
        "bandas": band_info,
    }


# ---------------------------------------------------------------------------
# Zonas geotérmicas y fallback por proximidad
# ---------------------------------------------------------------------------
ZONAS = [
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


def predecir_por_proximidad(lat: float, lon: float):
    """Fallback: sigmoide invertida sobre distancia a zona más cercana (Haversine aprox)."""
    dists = [np.sqrt((lat - z["lat"])**2 + (lon - z["lon"])**2) for z in ZONAS]
    idx = int(np.argmin(dists))
    d = dists[idx]
    z = ZONAS[idx]
    pred = float(1.0 / (1.0 + np.exp(8.0 * (d - 0.5))))
    if z["potencial"] == "Alto" and d < 0.3:
        pred = min(pred * 1.1, 0.99)
    return float(np.clip(pred, 0.01, 0.99)), z, d


# ---------------------------------------------------------------------------
# Validación de entrada (OWASP A03 – Injection)
# ---------------------------------------------------------------------------
def _validar_coordenadas(data):
    """Valida y extrae lat/lon del JSON de entrada."""
    if not isinstance(data, dict):
        return None, None, "Cuerpo JSON inválido"
    try:
        lat = float(data.get("lat", ""))
        lon = float(data.get("lon", ""))
    except (TypeError, ValueError):
        return None, None, "lat/lon deben ser números"
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None, None, f"Coordenadas fuera de Colombia ({LAT_MIN}-{LAT_MAX}°N, {LON_MIN}-{LON_MAX}°W)"
    return lat, lon, None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "status": "ok",
            "mensaje": "Endpoint /predict activo. Para realizar una prediccion envia un POST con {'lat': float, 'lon': float}.",
            "ejemplo": {"lat": 4.8951, "lon": -75.3222}
        })
    # Rate limiting
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if not _check_rate_limit(client_ip):
        return jsonify({"error": "Demasiadas solicitudes. Intenta en un minuto."}), 429

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Se requiere cuerpo JSON"}), 400

    lat, lon, err = _validar_coordenadas(data)
    if err:
        return jsonify({"error": err}), 400

    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in _PRED_CACHE:
        logging.info("Retornando prediccion desde cache para coordenadas: %s", cache_key)
        return jsonify(_PRED_CACHE[cache_key])

    # Intentar predicción CNN primero con timeout de 35 segundos
    modelo = _cargar_modelo()
    if modelo is not None:
        try:
            future = _EXECUTOR.submit(predecir_cnn, lat, lon, modelo)
            resultado = future.result(timeout=35.0)
        except concurrent.futures.TimeoutError:
            logging.warning("CNN/GEE excedio timeout de 35s en Render. Usando fallback por proximidad.")
            resultado = {"ok": False, "error": "Timeout en descarga de satelite (35s)"}
        except Exception as exc:
            logging.warning("Error ejecutando CNN: %s", exc)
            resultado = {"ok": False, "error": str(exc)}
        gc.collect()  # Liberar RAM después de CNN
        if resultado["ok"]:
            pred, zona, dist = predecir_por_proximidad(lat, lon)
            out_data = {
                "porcentaje": round(resultado["prob"] * 100, 1),
                "zona_cercana": zona["nombre"],
                "distancia_km": round(dist * 111, 2),
                "metodo": "cnn",
                "modelo": _modelo_nombre,
                "tiempos": {
                    "descarga": resultado.get("t_download"),
                    "prediccion": resultado.get("t_pred"),
                    "total": resultado.get("t_total"),
                },
                "satelite": resultado.get("satelite"),
                "bandas": resultado.get("bandas"),
            }
            _PRED_CACHE[cache_key] = out_data
            if len(_PRED_CACHE) > 50:
                _PRED_CACHE.pop(next(iter(_PRED_CACHE)))
            return jsonify(out_data)
        logging.warning("CNN falló, usando fallback: %s", resultado.get("error", "desconocido"))

    # Fallback por proximidad
    pred, zona, dist = predecir_por_proximidad(lat, lon)
    return jsonify({
        "porcentaje": round(pred * 100, 1),
        "zona_cercana": zona["nombre"],
        "distancia_km": round(dist * 111, 2),
        "metodo": "proximidad",
        "modelo": _modelo_nombre or "ninguno",
    })


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "servicio": "API Geotermia CNN Colombia",
        "version": "v3",
        "endpoints": {
            "health": "/health",
            "zonas": "/zonas",
            "predict": "/predict (GET o POST)"
        }
    })


@app.route("/zonas", methods=["GET"])
def get_zonas():
    return jsonify(ZONAS)


@app.route("/health", methods=["GET"])
def health():
    modelo = _cargar_modelo()
    info = {
        "status": "ok",
        "modelo_cargado": _modelo_nombre or "ninguno",
        "modelo_disponible": modelo is not None,
    }
    if modelo is not None:
        info["modelo_detalle"] = {
            "nombre": _modelo_nombre,
            "input_shape": list(modelo.input_shape) if hasattr(modelo, "input_shape") else None,
            "output_shape": list(modelo.output_shape) if hasattr(modelo, "output_shape") else None,
            "total_params": int(modelo.count_params()) if hasattr(modelo, "count_params") else None,
        }
    return jsonify(info)


# ---------------------------------------------------------------------------
# Precarga de modelo en producción / Render
# ---------------------------------------------------------------------------
if os.environ.get("RENDER") or os.environ.get("FLASK_ENV") == "production":
    try:
        logging.info("Precargando modelo CNN al iniciar Gunicorn...")
        _cargar_modelo()
    except Exception as _exc:
        logging.warning("No se pudo precargar modelo al arranque: %s", _exc)


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Cargando modelo al iniciar...")
    _cargar_modelo()
    logging.info("Iniciando API — http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)