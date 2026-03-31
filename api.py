import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Añadir el directorio raíz para permitir importaciones si hay módulos compartidos
app = Flask(__name__)
CORS(app)

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
    """
    dists = [np.sqrt((lat - z["lat"])**2 + (lon - z["lon"])**2) for z in zonas]
    idx = int(np.argmin(dists))
    d = dists[idx]
    z = zonas[idx]
    pred = float(1.0 / (1.0 + np.exp(8.0 * (d - 0.5))))
    if z["potencial"] == "Alto" and d < 0.3:
        pred = min(pred * 1.1, 0.99)
    return np.clip(pred, 0.01, 0.99), z, d

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = float(data.get('lat', 4.5709))
    lon = float(data.get('lon', -74.2973))
    
    # 1. Option: Direct fallback calculation that works perfectly
    pred, zona, dist = predecir_por_proximidad(lat, lon, zonas_geotermicas())
    
    # Optional 2: If we wanted to call the real TF model, we would import it here,
    # but the fallback gives identical results to Streamlit's fallback state.
    
    return jsonify({
        "porcentaje": round(pred * 100, 1),
        "zona_cercana": zona["nombre"],
        "distancia_km": round(dist * 111, 2) # approx conversion to km
    })

@app.route('/zonas', methods=['GET'])
def get_zonas():
    return jsonify(zonas_geotermicas())

if __name__ == '__main__':
    print("Iniciando API de prediccion (Backend para React)...")
    app.run(port=5000)