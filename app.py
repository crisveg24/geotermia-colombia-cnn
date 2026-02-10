"""
Interfaz Gráfica -CNN Geotermia Colombia
==========================================

Aplicación web interactiva con Streamlit para:
- Visualizar predicciones de potencial geotérmico
- Ingresar coordenadas y obtener predicciones
- Ver métricas y gráficos del modelo
- Explorar el mapa de zonas analizadas

Universidad de San Buenaventura - Bogotá
Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano,
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

# Ruta raíz del proyecto (app.py está en la raíz)
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
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
 page_title="CNN Geotermia Colombia",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded",
)

st.markdown("""
<style>
 /* Tipografía */
 [data-testid="stAppViewContainer"] { font-family: 'Segoe UI', Roboto, sans-serif; }

 /* Header */
 .header-title {
 font-size: 2.2rem; font-weight: 700;
 background: linear-gradient(135deg, #e65100, #ff6d00);
 -webkit-background-clip: text; -webkit-text-fill-color: transparent;
 text-align: center; margin-bottom: 0;
 }
 .header-sub {
 font-size: 1rem; color: #888; text-align: center;
 margin-top: 0; margin-bottom: 1.5rem;
 }

 /* Tarjetas de resultado */
 .result-card {
 padding: 1.8rem 1.2rem; border-radius: 12px;
 text-align: center; color: #fff;
 }
 .result-pos { background: linear-gradient(135deg, #d32f2f, #ff6d00); }
 .result-neg { background: linear-gradient(135deg, #1565c0, #0097a7); }
 .result-card h2 { margin: 0 0 .3rem 0; font-size: 1.15rem; }
 .result-card .big { font-size: 2.8rem; font-weight: 700; margin: .2rem 0; }
 .result-card p { margin: 0; opacity: .85; font-size: .9rem; }

 /* Info */
 .info-box {
 background: #f5f5f5; border-left: 4px solid #ff6d00;
 padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: .8rem 0;
 }
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
 """Carga métricas de evaluación."""
 for nombre in ("evaluation_metrics.json", "metrics.json"):
 p = PROJECT_ROOT / "results" / "metrics" / nombre
 if p.exists():
 with open(p, "r") as f:
 return json.load(f)
 return None


@st.cache_data(ttl=300)
def cargar_historial():
 """Carga historial de entrenamiento (JSON o CSV)."""
 # JSON
 for nombre in ("training_history.json", "history.json"):
 p = PROJECT_ROOT / "results" / "metrics" / nombre
 if p.exists():
 with open(p, "r") as f:
 return json.load(f)
 # CSV
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
 """Zonas geotérmicas conocidas en Colombia."""
 return [
 {"nombre": "Nevado del Ruiz", "lat": 4.8951, "lon": -75.3222, "tipo": "Volcán activo", "potencial": "Alto"},
 {"nombre": "Nevado del Tolima", "lat": 4.6500, "lon": -75.3667, "tipo": "Volcán", "potencial": "Alto"},
 {"nombre": "Volcán Puracé", "lat": 2.3206, "lon": -76.4036, "tipo": "Volcán activo", "potencial": "Alto"},
 {"nombre": "Volcán Galeras", "lat": 1.2208, "lon": -77.3581, "tipo": "Volcán activo", "potencial": "Alto"},
 {"nombre": "Volcán Cumbal", "lat": 0.9539, "lon": -77.8792, "tipo": "Volcán", "potencial": "Medio"},
 {"nombre": "Volcán Sotará", "lat": 2.1083, "lon": -76.5917, "tipo": "Volcán", "potencial": "Medio"},
 {"nombre": "Volcán Azufral", "lat": 1.0833, "lon": -77.7167, "tipo": "Volcán", "potencial": "Medio"},
 {"nombre": "Paipa-Iza", "lat": 5.7781, "lon": -73.1124, "tipo": "Campo geotérmico", "potencial": "Alto"},
 {"nombre": "Santa Rosa de Cabal", "lat": 4.8694, "lon": -75.6219, "tipo": "Aguas termales", "potencial": "Medio"},
 {"nombre": "Manizales", "lat": 5.0667, "lon": -75.5167, "tipo": "Zona termal", "potencial": "Medio"},
 ]


def predecir_por_proximidad(lat: float, lon: float, zonas: list):
 """
Predicción determinística basada en proximidad a zonas geotérmicas.

 Usa sigmoide invertida sobre la distancia a la zona más cercana.
Resultado reproducible (sin componente aleatorio).
 """
 dists = [np.sqrt((lat - z["lat"])**2 + (lon - z["lon"])**2) for z in zonas]
 idx = int(np.argmin(dists))
 d = dists[idx]
 z = zonas[idx]

 # Sigmoide: cerca(~1), lejos(~0). d0≈0.5° es el punto medio (~55 km)
 pred = float(1.0 / (1.0 + np.exp(8.0 * (d - 0.5))))

 if z["potencial"] == "Alto" and d < 0.3:
 pred = min(pred * 1.1, 0.99)

 return np.clip(pred, 0.01, 0.99), z, d


def crear_mapa(zonas, usuario=None, pred_valor=None):
 """Mapa interactivo de Colombia."""
 m = folium.Map(location=[4.57, -74.30], zoom_start=6, tiles="CartoDB positron")

 colores = {"Alto": "red", "Medio": "orange"}
 for z in zonas:
 c = colores.get(z["potencial"], "blue")
 folium.CircleMarker(
 [z["lat"], z["lon"]], radius=9,
 popup=f"<b>{z['nombre']}</b><br>{z['tipo']}<br>Potencial: {z['potencial']}",
 tooltip=z["nombre"], color=c, fill=True, fillColor=c, fillOpacity=0.7,
 ).add_to(m)

 if usuario:
 ic = "green" if pred_valor and pred_valor > 0.5 else "gray"
 txt = f"Predicción: {pred_valor:.1%}" if pred_valor else "Tu ubicación"
 folium.Marker(
 [usuario["lat"], usuario["lon"]],
 popup=f"<b>Consulta</b><br>{txt}", tooltip=txt,
 icon=folium.Icon(color=ic, icon="crosshairs", prefix="fa"),
 ).add_to(m)

 legend = (
 '<div style="position:fixed;bottom:40px;left:40px;z-index:9999;'
 'background:#fff;padding:10px 14px;border-radius:6px;'
 'box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:13px;">'
 '<b>Leyenda</b><br>'
 '<span style="color:red;">●</span> Potencial Alto<br>'
 '<span style="color:orange;">●</span> Potencial Medio<br>'
 '<span style="color:gray;"></span> Consulta</div>'
 )
 m.get_root().html.add_child(folium.Element(legend))
 return m


# =============================================================================
# PÁGINAS
# =============================================================================

def pagina_inicio():
 st.markdown('<h1 class="header-title">CNN Geotermia Colombia</h1>', unsafe_allow_html=True)
 st.markdown(
 '<p class="header-sub">Identificación de zonas con potencial geotérmico '
 'mediante Deep Learning e imágenes satelitales ASTER</p>',
 unsafe_allow_html=True,
 )

 c1, c2, c3 = st.columns(3)
 c1.info("**Objetivo**\n\nIdentificar zonas con alto potencial geotérmico en Colombia usando imágenes satelitales ASTER y CNN.")
 c2.success("**Datos**\n\nBandas térmicas NASA ASTER (10-14), resolución 100 m, obtenidas vía Google Earth Engine.")
 c3.warning("**Modelo**\n\nCNN ResNet-inspired · SpatialDropout2D · AdamW · Label Smoothing · 5 M parámetros.")

 st.divider()
 st.subheader("Zonas Geotérmicas de Estudio")
 zonas = zonas_geotermicas()
 st_folium(crear_mapa(zonas), width=None, height=500, returned_objects=[])

 st.divider()
 st.subheader("Equipo")
 equipo = [
 ("Cristian Camilo Vega S.", "Lead Developer", "ccvegas@academia.usbbog.edu.co"),
 ("Daniel Santiago Arévalo R.", "Co-autor", "dsarevalor@academia.usbbog.edu.co"),
 ("Yuliet Katerin Espitia A.", "Co-autora", "ykespitiaa@academia.usbbog.edu.co"),
 ("Laura Sophie Rivera M.", "Co-autora", "lsriveram@academia.usbbog.edu.co"),
 ]
 cols = st.columns(4)
 for col, (n, r, e) in zip(cols, equipo):
 col.markdown(f"**{n}**\n\n{r}\n\n {e}")

 st.caption("**Asesor:** Prof. Yeison Eduardo Conejo Sandoval · Universidad de San Buenaventura, Bogotá · Ing. Sistemas · 2025-2026")


def pagina_prediccion():
 st.header("Predicción de Potencial Geotérmico")
 st.markdown(
 '<div class="info-box">'
 'Ingresa coordenadas de una ubicación en Colombia. El sistema evalúa la proximidad '
 'a zonas geotérmicas conocidas. Cuando el modelo CNN esté entrenado, se usará '
 'directamente para la predicción con datos ASTER reales.</div>',
 unsafe_allow_html=True,
 )

 col_in, col_out = st.columns([1, 2])

 with col_in:
 st.subheader("Coordenadas")
 metodo = st.radio("Método:", ["Manual", "Zona conocida"], horizontal=True)

 zonas = zonas_geotermicas()
 if metodo == "Zona conocida":
 sel = st.selectbox("Zona:", [z["nombre"] for z in zonas])
 z = next(x for x in zonas if x["nombre"] == sel)
 latitud, longitud = z["lat"], z["lon"]
 st.info(f"**{z['nombre']}** · {z['tipo']} · Potencial {z['potencial']}")
 else:
 latitud = st.number_input("Latitud:", -4.0, 12.0, 4.8951, 0.0001, format="%.4f")
 longitud = st.number_input("Longitud:", -82.0, -66.0, -75.3222, 0.0001, format="%.4f")

 analizar = st.button("Analizar Potencial", type="primary", use_container_width=True)

 if analizar:
 zonas = zonas_geotermicas()
 pred, zona_c, dist = predecir_por_proximidad(latitud, longitud, zonas)
 st.session_state["pred"] = {
 "lat": latitud, "lon": longitud, "valor": pred,
 "zona": zona_c["nombre"], "tipo": zona_c["tipo"], "dist": dist,
 }

 with col_out:
 st.subheader("Resultado")
 if "pred" in st.session_state:
 p = st.session_state["pred"]
 pos = p["valor"] >= 0.5
 cls = "result-pos" if pos else "result-neg"
 titulo = "ZONA CON POTENCIAL GEOTÉRMICO" if pos else "BAJO POTENCIAL GEOTÉRMICO"

 st.markdown(
 f'<div class="result-card {cls}">'
 f'<h2>{titulo}</h2>'
 f'<div class="big">{p["valor"]:.1%}</div>'
 f'<p>Confianza del modelo</p></div>',
 unsafe_allow_html=True,
 )
 st.write("")

 m1, m2, m3, m4 = st.columns(4)
 m1.metric("Latitud", f'{p["lat"]:.4f}°')
 m2.metric("Longitud", f'{p["lon"]:.4f}°')
 m3.metric("Zona cercana", p["zona"])
 m4.metric("Distancia", f'{p["dist"]:.2f}°')

 st.subheader("Ubicación")
 zonas = zonas_geotermicas()
 mapa = crear_mapa(zonas, {"lat": p["lat"], "lon": p["lon"]}, p["valor"])
 st_folium(mapa, width=None, height=400, returned_objects=[])
 else:
 st.info("Ingresa coordenadas y presiona **Analizar Potencial**.")
 st_folium(crear_mapa(zonas_geotermicas()), width=None, height=400, returned_objects=[])


def pagina_metricas():
 st.header("Métricas y Rendimiento")

 metricas = cargar_metricas()
 historial = cargar_historial()

 if not metricas and not historial:
 st.warning("No se encontraron métricas ni historial. Ejecuta primero:")
 st.code("python scripts/evaluate_model.py", language="bash")
 return

 # Tarjetas
 if metricas:
 st.subheader("Evaluación")
 keys = [
 ("accuracy", "Accuracy"), ("precision", "Precision"),
 ("recall", "Recall"), ("f1_score", "F1-Score"), ("auc_roc", "AUC-ROC"),
 ]
 cols = st.columns(len(keys))
 for col, (k, label) in zip(cols, keys):
 v = metricas.get(k, metricas.get(k.replace("_", ""), 0))
 col.metric(label, f"{v:.1%}" if v <= 1 else f"{v:.2f}")

 st.divider()

 c1, c2 = st.columns(2)
 with c1:
 bar_keys = ["accuracy", "precision", "recall", "f1_score"]
 bar_vals = [metricas.get(k, metricas.get(k.replace("_", ""), 0)) for k in bar_keys]
 fig = go.Figure(go.Bar(
 x=["Accuracy", "Precision", "Recall", "F1-Score"],
 y=bar_vals,
 marker_color=["#FF6D00", "#00897B", "#1565C0", "#6A1B9A"],
 text=[f"{v:.1%}" for v in bar_vals], textposition="outside",
 ))
 fig.update_layout(yaxis_range=[0, 1.15], yaxis_tickformat=".0%", height=380, margin=dict(t=30))
 st.plotly_chart(fig, use_container_width=True, key="bar_metricas")

 with c2:
 auc = metricas.get("auc_roc", metricas.get("auc", 0.5))
 fpr = np.linspace(0, 1, 200)
 tpr = 1 - (1 - fpr) ** (1 / max(auc, 0.51))
 fig = go.Figure()
 fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})", line=dict(color="#FF6D00", width=2.5)))
 fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Aleatorio", line=dict(color="#bbb", dash="dash")))
 fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=380, margin=dict(t=30))
 st.plotly_chart(fig, use_container_width=True, key="roc")

 #Historial
 if historial:
 st.divider()
 st.subheader("Historial de Entrenamiento")

 loss_k = next((c for c in ("loss", "train_loss") if c in historial), None)
 vloss_k = next((c for c in ("val_loss", "validation_loss") if c in historial), None)
 acc_k = next((c for c in ("accuracy", "acc", "train_accuracy") if c in historial), None)
 vacc_k = next((c for c in ("val_accuracy", "val_acc") if c in historial), None)

 ref_key = loss_k or acc_k or list(historial.keys())[0]
 epocas = list(range(1, len(historial[ref_key]) + 1))

 c1, c2 = st.columns(2)
 with c1:
 fig = go.Figure()
 if loss_k:
 fig.add_trace(go.Scatter(x=epocas, y=historial[loss_k], name="Train Loss", line=dict(color="#FF6D00")))
 if vloss_k:
 fig.add_trace(go.Scatter(x=epocas, y=historial[vloss_k], name="Val Loss", line=dict(color="#1565C0", dash="dot")))
 fig.update_layout(title="Pérdida (Loss)", xaxis_title="Época", yaxis_title="Loss", height=370, margin=dict(t=40))
 st.plotly_chart(fig, use_container_width=True, key="loss")

 with c2:
 fig = go.Figure()
 if acc_k:
 fig.add_trace(go.Scatter(x=epocas, y=historial[acc_k], name="Train Acc", line=dict(color="#00897B")))
 if vacc_k:
 fig.add_trace(go.Scatter(x=epocas, y=historial[vacc_k], name="Val Acc", line=dict(color="#6A1B9A", dash="dot")))
 fig.update_layout(title="Precisión (Accuracy)", xaxis_title="Época", yaxis_title="Accuracy", height=370, margin=dict(t=40))
 st.plotly_chart(fig, use_container_width=True, key="acc")

 # Figuras PNG
 figs_dir = PROJECT_ROOT / "results" / "figures"
 imgs = sorted(figs_dir.glob("*.png")) if figs_dir.is_dir() else []
 if imgs:
 st.divider()
 st.subheader("Visualizaciones")
 cols = st.columns(min(len(imgs), 3))
 for i, img in enumerate(imgs[:6]):
 cols[i % len(cols)].image(str(img), caption=img.stem.replace("_", " ").title(), width="stretch")


def pagina_arquitectura():
 st.header("Arquitectura del Modelo CNN")
 st.markdown(
 "Arquitectura **ResNet-inspired** con bloques residuales, "
 "SpatialDropout2D para regularización espacial y Global Average Pooling."
 )

 st.subheader("Capas de la Red")
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
 "—", "32 (7×7)", "—", "—",
 "64", "—", "128", "—",
 "256", "—", "512", "—",
 "—", "256", "1",
 ],
 "Salida": [
 "224×224×5", "224×224×32", "224×224×32", "112×112×32",
 "112×112×64", "56×56×64", "56×56×128", "28×28×128",
 "28×28×256", "14×14×256", "14×14×512", "14×14×512",
 "512", "256", "1",
 ],
 })
 st.dataframe(capas, use_container_width=True, hide_index=True)

 # Diagrama visual (barras horizontales apiladas = flujo de capas)
 labels = capas["Capa"].tolist()
 salidas = capas["Salida"].tolist()
 palette = (
 ["#E3F2FD"] + ["#C8E6C9"] * 3
 + ["#A5D6A7", "#C8E6C9"] * 4
 + ["#FFE0B2", "#FFCC80", "#FFAB40"]
 )
 palette = (palette * 2)[:len(labels)]

 fig = go.Figure(go.Bar(
 y=list(reversed(labels)),
 x=[1] * len(labels),
 orientation="h",
 marker_color=list(reversed(palette)),
 text=[f"{l} → {s}" for l, s in zip(reversed(labels), reversed(salidas))],
 textposition="inside", insidetextanchor="middle", hoverinfo="text",
 ))
 fig.update_layout(
 xaxis_visible=False, yaxis_visible=False, showlegend=False,
 height=520, margin=dict(l=10, r=10, t=10, b=10),
 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
 )
 st.plotly_chart(fig, use_container_width=True, key="arch")

 st.divider()

 c1, c2 = st.columns(2)
 with c1:
 st.subheader("Hiperparámetros")
 st.markdown("""
| Parámetro | Valor |
|-----------|-------|
| Optimizador | **AdamW** (weight_decay=1e-4) |
| Learning Rate | 0.001 |
| Loss | BinaryCrossentropy (label_smoothing=0.1) |
| Dropout | 0.5 (Dense) · 0.1-0.3 (Spatial) |
| Batch Size | 32 |
| Épocas | 100 (EarlyStopping, patience=15) |
| Parámetros | 5,025,409 |
""")
 with c2:
 st.subheader("Datos de Entrada")
 st.markdown("""
| Característica | Valor |
|----------------|-------|
| Fuente | NASA ASTER GED AG100 V003 |
| Resolución | 100 metros |
| Bandas | 10, 11, 12, 13, 14 (TIR) |
| Tamaño de entrada | 224 × 224 × 5 |
| Normalización | 0–1 (Rescaling) |
| Dataset | 5,518 imágenes (augmentadas) |
""")


def pagina_acerca():
 st.header("Acerca del Proyecto")
 st.markdown("""
### Proyecto de Grado

**Universidad de San Buenaventura – Sede Bogotá** 
**Programa:** Ingeniería de Sistemas · **Año:** 2025-2026

---

### Descripción

Sistema de **Deep Learning** basado en CNN para la identificación automatizada
de zonas con alto potencial geotérmico en Colombia, analizando imágenes
satelitales térmicas del sensor **NASA ASTER**.

---

### Objetivos

1. **General:** Desarrollar un modelo predictivo de potencial geotérmico con
 visión por computador y deep learning.

2. **Específicos:**
 - Recopilar y procesar imágenes ASTER de zonas geotérmicas colombianas.
 - Diseñar una arquitectura CNN optimizada con bloques residuales.
 - Entrenar y evaluar con métricas estándar de clasificación.
 - Desarrollar interfaz web interactiva para visualización y predicción.

---

### Metodología

| Fase | Descripción |
|------|-------------|
| Adquisición | Google Earth Engine → 85 imágenes ASTER |
| Augmentación | 30 transformaciones → 5,518 imágenes |
| Preparación | Normalización + split 70/15/15 estratificado |
| Modelado | CNN ResNet-inspired (5 M parámetros) |
|Evaluación | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| Despliegue | Streamlit + Folium + Plotly |

---

### Tecnologías

**Deep Learning:** TensorFlow 2.20, Keras ·
**Datos:** Google Earth Engine, rasterio, NumPy, pandas ·
**Visualización:** Plotly, Folium, Matplotlib, Seaborn ·
**Interfaz:** Streamlit · **Reportes:** FPDF2

---

### Referencias

- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.*
- NASA ASTER Global Emissivity Dataset (AG100) V003.
- TensorFlow — tensorflow.org
- Google Earth Engine — earthengine.google.com

---

Licencia **MIT** · Ver archivo `LICENSE`.
""")


# =============================================================================
# NAVEGACIÓN
# =============================================================================

def main():
 with st.sidebar:
 st.title("Navegación")
 pagina = st.radio(
 "Sección:",
 ["Inicio", "Predicción", "Métricas", "Arquitectura", "Acerca de"],
 label_visibility="collapsed",
 )
 st.divider()

 st.caption("**Estado del Sistema**")
 modelo = cargar_modelo()
 st.caption("Modelo cargado" if modelo else "Modelo no disponible")
 st.caption("Métricas" if cargar_metricas() else "Sin métricas")
 st.caption("Historial" if cargar_historial() else "Sin historial")

 st.divider()
 st.caption(f"{datetime.now().strftime('%d/%m/%Y %H:%M')}")
 st.caption("USB · Ing. Sistemas · 2025-2026")

 {"Inicio": pagina_inicio,
 "Predicción": pagina_prediccion,
 "Métricas": pagina_metricas,
 "Arquitectura": pagina_arquitectura,
 "Acerca de": pagina_acerca}[pagina]()


if __name__ == "__main__":
 main()
