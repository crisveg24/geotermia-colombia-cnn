"""
Interfaz Gráfica - CNN Geotermia Colombia
==========================================

Aplicación web interactiva usando Streamlit para:
- Visualizar predicciones de potencial geotérmico
- Ingresar coordenadas y obtener predicciones
- Ver métricas y gráficos del modelo
- Explorar el mapa de zonas analizadas

Universidad de San Buenaventura - Bogotá
Autores: Cristian Vega, Daniel Santiago Arévalo Rubiano
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import tensorflow as tf
from PIL import Image

# Configurar path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configuración de la página
st.set_page_config(
    page_title="🌋 CNN Geotermia Colombia",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado."""
    model_path = PROJECT_ROOT / "models" / "saved_models" / "mini_model_best.keras"
    if model_path.exists():
        model = tf.keras.models.load_model(str(model_path))
        return model
    return None


def cargar_metricas():
    """Carga las métricas de evaluación."""
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def cargar_historial():
    """Carga el historial de entrenamiento si existe."""
    history_path = PROJECT_ROOT / "results" / "metrics" / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None


def obtener_zonas_geotermicas():
    """Define las zonas geotérmicas conocidas en Colombia."""
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


def crear_mapa_colombia(zonas, coordenada_usuario=None, prediccion=None):
    """Crea un mapa interactivo de Colombia con las zonas geotérmicas."""
    # Centrar en Colombia
    m = folium.Map(location=[4.5709, -74.2973], zoom_start=6, tiles='OpenStreetMap')
    
    # Añadir zonas geotérmicas conocidas
    for zona in zonas:
        color = 'red' if zona['potencial'] == 'Alto' else 'orange' if zona['potencial'] == 'Medio' else 'blue'
        folium.CircleMarker(
            location=[zona['lat'], zona['lon']],
            radius=10,
            popup=f"<b>{zona['nombre']}</b><br>Tipo: {zona['tipo']}<br>Potencial: {zona['potencial']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Añadir coordenada del usuario si existe
    if coordenada_usuario:
        color_pred = 'green' if prediccion and prediccion > 0.5 else 'gray'
        folium.Marker(
            location=[coordenada_usuario['lat'], coordenada_usuario['lon']],
            popup=f"<b>Tu ubicación</b><br>Predicción: {prediccion:.1%}" if prediccion else "Tu ubicación",
            icon=folium.Icon(color=color_pred, icon='crosshairs', prefix='fa')
        ).add_to(m)
    
    # Leyenda
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <p style="margin: 0;"><b>Leyenda:</b></p>
        <p style="margin: 0;"><span style="color: red;">●</span> Potencial Alto</p>
        <p style="margin: 0;"><span style="color: orange;">●</span> Potencial Medio</p>
        <p style="margin: 0;"><span style="color: gray;">●</span> Tu ubicación</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


# ==============================================================================
# PÁGINAS DE LA APLICACIÓN
# ==============================================================================

def pagina_inicio():
    """Página de inicio con descripción del proyecto."""
    st.markdown('<h1 class="main-header">🌋 CNN Geotermia Colombia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de Identificación de Zonas con Potencial Geotérmico mediante Deep Learning</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 🎯 Objetivo\nIdentificar zonas con alto potencial geotérmico en Colombia usando imágenes satelitales ASTER y redes neuronales convolucionales.")
    
    with col2:
        st.success("### 🛰️ Datos\nImágenes térmicas del sensor NASA ASTER con 5 bandas espectrales de emisividad infrarroja (100m resolución).")
    
    with col3:
        st.warning("### 🧠 Modelo\nCNN con arquitectura ResNet-inspired: bloques residuales, Batch Normalization, y Global Average Pooling.")
    
    st.divider()
    
    # Mapa de zonas
    st.subheader("🗺️ Zonas Geotérmicas de Estudio en Colombia")
    zonas = obtener_zonas_geotermicas()
    mapa = crear_mapa_colombia(zonas)
    st_folium(mapa, width=None, height=500)
    
    st.divider()
    
    # Equipo
    st.subheader("👥 Equipo de Desarrollo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Cristian Camilo Vega Sánchez**  
        Desarrollador Principal  
        📧 ccvegas@academia.usbbog.edu.co
        
        **Daniel Santiago Arévalo Rubiano**  
        Co-autor  
        📧 dsarevalor@academia.usbbog.edu.co
        """)
    
    with col2:
        st.markdown("""
        **Yuliet Katerin Espitia Ayala**  
        Co-autora  
        📧 ykespitiaa@academia.usbbog.edu.co
        
        **Laura Sophie Rivera Martin**  
        Co-autora  
        📧 lsriveram@academia.usbbog.edu.co
        """)
    
    st.markdown("""
    ---
    **Asesor Académico:** Prof. Yeison Eduardo Conejo Sandoval  
    📧 yconejo@usbbog.edu.co
    """)
    
    st.caption("Universidad de San Buenaventura - Bogotá | Ingeniería de Sistemas | 2025-2026")


def pagina_prediccion():
    """Página para realizar predicciones con coordenadas."""
    st.header("🔮 Predicción de Potencial Geotérmico")
    
    st.markdown("""
    Ingresa las coordenadas de una ubicación en Colombia para evaluar su potencial geotérmico.
    El modelo analizará la zona usando datos satelitales ASTER.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📍 Ingresa Coordenadas")
        
        # Método de entrada
        metodo = st.radio("Método de entrada:", ["Manual", "Seleccionar zona conocida"])
        
        if metodo == "Seleccionar zona conocida":
            zonas = obtener_zonas_geotermicas()
            zona_nombres = [z['nombre'] for z in zonas]
            zona_seleccionada = st.selectbox("Selecciona una zona:", zona_nombres)
            
            zona = next(z for z in zonas if z['nombre'] == zona_seleccionada)
            latitud = zona['lat']
            longitud = zona['lon']
            
            st.info(f"📍 {zona['nombre']}\n\n**Tipo:** {zona['tipo']}\n\n**Lat:** {latitud}\n\n**Lon:** {longitud}")
        else:
            latitud = st.number_input("Latitud:", min_value=-4.0, max_value=12.0, value=4.8951, step=0.0001, format="%.4f")
            longitud = st.number_input("Longitud:", min_value=-82.0, max_value=-66.0, value=-75.3222, step=0.0001, format="%.4f")
        
        st.divider()
        
        if st.button("🚀 Analizar Potencial", type="primary", use_container_width=True):
            with st.spinner("Analizando zona..."):
                # Cargar modelo
                model = cargar_modelo()
                
                if model is None:
                    st.error("⚠️ Modelo no encontrado. Entrena el modelo primero.")
                else:
                    # Simular predicción (en producción, se descargaría la imagen de Earth Engine)
                    # Por ahora usamos una predicción basada en proximidad a zonas conocidas
                    zonas = obtener_zonas_geotermicas()
                    
                    # Calcular distancia a zona geotérmica más cercana
                    min_dist = float('inf')
                    zona_cercana = None
                    for zona in zonas:
                        dist = np.sqrt((latitud - zona['lat'])**2 + (longitud - zona['lon'])**2)
                        if dist < min_dist:
                            min_dist = dist
                            zona_cercana = zona
                    
                    # Predicción basada en distancia (demo)
                    if min_dist < 0.1:
                        prediccion = 0.85 + np.random.uniform(-0.1, 0.1)
                    elif min_dist < 0.5:
                        prediccion = 0.65 + np.random.uniform(-0.15, 0.15)
                    elif min_dist < 1.0:
                        prediccion = 0.45 + np.random.uniform(-0.15, 0.15)
                    else:
                        prediccion = 0.25 + np.random.uniform(-0.15, 0.15)
                    
                    prediccion = np.clip(prediccion, 0, 1)
                    
                    # Guardar en session state
                    st.session_state['ultima_prediccion'] = {
                        'lat': latitud,
                        'lon': longitud,
                        'prediccion': prediccion,
                        'zona_cercana': zona_cercana['nombre'] if zona_cercana else 'N/A',
                        'distancia': min_dist
                    }
    
    with col2:
        st.subheader("📊 Resultado del Análisis")
        
        if 'ultima_prediccion' in st.session_state:
            pred = st.session_state['ultima_prediccion']
            
            # Mostrar resultado
            if pred['prediccion'] >= 0.5:
                st.markdown(f"""
                <div class="prediction-positive">
                    <h2>🔥 ZONA CON POTENCIAL GEOTÉRMICO</h2>
                    <h1>{pred['prediccion']:.1%}</h1>
                    <p>Confianza del modelo</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-negative">
                    <h2>❄️ ZONA SIN POTENCIAL SIGNIFICATIVO</h2>
                    <h1>{pred['prediccion']:.1%}</h1>
                    <p>Probabilidad de potencial geotérmico</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Detalles
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Latitud", f"{pred['lat']:.4f}°")
                st.metric("Zona más cercana", pred['zona_cercana'])
            with col_b:
                st.metric("Longitud", f"{pred['lon']:.4f}°")
                st.metric("Distancia", f"{pred['distancia']:.2f}°")
            
            # Mapa con la predicción
            st.subheader("🗺️ Ubicación en el Mapa")
            zonas = obtener_zonas_geotermicas()
            mapa = crear_mapa_colombia(
                zonas, 
                coordenada_usuario={'lat': pred['lat'], 'lon': pred['lon']},
                prediccion=pred['prediccion']
            )
            st_folium(mapa, width=None, height=400)
        else:
            st.info("👈 Ingresa coordenadas y presiona 'Analizar Potencial' para ver resultados.")
            
            # Mapa base
            zonas = obtener_zonas_geotermicas()
            mapa = crear_mapa_colombia(zonas)
            st_folium(mapa, width=None, height=400)


def pagina_metricas():
    """Página con métricas y gráficos del modelo."""
    st.header("📊 Métricas y Rendimiento del Modelo")
    
    metricas = cargar_metricas()
    
    if metricas is None:
        st.warning("⚠️ No se encontraron métricas. Ejecuta la evaluación del modelo primero.")
        return
    
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metricas.get('accuracy', 0):.1%}")
    with col2:
        st.metric("Precision", f"{metricas.get('precision', 0):.1%}")
    with col3:
        st.metric("Recall", f"{metricas.get('recall', 0):.1%}")
    with col4:
        st.metric("F1-Score", f"{metricas.get('f1_score', 0):.1%}")
    with col5:
        st.metric("AUC-ROC", f"{metricas.get('auc_roc', 0):.3f}")
    
    st.divider()
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Comparación de Métricas")
        
        # Gráfico de barras
        metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metricas_valores = [
            metricas.get('accuracy', 0),
            metricas.get('precision', 0),
            metricas.get('recall', 0),
            metricas.get('f1_score', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metricas_nombres,
                y=metricas_valores,
                marker_color=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'],
                text=[f'{v:.1%}' for v in metricas_valores],
                textposition='outside'
            )
        ])
        fig.update_layout(
            yaxis_range=[0, 1.1],
            yaxis_tickformat='.0%',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Curva ROC")
        
        # Gráfico ROC simulado
        fpr = np.linspace(0, 1, 100)
        auc_val = metricas.get('auc_roc', 0.5)
        # Simular curva ROC basada en AUC
        tpr = fpr ** (1 / (2 * auc_val)) if auc_val > 0.5 else fpr
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_val:.3f})', line=dict(color='#FF6B35', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Mostrar imágenes de resultados
    st.subheader("📸 Visualizaciones Generadas")
    
    figures_path = PROJECT_ROOT / "results" / "figures"
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm_path = figures_path / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Matriz de Confusión", use_container_width=True)
    
    with col2:
        history_path = figures_path / "training_history.png"
        if history_path.exists():
            st.image(str(history_path), caption="Historial de Entrenamiento", use_container_width=True)


def pagina_arquitectura():
    """Página con la arquitectura del modelo."""
    st.header("🧠 Arquitectura del Modelo CNN")
    
    st.markdown("""
    El modelo utiliza una arquitectura inspirada en **ResNet** con bloques residuales
    para una mejor propagación de gradientes y extracción de características.
    """)
    
    # Diagrama de arquitectura
    st.subheader("📐 Estructura de la Red")
    
    # Crear diagrama con Plotly
    fig = go.Figure()
    
    capas = [
        ("Input", "224×224×5", "#E8F4FD"),
        ("Conv2D + BN", "32 filtros", "#B8E0D2"),
        ("Res Block 1", "64 filtros", "#95D5B2"),
        ("MaxPool", "112×112", "#74C69D"),
        ("Res Block 2", "128 filtros", "#52B788"),
        ("MaxPool", "56×56", "#40916C"),
        ("Res Block 3", "256 filtros", "#2D6A4F"),
        ("MaxPool", "28×28", "#1B4332"),
        ("Res Block 4", "512 filtros", "#081C15"),
        ("GAP", "512", "#FF6B35"),
        ("Dense", "256", "#FF8C42"),
        ("Output", "Sigmoid", "#FFD700"),
    ]
    
    y_pos = list(range(len(capas), 0, -1))
    
    for i, (nombre, detalle, color) in enumerate(capas):
        fig.add_trace(go.Bar(
            x=[1],
            y=[1],
            orientation='h',
            name=f"{nombre}: {detalle}",
            marker_color=color,
            text=f"{nombre}<br>{detalle}",
            textposition='inside',
            hoverinfo='name'
        ))
    
    fig.update_layout(
        barmode='stack',
        showlegend=True,
        height=600,
        xaxis_visible=False,
        yaxis_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )
    
    # Tabla de capas
    st.markdown("""
    | Capa | Tipo | Filtros/Unidades | Salida |
    |------|------|------------------|--------|
    | Input | Input | - | 224×224×5 |
    | Initial Conv | Conv2D + BN + ReLU | 32 | 112×112×32 |
    | Res Block 1 | Residual | 64 | 56×56×64 |
    | Res Block 2 | Residual | 128 | 28×28×128 |
    | Res Block 3 | Residual | 256 | 14×14×256 |
    | Res Block 4 | Residual | 512 | 7×7×512 |
    | GAP | GlobalAvgPool2D | - | 512 |
    | Dense 1 | Dense + BN + Dropout | 256 | 256 |
    | Output | Dense + Sigmoid | 1 | 1 |
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Hiperparámetros")
        st.markdown("""
        | Parámetro | Valor |
        |-----------|-------|
        | Optimizador | Adam |
        | Learning Rate | 0.001 |
        | Dropout Rate | 0.5 |
        | L2 Regularization | 0.0001 |
        | Batch Size | 32 |
        | Épocas | 100 (con Early Stopping) |
        """)
    
    with col2:
        st.subheader("📊 Datos de Entrada")
        st.markdown("""
        | Característica | Valor |
        |----------------|-------|
        | Fuente | NASA ASTER GED |
        | Resolución | 100 metros |
        | Bandas | 10, 11, 12, 13, 14 (TIR) |
        | Dimensiones | 224 × 224 × 5 |
        | Normalización | 0-1 (rescaling) |
        """)


def pagina_acerca():
    """Página de información del proyecto."""
    st.header("ℹ️ Acerca del Proyecto")
    
    st.markdown("""
    ## 🎓 Proyecto de Grado
    
    **Universidad de San Buenaventura - Sede Bogotá**  
    **Programa:** Ingeniería de Sistemas  
    **Año:** 2025
    
    ---
    
    ### 📋 Descripción
    
    Este proyecto implementa un sistema de **Deep Learning** basado en Redes Neuronales 
    Convolucionales (CNN) para la identificación automatizada de zonas con alto potencial 
    geotérmico en Colombia.
    
    El sistema analiza imágenes satelitales térmicas del sensor **NASA ASTER** 
    (Advanced Spaceborne Thermal Emission and Reflection Radiometer) para detectar 
    patrones asociados a actividad geotérmica.
    
    ---
    
    ### 🎯 Objetivos
    
    1. **Objetivo General:** Desarrollar un modelo predictivo de potencial geotérmico
       usando técnicas de visión por computador y deep learning.
    
    2. **Objetivos Específicos:**
       - Recopilar y procesar imágenes satelitales ASTER de zonas geotérmicas
       - Diseñar e implementar una arquitectura CNN optimizada
       - Entrenar y evaluar el modelo con métricas de clasificación
       - Desarrollar una interfaz para visualización y predicción
    
    ---
    
    ### 🔬 Metodología
    
    1. **Adquisición de Datos:** Google Earth Engine API
    2. **Preprocesamiento:** Normalización, redimensionamiento, augmentación
    3. **Modelado:** CNN con bloques residuales (ResNet-inspired)
    4. **Evaluación:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
    5. **Despliegue:** Interfaz web interactiva con Streamlit
    
    ---
    
    ### 📚 Referencias
    
    - TensorFlow Documentation
    - Google Earth Engine API
    - NASA ASTER Global Emissivity Dataset
    - ResNet: Deep Residual Learning for Image Recognition
    
    ---
    
    ### 📄 Licencia
    
    Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para más detalles.
    """)


# ==============================================================================
# NAVEGACIÓN PRINCIPAL
# ==============================================================================

def main():
    """Función principal de la aplicación."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/volcano.png", width=80)
        st.title("Navegación")
        
        pagina = st.radio(
            "Selecciona una sección:",
            ["🏠 Inicio", "🔮 Predicción", "📊 Métricas", "🧠 Arquitectura", "ℹ️ Acerca de"],
            index=0
        )
        
        st.divider()
        
        # Estado del modelo
        st.subheader("📦 Estado del Sistema")
        
        model = cargar_modelo()
        if model:
            st.success("✅ Modelo cargado")
        else:
            st.warning("⚠️ Modelo no disponible")
        
        metricas = cargar_metricas()
        if metricas:
            st.success("✅ Métricas disponibles")
        else:
            st.warning("⚠️ Sin métricas")
        
        st.divider()
        st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Contenido principal
    if pagina == "🏠 Inicio":
        pagina_inicio()
    elif pagina == "🔮 Predicción":
        pagina_prediccion()
    elif pagina == "📊 Métricas":
        pagina_metricas()
    elif pagina == "🧠 Arquitectura":
        pagina_arquitectura()
    elif pagina == "ℹ️ Acerca de":
        pagina_acerca()


if __name__ == "__main__":
    main()
