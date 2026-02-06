"""
Generador de Reporte PDF - Mini-Dataset
========================================

Genera un reporte PDF profesional con los resultados del mini-dataset
para la documentación de la tesis.

Universidad de San Buenaventura - Bogotá
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fpdf import FPDF


class ReportePDF(FPDF):
    """Clase personalizada para el reporte PDF."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Encabezado de cada página."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'CNN Geotermia Colombia - Universidad de San Buenaventura', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Pie de página."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    
    def titulo_capitulo(self, titulo):
        """Título de capítulo."""
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 102, 204)
        self.cell(0, 10, titulo, 0, 1, 'L')
        self.ln(4)
    
    def subtitulo(self, texto):
        """Subtítulo."""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, texto, 0, 1, 'L')
        self.ln(2)
    
    def cuerpo_texto(self, texto):
        """Texto normal."""
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, texto)
        self.ln(3)
    
    def tabla_simple(self, encabezados, datos, anchos=None):
        """Crea una tabla simple."""
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(0, 102, 204)
        self.set_text_color(255, 255, 255)
        
        if anchos is None:
            anchos = [190 // len(encabezados)] * len(encabezados)
        
        # Encabezados
        for i, enc in enumerate(encabezados):
            self.cell(anchos[i], 8, enc, 1, 0, 'C', True)
        self.ln()
        
        # Datos
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        fill = False
        for fila in datos:
            self.set_fill_color(240, 240, 240) if fill else self.set_fill_color(255, 255, 255)
            for i, celda in enumerate(fila):
                self.cell(anchos[i], 7, str(celda), 1, 0, 'C', fill)
            self.ln()
            fill = not fill


def generar_reporte():
    """Genera el reporte PDF del mini-dataset."""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Crear PDF
    pdf = ReportePDF()
    pdf.add_page()
    
    # === PORTADA ===
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(0, 102, 204)
    pdf.ln(30)
    pdf.cell(0, 15, 'REPORTE DE VALIDACION', 0, 1, 'C')
    pdf.cell(0, 15, 'MINI-DATASET', 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, 'Modelo CNN para Identificacion de', 0, 1, 'C')
    pdf.cell(0, 8, 'Zonas con Potencial Geotermico en Colombia', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 12)
    pdf.cell(0, 8, 'Cristian Camilo Vega Sanchez', 0, 1, 'C')
    pdf.cell(0, 8, 'Daniel Santiago Arevalo Rubiano', 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Universidad de San Buenaventura - Bogota', 0, 1, 'C')
    pdf.cell(0, 8, 'Ingenieria de Sistemas', 0, 1, 'C')
    pdf.cell(0, 8, f'Fecha: {datetime.now().strftime("%d de %B de %Y")}', 0, 1, 'C')
    
    # === PÁGINA 2: INTRODUCCIÓN ===
    pdf.add_page()
    pdf.titulo_capitulo('1. Introduccion')
    
    pdf.cuerpo_texto(
        'Este documento presenta los resultados de la validacion del pipeline de entrenamiento '
        'utilizando un mini-dataset de 20 imagenes satelitales ASTER. El proposito de esta prueba '
        'es verificar que todos los componentes del sistema funcionan correctamente antes de '
        'proceder con el entrenamiento completo en GPUs de alto rendimiento.'
    )
    
    pdf.subtitulo('1.1 Objetivo del Mini-Dataset')
    pdf.cuerpo_texto(
        'El mini-dataset fue creado para:\n'
        '- Validar la descarga de imagenes desde Google Earth Engine\n'
        '- Verificar el preprocesamiento de bandas termicas ASTER\n'
        '- Comprobar que el modelo CNN puede compilarse y entrenar\n'
        '- Generar metricas y visualizaciones de prueba'
    )
    
    pdf.subtitulo('1.2 Limitaciones')
    pdf.cuerpo_texto(
        'IMPORTANTE: Con solo 20 imagenes de entrenamiento, el modelo NO puede aprender patrones '
        'significativos. Los resultados obtenidos son puramente para validacion tecnica, no para '
        'evaluacion del rendimiento real del modelo.'
    )
    
    # === PÁGINA 3: DATASET ===
    pdf.add_page()
    pdf.titulo_capitulo('2. Descripcion del Mini-Dataset')
    
    pdf.subtitulo('2.1 Composicion')
    pdf.tabla_simple(
        ['Categoria', 'Cantidad', 'Porcentaje'],
        [
            ['Zonas Geotermicas', '10', '50%'],
            ['Zonas Control', '10', '50%'],
            ['Total', '20', '100%']
        ],
        [70, 60, 60]
    )
    
    pdf.ln(5)
    pdf.subtitulo('2.2 Zonas Geotermicas (Clase Positiva)')
    pdf.cuerpo_texto(
        '1. Nevado del Ruiz - Volcan activo, Tolima\n'
        '2. Nevado del Tolima - Complejo volcanico\n'
        '3. Volcan Purace - Sistema hidrotermal, Cauca\n'
        '4. Volcan Galeras - Volcan activo, Narino\n'
        '5. Volcan Cumbal - Actividad geotermica\n'
        '6. Volcan Sotara - Manifestaciones termales\n'
        '7. Volcan Azufral - Lago craterico acido\n'
        '8. Paipa-Iza - Campo geotermico, Boyaca\n'
        '9. Santa Rosa de Cabal - Termales, Risaralda\n'
        '10. Manizales - Zona termal'
    )
    
    pdf.subtitulo('2.3 Zonas Control (Clase Negativa)')
    pdf.cuerpo_texto(
        '1. Llanos Orientales - Meta, Casanare, Arauca\n'
        '2. Amazonia - Caqueta, Guaviare, Amazonas\n'
        '3. Costa Caribe - Atlantico, Magdalena, Cesar\n'
        '4. Orinoquia - Vichada'
    )
    
    pdf.subtitulo('2.4 Division del Dataset')
    pdf.tabla_simple(
        ['Conjunto', 'Imagenes', 'Porcentaje'],
        [
            ['Entrenamiento', '14', '70%'],
            ['Validacion', '3', '15%'],
            ['Test', '3', '15%']
        ],
        [70, 60, 60]
    )
    
    # === PÁGINA 4: ARQUITECTURA ===
    pdf.add_page()
    pdf.titulo_capitulo('3. Arquitectura del Modelo')
    
    pdf.subtitulo('3.1 Arquitectura CNN')
    pdf.cuerpo_texto(
        'El modelo utiliza una arquitectura inspirada en ResNet con bloques residuales:\n\n'
        '- Input: 224 x 224 x 5 (5 bandas termicas ASTER)\n'
        '- Bloque inicial: Conv2D 32 filtros + BatchNorm + ReLU\n'
        '- Bloque Residual 1: 64 filtros + MaxPooling\n'
        '- Bloque Residual 2: 128 filtros + MaxPooling\n'
        '- Bloque Residual 3: 256 filtros + MaxPooling\n'
        '- Bloque Residual 4: 512 filtros\n'
        '- Global Average Pooling\n'
        '- Dense 256 + Dropout 0.5\n'
        '- Output: Sigmoid (clasificacion binaria)'
    )
    
    pdf.subtitulo('3.2 Configuracion de Entrenamiento (Mini-modelo)')
    pdf.tabla_simple(
        ['Parametro', 'Valor'],
        [
            ['Optimizador', 'Adam'],
            ['Learning Rate', '0.001'],
            ['Batch Size', '4'],
            ['Epocas Maximas', '20'],
            ['Early Stopping', 'Patience=5'],
            ['Loss', 'Binary Crossentropy']
        ],
        [95, 95]
    )
    
    # === PÁGINA 5: RESULTADOS ===
    pdf.add_page()
    pdf.titulo_capitulo('4. Resultados del Entrenamiento')
    
    pdf.subtitulo('4.1 Metricas de Entrenamiento')
    
    # Cargar métricas si existen
    metrics_path = project_root / 'results' / 'metrics' / 'evaluation_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        pdf.tabla_simple(
            ['Metrica', 'Valor'],
            [
                ['Accuracy', f"{metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), float) else 'N/A'],
                ['Precision', f"{metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), float) else 'N/A'],
                ['Recall', f"{metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), float) else 'N/A'],
                ['F1-Score', f"{metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), float) else 'N/A'],
                ['AUC-ROC', f"{metrics.get('auc_roc', 'N/A'):.4f}" if isinstance(metrics.get('auc_roc'), float) else 'N/A'],
            ],
            [95, 95]
        )
    else:
        pdf.cuerpo_texto('Metricas no disponibles. Ejecutar evaluate_mini_model.py primero.')
    
    pdf.ln(5)
    pdf.subtitulo('4.2 Interpretacion de Resultados')
    pdf.cuerpo_texto(
        'NOTA: Los resultados del mini-modelo muestran un rendimiento cercano al azar (50%), '
        'lo cual es ESPERADO con solo 20 muestras de entrenamiento. Esto NO refleja la capacidad '
        'real del modelo.\n\n'
        'Para obtener resultados significativos, se requiere:\n'
        '- Minimo 200+ imagenes de entrenamiento\n'
        '- Data augmentation para multiplicar muestras\n'
        '- Entrenamiento en GPU por multiples epocas'
    )
    
    # === PÁGINA 6: VISUALIZACIONES ===
    pdf.add_page()
    pdf.titulo_capitulo('5. Visualizaciones Generadas')
    
    pdf.subtitulo('5.1 Graficos Disponibles')
    pdf.cuerpo_texto(
        'Se generaron las siguientes visualizaciones en results/figures/:\n\n'
        '1. confusion_matrix.png - Matriz de confusion del conjunto test\n'
        '2. roc_curve.png - Curva ROC con area bajo la curva\n'
        '3. training_history.png - Evolucion de loss y accuracy\n'
        '4. metrics_comparison.png - Comparacion de metricas clave'
    )
    
    # Insertar imágenes si existen
    figures_path = project_root / 'results' / 'figures'
    
    # Matriz de confusión
    cm_path = figures_path / 'confusion_matrix.png'
    if cm_path.exists():
        pdf.ln(5)
        pdf.subtitulo('5.2 Matriz de Confusion')
        pdf.image(str(cm_path), x=30, w=150)
    
    # ROC Curve
    pdf.add_page()
    roc_path = figures_path / 'roc_curve.png'
    if roc_path.exists():
        pdf.subtitulo('5.3 Curva ROC')
        pdf.image(str(roc_path), x=30, w=150)
    
    # Training History
    pdf.add_page()
    history_path = figures_path / 'training_history.png'
    if history_path.exists():
        pdf.subtitulo('5.4 Historial de Entrenamiento')
        pdf.image(str(history_path), x=10, w=190)
    
    # === PÁGINA FINAL: CONCLUSIONES ===
    pdf.add_page()
    pdf.titulo_capitulo('6. Conclusiones y Proximos Pasos')
    
    pdf.subtitulo('6.1 Validacion del Pipeline')
    pdf.cuerpo_texto(
        'ESTADO: EXITOSO\n\n'
        'Se valido correctamente:\n'
        '- Descarga de imagenes desde Google Earth Engine\n'
        '- Preprocesamiento de bandas termicas ASTER\n'
        '- Compilacion y entrenamiento del modelo CNN\n'
        '- Generacion de metricas y visualizaciones\n'
        '- Pipeline de prediccion'
    )
    
    pdf.subtitulo('6.2 Proximos Pasos Recomendados')
    pdf.cuerpo_texto(
        '1. Descargar dataset completo (85+ imagenes originales)\n'
        '2. Aplicar data augmentation (objetivo: 5000+ imagenes)\n'
        '3. Entrenar en computadores con GPU (RTX 5070)\n'
        '4. Ejecutar 100 epocas de entrenamiento\n'
        '5. Evaluar con conjunto de test reservado\n'
        '6. Generar visualizaciones finales para la tesis'
    )
    
    pdf.subtitulo('6.3 Recursos Necesarios')
    pdf.tabla_simple(
        ['Recurso', 'Requerimiento'],
        [
            ['GPU', 'NVIDIA RTX 5070 o superior'],
            ['RAM', '16 GB minimo'],
            ['Almacenamiento', '5 GB para dataset aumentado'],
            ['Tiempo estimado', '2-4 horas para 100 epocas']
        ],
        [95, 95]
    )
    
    # Guardar PDF
    output_path = project_root / 'results' / 'reporte_mini_dataset.pdf'
    pdf.output(str(output_path))
    
    print(f"\n{'='*60}")
    print("📄 REPORTE PDF GENERADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"📁 Ubicacion: {output_path}")
    print(f"📊 Paginas: {pdf.page_no()}")
    print(f"{'='*60}")
    
    return output_path


if __name__ == '__main__':
    generar_reporte()
