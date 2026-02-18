"""
Generador de Reporte PDF - Dataset Completo
=============================================

Genera un reporte PDF profesional con los resultados del entrenamiento
completo del modelo CNN para identificacion de potencial geotermico.

Autores: Cristian Camilo Vega Sanchez, Daniel Santiago Arevalo Rubiano,
         Yuliet Katerin Espitia Ayala, Laura Sophie Rivera Martin
Asesor: Prof. Yeison Eduardo Conejo Sandoval
Universidad de San Buenaventura - Bogota
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpdf import FPDF


class ReportePDF(FPDF):
    """Clase personalizada para el reporte PDF."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Encabezado de cada pagina."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'CNN Geotermia Colombia - Universidad de San Buenaventura', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Pie de pagina."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def titulo_capitulo(self, titulo):
        """Titulo de capitulo."""
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 102, 204)
        self.cell(0, 10, titulo, 0, 1, 'L')
        self.ln(4)

    def subtitulo(self, texto):
        """Subtitulo."""
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
    """Genera el reporte PDF del dataset completo."""

    project_root = Path(__file__).parent.parent

    # Cargar metricas reales
    metrics_path = project_root / 'results' / 'metrics' / 'evaluation_metrics.json'
    history_path = project_root / 'logs' / 'history_custom.json'

    metrics = {}
    history = {}

    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

    total_epochs = len(history.get('loss', []))
    best_val_loss = min(history.get('val_loss', [999])) if history.get('val_loss') else 'N/A'
    best_epoch = history.get('val_loss', []).index(best_val_loss) + 1 if isinstance(best_val_loss, float) else 'N/A'
    best_val_acc = max(history.get('val_accuracy', [0])) if history.get('val_accuracy') else 'N/A'

    # Crear PDF
    pdf = ReportePDF()
    pdf.add_page()

    # === PORTADA ===
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(0, 102, 204)
    pdf.ln(30)
    pdf.cell(0, 15, 'REPORTE DE RESULTADOS', 0, 1, 'C')
    pdf.cell(0, 15, 'ENTRENAMIENTO COMPLETO', 0, 1, 'C')

    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, 'Modelo CNN para Identificacion de', 0, 1, 'C')
    pdf.cell(0, 8, 'Zonas con Potencial Geotermico en Colombia', 0, 1, 'C')
    pdf.cell(0, 8, 'Usando Imagenes Satelitales ASTER', 0, 1, 'C')

    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 12)
    pdf.cell(0, 8, 'Cristian Camilo Vega Sanchez', 0, 1, 'C')
    pdf.cell(0, 8, 'Daniel Santiago Arevalo Rubiano', 0, 1, 'C')
    pdf.cell(0, 8, 'Yuliet Katerin Espitia Ayala', 0, 1, 'C')
    pdf.cell(0, 8, 'Laura Sophie Rivera Martin', 0, 1, 'C')

    pdf.ln(10)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Universidad de San Buenaventura - Bogota', 0, 1, 'C')
    pdf.cell(0, 8, 'Ingenieria de Sistemas', 0, 1, 'C')
    pdf.cell(0, 8, f'Fecha: {datetime.now().strftime("%d/%m/%Y")}', 0, 1, 'C')

    # === PAGINA 2: INTRODUCCION ===
    pdf.add_page()
    pdf.titulo_capitulo('1. Introduccion')

    pdf.cuerpo_texto(
        'Este documento presenta los resultados del entrenamiento completo del modelo CNN '
        '(Red Neuronal Convolucional) disenado para identificar zonas con potencial geotermico '
        'en Colombia a partir de imagenes satelitales ASTER (Advanced Spaceborne Thermal '
        'Emission and Reflection Radiometer).'
    )

    pdf.cuerpo_texto(
        'El modelo fue entrenado utilizando 85 imagenes ASTER originales de zonas geotermicas '
        'y zonas de control en Colombia, las cuales fueron aumentadas a 2,635 imagenes mediante '
        'tecnicas de data augmentation. El entrenamiento se realizo utilizando las 5 bandas '
        'termicas de ASTER (TIR: bandas 10-14), que capturan informacion termica en el rango '
        'de 8.125 a 11.65 micrometros.'
    )

    pdf.subtitulo('1.1 Objetivo')
    pdf.cuerpo_texto(
        'Desarrollar y evaluar un modelo de clasificacion binaria basado en deep learning capaz '
        'de distinguir entre zonas con potencial geotermico y zonas sin actividad geotermica '
        'significativa, utilizando exclusivamente datos termicos satelitales.'
    )

    # === PAGINA 3: DATASET ===
    pdf.add_page()
    pdf.titulo_capitulo('2. Dataset')

    pdf.subtitulo('2.1 Datos Originales')
    pdf.tabla_simple(
        ['Categoria', 'Cantidad', 'Descripcion'],
        [
            ['Positivas (Geotermicas)', '45', 'Zonas volcanes/termales activos'],
            ['Negativas (Control)', '40', 'Zonas sin actividad geotermica'],
            ['Total Original', '85', 'Imagenes ASTER descargadas via GEE'],
        ],
        [65, 40, 85]
    )

    pdf.ln(5)
    pdf.subtitulo('2.2 Data Augmentation')
    pdf.cuerpo_texto(
        'Se aplicaron 30 tecnicas de augmentation por imagen incluyendo:\n'
        '- Rotaciones (90, 180, 270 grados)\n'
        '- Reflexiones horizontal y vertical\n'
        '- Adicion de ruido gaussiano\n'
        '- Ajuste de brillo y contraste\n'
        '- Zoom y desplazamiento\n'
        '- Combinaciones de transformaciones'
    )

    pdf.tabla_simple(
        ['Conjunto', 'Positivas', 'Negativas', 'Total'],
        [
            ['Post-Augmentation', '1,395', '1,240', '2,635'],
            ['Entrenamiento (70%)', '~977', '~866', '1,843'],
            ['Validacion (15%)', '~209', '~187', '396'],
            ['Test (15%)', '~209', '~187', '396'],
        ],
        [55, 45, 45, 45]
    )

    pdf.ln(5)
    pdf.subtitulo('2.3 Especificaciones de Entrada')
    pdf.tabla_simple(
        ['Parametro', 'Valor'],
        [
            ['Dimension de imagen', '224 x 224 pixeles'],
            ['Numero de bandas', '5 (TIR ASTER: bandas 10-14)'],
            ['Tipo de dato', 'Float32 normalizado'],
            ['Fuente', 'Google Earth Engine (ASTER L1T)'],
        ],
        [95, 95]
    )

    # === PAGINA 4: ZONAS GEOGRAFICAS ===
    pdf.add_page()
    pdf.titulo_capitulo('3. Zonas Geograficas de Estudio')

    pdf.subtitulo('3.1 Zonas Geotermicas (Clase Positiva)')
    pdf.cuerpo_texto(
        'Se seleccionaron 45 muestreos de las principales zonas geotermicas de Colombia:\n\n'
        '- Complejo Volcanico Nevado del Ruiz (Tolima/Caldas)\n'
        '- Nevado del Tolima (Tolima)\n'
        '- Volcan Purace - Sistema hidrotermal (Cauca)\n'
        '- Volcan Galeras (Narino)\n'
        '- Volcan Cumbal (Narino)\n'
        '- Volcan Sotara (Cauca)\n'
        '- Volcan Azufral - Lago craterico (Narino)\n'
        '- Campo geotermico Paipa-Iza (Boyaca)\n'
        '- Termales Santa Rosa de Cabal (Risaralda)\n'
        '- Zona termal Manizales (Caldas)\n'
        '- Y otras zonas con manifestaciones termales'
    )

    pdf.subtitulo('3.2 Zonas Control (Clase Negativa)')
    pdf.cuerpo_texto(
        'Para la clase negativa se tomaron 40 muestreos de regiones sin actividad geotermica:\n\n'
        '- Llanos Orientales (Meta, Casanare, Arauca)\n'
        '- Amazonia colombiana (Caqueta, Guaviare, Amazonas)\n'
        '- Costa Caribe (Atlantico, Magdalena, Cesar)\n'
        '- Orinoquia (Vichada)\n'
        '- Altiplano Cundiboyacense (areas no termales)'
    )

    # === PAGINA 5: ARQUITECTURA ===
    pdf.add_page()
    pdf.titulo_capitulo('4. Arquitectura del Modelo')

    pdf.subtitulo('4.1 Arquitectura CNN (ResNet-Inspired)')
    pdf.cuerpo_texto(
        'El modelo utiliza una arquitectura inspirada en ResNet con conexiones residuales:\n\n'
        '- Input Layer: 224 x 224 x 5\n'
        '- Rescaling (normalizacion 0-1)\n'
        '- Bloque Inicial: Conv2D 32 filtros (7x7) + BN + ReLU + SpatialDropout\n'
        '- Bloque Residual 1: 64 filtros + shortcut connection + MaxPool\n'
        '- Bloque Residual 2: 128 filtros + shortcut connection + MaxPool\n'
        '- Bloque Residual 3: 256 filtros + shortcut connection + MaxPool\n'
        '- Bloque Residual 4: 512 filtros + shortcut connection\n'
        '- Global Average Pooling 2D\n'
        '- Dense 256 + BatchNorm + ReLU + Dropout(0.5)\n'
        '- Output: Dense 1 + Sigmoid'
    )

    pdf.subtitulo('4.2 Parametros del Modelo')
    pdf.tabla_simple(
        ['Parametro', 'Valor'],
        [
            ['Total parametros', '5,025,409'],
            ['Parametros entrenables', '5,020,993'],
            ['Parametros no entrenables', '4,416'],
            ['Tamano del modelo', '57.69 MB'],
        ],
        [95, 95]
    )

    pdf.ln(5)
    pdf.subtitulo('4.3 Configuracion de Entrenamiento')
    pdf.tabla_simple(
        ['Parametro', 'Valor'],
        [
            ['Optimizador', 'AdamW (weight decay = 1e-4)'],
            ['Learning Rate inicial', '0.001'],
            ['Batch Size', '32'],
            ['Epocas Maximas', '100'],
            ['Early Stopping', 'Patience = 15 (monitor: val_loss)'],
            ['ReduceLROnPlateau', 'Factor 0.5, patience 5'],
            ['Loss Function', 'Binary Crossentropy (label smooth 0.1)'],
            ['Data Augmentation', 'RandomFlip, Rotation, Zoom, Translation'],
            ['Regularizacion', 'L2 (1e-4) + Dropout (0.5) + SpatialDropout'],
        ],
        [80, 110]
    )

    # === PAGINA 6: RESULTADOS ===
    pdf.add_page()
    pdf.titulo_capitulo('5. Resultados del Entrenamiento')

    pdf.subtitulo('5.1 Resumen de Entrenamiento')

    if total_epochs > 0:
        pdf.tabla_simple(
            ['Parametro', 'Valor'],
            [
                ['Epocas ejecutadas', str(total_epochs)],
                ['Mejor epoca (best weights)', str(best_epoch)],
                ['Mejor val_loss', f'{best_val_loss:.4f}' if isinstance(best_val_loss, float) else 'N/A'],
                ['Mejor val_accuracy', f'{best_val_acc:.4f}' if isinstance(best_val_acc, float) else 'N/A'],
                ['Motivo de parada', f'EarlyStopping (sin mejora en 15 epocas)'],
                ['Hardware', 'Intel i5-10300H (CPU), 12 GB RAM'],
                ['Tiempo aprox.', '~35 minutos (23 epocas x ~90 s/epoca)'],
            ],
            [80, 110]
        )
    else:
        pdf.cuerpo_texto('No se encontro historial de entrenamiento.')

    pdf.ln(5)
    pdf.subtitulo('5.2 Metricas de Evaluacion (Conjunto de Test)')

    if metrics:
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        roc = metrics.get('roc_auc', 0)
        r2 = metrics.get('r2_score', 0)

        pdf.tabla_simple(
            ['Metrica', 'Valor', 'Porcentaje'],
            [
                ['Accuracy', f'{acc:.4f}', f'{acc*100:.2f}%'],
                ['Precision', f'{prec:.4f}', f'{prec*100:.2f}%'],
                ['Recall (Sensibilidad)', f'{rec:.4f}', f'{rec*100:.2f}%'],
                ['F1-Score', f'{f1:.4f}', f'{f1*100:.2f}%'],
                ['ROC AUC', f'{roc:.4f}', f'{roc*100:.2f}%'],
                ['R2 Score', f'{r2:.4f}', f'{r2*100:.2f}%'],
            ],
            [70, 60, 60]
        )
    else:
        pdf.cuerpo_texto('Metricas no disponibles. Ejecutar evaluate_model.py primero.')

    pdf.ln(5)
    pdf.subtitulo('5.3 Matriz de Confusion')

    cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    if cm and len(cm) == 2:
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        pdf.tabla_simple(
            ['', 'Pred: Sin Potencial', 'Pred: Con Potencial'],
            [
                ['Real: Sin Potencial', f'TN = {tn}', f'FP = {fp}'],
                ['Real: Con Potencial', f'FN = {fn}', f'TP = {tp}'],
            ],
            [60, 65, 65]
        )

    pdf.ln(5)
    pdf.subtitulo('5.4 Interpretacion de Resultados')
    pdf.cuerpo_texto(
        f'El modelo alcanzo una precision (Precision) del {prec*100:.1f}%, lo que indica que '
        f'cuando predice que una zona tiene potencial geotermico, acierta en {prec*100:.1f}% de '
        f'los casos. Sin embargo, el recall de {rec*100:.1f}% indica que hay zonas geotermicas '
        f'que el modelo no logra identificar (falsos negativos).\n\n'
        f'El ROC AUC de {roc:.4f} indica una capacidad discriminativa significativa: el modelo '
        f'distingue correctamente entre zonas geotermicas y no geotermicas en el {roc*100:.1f}% '
        f'de los casos cuando se varian los umbrales de decision.\n\n'
        f'Se observa overfitting moderado (train accuracy ~92% vs test accuracy {acc*100:.1f}%), '
        f'lo cual es esperado dado que las 2,635 imagenes de entrenamiento provienen de solo 85 '
        f'imagenes ASTER originales mediante augmentation. Con un dataset mas diverso, se espera '
        f'que la brecha disminuya.'
    )

    # === PAGINA 7: VISUALIZACIONES ===
    pdf.add_page()
    pdf.titulo_capitulo('6. Visualizaciones')

    figures_path = project_root / 'results' / 'figures'

    # Training History
    history_fig = figures_path / 'training_history.png'
    if history_fig.exists():
        pdf.subtitulo('6.1 Historial de Entrenamiento')
        pdf.image(str(history_fig), x=10, w=190)

    # Confusion Matrix
    pdf.add_page()
    cm_fig = figures_path / 'confusion_matrix.png'
    if cm_fig.exists():
        pdf.subtitulo('6.2 Matriz de Confusion')
        pdf.image(str(cm_fig), x=30, w=150)

    # ROC Curve
    pdf.add_page()
    roc_fig = figures_path / 'roc_curve.png'
    if roc_fig.exists():
        pdf.subtitulo('6.3 Curva ROC')
        pdf.image(str(roc_fig), x=30, w=150)

    # Metrics Comparison
    pdf.add_page()
    comp_fig = figures_path / 'metrics_comparison.png'
    if comp_fig.exists():
        pdf.subtitulo('6.4 Comparacion de Metricas')
        pdf.image(str(comp_fig), x=30, w=150)

    # === PAGINA FINAL: CONCLUSIONES ===
    pdf.add_page()
    pdf.titulo_capitulo('7. Conclusiones')

    pdf.subtitulo('7.1 Logros Alcanzados')
    pdf.cuerpo_texto(
        '1. Se implemento exitosamente un pipeline completo de deep learning para '
        'clasificacion de imagenes satelitales ASTER.\n\n'
        '2. Se disenouna arquitectura CNN inspirada en ResNet con 5 millones de parametros '
        'optimizada para imagenes multiespectrales de 5 bandas.\n\n'
        '3. Se logro un ROC AUC de 0.82, demostrando que el modelo tiene capacidad '
        'discriminativa real para distinguir zonas geotermicas.\n\n'
        '4. Se automatizo la descarga de datos desde Google Earth Engine, el augmentation '
        'y el pipeline completo de entrenamiento/evaluacion.'
    )

    pdf.subtitulo('7.2 Limitaciones')
    pdf.cuerpo_texto(
        '- El dataset original de 85 imagenes es limitado. Data augmentation ayuda pero '
        'no reemplaza la diversidad de datos reales.\n'
        '- El entrenamiento se ejecuto en CPU (Intel i5-10300H) en lugar de GPU, lo cual '
        'limito el numero de epocas y experimentos posibles.\n'
        '- El overfitting observado (92% train vs 68% test) requiere mas datos originales '
        'para ser mitigado de manera efectiva.'
    )

    pdf.subtitulo('7.3 Trabajo Futuro')
    pdf.cuerpo_texto(
        '- Ampliar el dataset con imagenes de mas zonas geotermicas de Colombia y el mundo.\n'
        '- Experimentar con transfer learning usando modelos pre-entrenados.\n'
        '- Entrenar con GPU dedicada para explorar mas hiperparametros.\n'
        '- Integrar bandas SWIR de ASTER para mayor discriminacion espectral.\n'
        '- Desarrollar la interfaz web (Streamlit) para prediccion en tiempo real.'
    )

    # Guardar PDF
    output_path = project_root / 'results' / 'reporte_entrenamiento_completo.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))

    print(f"\n{'='*60}")
    print("REPORTE PDF GENERADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"Ubicacion: {output_path}")
    print(f"Paginas: {pdf.page_no()}")
    print(f"{'='*60}")

    return output_path


if __name__ == '__main__':
    generar_reporte()
