"""
Script para visualizar la arquitectura del modelo CNN de geotermia.
Genera diagramas y resúmenes detallados de la arquitectura.

Autores: Cristian Camilo Vega Sánchez, Daniel Santiago Arévalo Rubiano
Universidad de San Buenaventura - Bogotá
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from models.cnn_geotermia import create_geotermia_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow no está instalado. Se generará solo documentación visual.")


class ArchitectureVisualizer:
    """Clase para visualizar la arquitectura del modelo CNN."""
    
    def __init__(self, output_dir: str = 'results/architecture'):
        """
        Inicializa el visualizador de arquitectura.
        
        Args:
            output_dir: Directorio donde guardar las visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colores para diferentes tipos de capas
        self.layer_colors = {
            'input': '#e8f5e9',
            'conv': '#4caf50',
            'pool': '#2196f3',
            'dropout': '#ff9800',
            'dense': '#9c27b0',
            'batch_norm': '#00bcd4',
            'activation': '#f44336',
            'global_pool': '#3f51b5',
            'rescaling': '#8bc34a',
            'add': '#ffc107',
            'output': '#e91e63'
        }
    
    def generate_model_summary(self, model_type: str = 'custom') -> dict:
        """
        Genera un resumen detallado del modelo.
        
        Args:
            model_type: Tipo de modelo ('custom' o 'transfer_learning')
        
        Returns:
            Diccionario con información del modelo
        """
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow no disponible. Generando resumen manual.")
            return self._get_manual_summary(model_type)
        
        # Crear modelo
        model = create_geotermia_model(
            input_shape=(224, 224, 5),
            model_type=model_type,
            num_classes=1
        )
        
        # Capturar resumen del modelo
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        
        # Guardar resumen en archivo de texto
        summary_path = self.output_dir / f'model_summary_{model_type}.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Resumen del Modelo CNN de Geotermia - Tipo: {model_type}\n")
            f.write("=" * 80 + "\n\n")
            f.write('\n'.join(summary_lines))
        
        print(f"✅ Resumen del modelo guardado en: {summary_path}")
        
        # Extraer información de capas
        layers_info = []
        total_params = 0
        trainable_params = 0
        
        for layer in model.layers:
            # Manejar InputLayer que puede no tener output_shape
            try:
                output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else str(model.output_shape)
            except:
                output_shape = "N/A"
            
            layer_config = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': output_shape,
                'params': layer.count_params()
            }
            layers_info.append(layer_config)
            
            total_params += layer.count_params()
            if layer.trainable:
                trainable_params += layer.count_params()
        
        model_info = {
            'model_type': model_type,
            'total_layers': len(model.layers),
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'non_trainable_params': int(total_params - trainable_params),
            'input_shape': (224, 224, 5),
            'output_shape': (1,),
            'layers': layers_info
        }
        
        # Guardar en JSON
        json_path = self.output_dir / f'model_info_{model_type}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Información del modelo guardada en: {json_path}")
        
        return model_info
    
    def _get_manual_summary(self, model_type: str) -> dict:
        """Genera resumen manual cuando TensorFlow no está disponible."""
        if model_type == 'custom':
            layers_info = [
                {'name': 'input', 'type': 'InputLayer', 'output_shape': '(None, 224, 224, 5)', 'params': 0},
                {'name': 'rescaling', 'type': 'Rescaling', 'output_shape': '(None, 224, 224, 5)', 'params': 0},
                {'name': 'conv2d_initial', 'type': 'Conv2D', 'output_shape': '(None, 109, 109, 32)', 'params': 7872},
                {'name': 'batch_norm_initial', 'type': 'BatchNormalization', 'output_shape': '(None, 109, 109, 32)', 'params': 128},
                {'name': 'max_pooling2d', 'type': 'MaxPooling2D', 'output_shape': '(None, 54, 54, 32)', 'params': 0},
                
                {'name': 'residual_block_1_conv1', 'type': 'Conv2D', 'output_shape': '(None, 54, 54, 64)', 'params': 18496},
                {'name': 'residual_block_1_conv2', 'type': 'Conv2D', 'output_shape': '(None, 54, 54, 64)', 'params': 36928},
                {'name': 'residual_block_1_add', 'type': 'Add', 'output_shape': '(None, 54, 54, 64)', 'params': 0},
                {'name': 'max_pooling2d_1', 'type': 'MaxPooling2D', 'output_shape': '(None, 27, 27, 64)', 'params': 0},
                
                {'name': 'residual_block_2', 'type': 'ResidualBlock', 'output_shape': '(None, 13, 13, 128)', 'params': 147584},
                {'name': 'residual_block_3', 'type': 'ResidualBlock', 'output_shape': '(None, 6, 6, 256)', 'params': 590080},
                {'name': 'residual_block_4', 'type': 'ResidualBlock', 'output_shape': '(None, 6, 6, 512)', 'params': 2360320},
                
                {'name': 'global_average_pooling2d', 'type': 'GlobalAveragePooling2D', 'output_shape': '(None, 512)', 'params': 0},
                {'name': 'dense', 'type': 'Dense', 'output_shape': '(None, 256)', 'params': 131328},
                {'name': 'batch_normalization', 'type': 'BatchNormalization', 'output_shape': '(None, 256)', 'params': 1024},
                {'name': 'dropout', 'type': 'Dropout', 'output_shape': '(None, 256)', 'params': 0},
                {'name': 'output', 'type': 'Dense', 'output_shape': '(None, 1)', 'params': 257}
            ]
            
            total_params = sum(layer['params'] for layer in layers_info)
            
            model_info = {
                'model_type': model_type,
                'total_layers': len(layers_info),
                'total_params': total_params,
                'trainable_params': total_params,
                'non_trainable_params': 0,
                'input_shape': (224, 224, 5),
                'output_shape': (1,),
                'layers': layers_info
            }
        else:  # transfer_learning
            model_info = {
                'model_type': model_type,
                'total_layers': 150,  # Aproximado para EfficientNet
                'total_params': 5300000,  # Aproximado
                'trainable_params': 4500000,
                'non_trainable_params': 800000,
                'input_shape': (224, 224, 5),
                'output_shape': (1,),
                'layers': [
                    {'name': 'efficientnetb0', 'type': 'EfficientNetB0', 'output_shape': '(None, 7, 7, 1280)', 'params': 4049564},
                    {'name': 'global_average_pooling', 'type': 'GlobalAveragePooling2D', 'output_shape': '(None, 1280)', 'params': 0},
                    {'name': 'dense', 'type': 'Dense', 'output_shape': '(None, 256)', 'params': 327936},
                    {'name': 'dropout', 'type': 'Dropout', 'output_shape': '(None, 256)', 'params': 0},
                    {'name': 'output', 'type': 'Dense', 'output_shape': '(None, 1)', 'params': 257}
                ]
            }
        
        return model_info
    
    def visualize_architecture_diagram(self, model_type: str = 'custom'):
        """
        Genera un diagrama visual de la arquitectura del modelo.
        
        Args:
            model_type: Tipo de modelo a visualizar
        """
        fig, ax = plt.subplots(figsize=(14, 20), facecolor='white')
        
        # Configuración del diagrama
        y_start = 19
        y_step = 0.8
        x_center = 7
        box_width = 5
        box_height = 0.6
        
        # Título
        ax.text(x_center, y_start + 0.5, 
                f'Arquitectura CNN - {model_type.replace("_", " ").title()}',
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Capas del modelo custom
        if model_type == 'custom':
            layers_config = [
                ('Input\n224×224×5', 'input'),
                ('Rescaling\n[0, 1]', 'rescaling'),
                ('Conv2D (32, 7×7)\nstride=2', 'conv'),
                ('Batch Norm', 'batch_norm'),
                ('ReLU', 'activation'),
                ('MaxPool (3×3)\nstride=2', 'pool'),
                
                ('─── Residual Block 1 (64) ───', None),
                ('Conv2D (64, 3×3)', 'conv'),
                ('Batch Norm + ReLU', 'batch_norm'),
                ('Conv2D (64, 3×3)', 'conv'),
                ('Batch Norm', 'batch_norm'),
                ('Add (Shortcut)', 'add'),
                ('ReLU', 'activation'),
                ('MaxPool (2×2)', 'pool'),
                
                ('─── Residual Block 2 (128) ───', None),
                ('Similar structure\nwith 128 filters', 'conv'),
                ('MaxPool (2×2)', 'pool'),
                
                ('─── Residual Block 3 (256) ───', None),
                ('Similar structure\nwith 256 filters', 'conv'),
                ('MaxPool (2×2)', 'pool'),
                
                ('─── Residual Block 4 (512) ───', None),
                ('Similar structure\nwith 512 filters', 'conv'),
                
                ('Global Average\nPooling', 'global_pool'),
                ('Dense (256)', 'dense'),
                ('Batch Norm + ReLU', 'batch_norm'),
                ('Dropout (0.5)', 'dropout'),
                ('Dense (1)\nSigmoid', 'output'),
                ('Output:\nProbability [0,1]', 'output')
            ]
        else:  # transfer_learning
            layers_config = [
                ('Input\n224×224×5', 'input'),
                ('EfficientNetB0\n(Pre-trained)', 'conv'),
                ('Feature Maps\n7×7×1280', 'conv'),
                ('Global Average\nPooling', 'global_pool'),
                ('Dense (256)', 'dense'),
                ('Batch Norm + ReLU', 'batch_norm'),
                ('Dropout (0.5)', 'dropout'),
                ('Dense (1)\nSigmoid', 'output'),
                ('Output:\nProbability [0,1]', 'output')
            ]
        
        y = y_start
        prev_y = None
        
        for i, (layer_name, layer_type) in enumerate(layers_config):
            if layer_type is None:
                # Es un título de bloque
                ax.text(x_center, y, layer_name,
                       ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='lightgray', 
                                edgecolor='black', linewidth=1))
            else:
                # Es una capa
                color = self.layer_colors.get(layer_type, '#cccccc')
                
                # Dibujar rectángulo de capa
                rect = patches.FancyBboxPatch(
                    (x_center - box_width/2, y - box_height/2),
                    box_width, box_height,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2
                )
                ax.add_patch(rect)
                
                # Texto de la capa
                ax.text(x_center, y, layer_name,
                       ha='center', va='center',
                       fontsize=10, fontweight='bold')
                
                # Conectar con capa anterior
                if prev_y is not None:
                    ax.arrow(x_center, prev_y - box_height/2, 0, 
                            -(y_step - box_height) + 0.1,
                            head_width=0.3, head_length=0.2,
                            fc='black', ec='black')
                
                prev_y = y
            
            y -= y_step
        
        # Leyenda
        legend_y = y - 1
        legend_x_start = 2
        legend_items = [
            ('Input/Output', 'input'),
            ('Convolution', 'conv'),
            ('Pooling', 'pool'),
            ('Dense', 'dense'),
            ('Dropout', 'dropout'),
            ('Batch Norm', 'batch_norm')
        ]
        
        ax.text(legend_x_start, legend_y + 0.5, 'Leyenda:',
               fontsize=12, fontweight='bold')
        
        for i, (label, layer_type) in enumerate(legend_items):
            color = self.layer_colors.get(layer_type, '#cccccc')
            legend_rect = patches.Rectangle(
                (legend_x_start, legend_y - i*0.4 - 0.3),
                0.5, 0.3,
                facecolor=color,
                edgecolor='black'
            )
            ax.add_patch(legend_rect)
            ax.text(legend_x_start + 0.7, legend_y - i*0.4 - 0.15,
                   label, fontsize=10, va='center')
        
        # Configurar ejes
        ax.set_xlim(0, 14)
        ax.set_ylim(legend_y - len(legend_items)*0.4 - 1, y_start + 1)
        ax.axis('off')
        
        # Guardar figura
        output_path = self.output_dir / f'architecture_diagram_{model_type}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Diagrama de arquitectura guardado en: {output_path}")
        plt.close()
    
    def create_latex_table(self, model_info: dict):
        """
        Genera tabla en formato LaTeX para la tesis.
        
        Args:
            model_info: Información del modelo
        """
        latex_path = self.output_dir / f'architecture_table_{model_info["model_type"]}.tex'
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("% Tabla de arquitectura del modelo CNN\n")
            f.write("% Copiar este código en tu documento LaTeX\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Arquitectura del modelo CNN para clasificación de potencial geotérmico}\n")
            f.write("\\label{tab:cnn_architecture}\n")
            f.write("\\begin{tabular}{|l|l|l|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Capa} & \\textbf{Tipo} & \\textbf{Salida} & \\textbf{Parámetros} \\\\\n")
            f.write("\\hline\n")
            
            for layer in model_info['layers'][:10]:  # Primeras 10 capas
                f.write(f"{layer['name']} & {layer['type']} & {layer['output_shape']} & {layer['params']:,} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\multicolumn{3}{|l|}{\\textbf{Total de parámetros}} & ")
            f.write(f"{model_info['total_params']:,} \\\\\n")
            f.write("\\multicolumn{3}{|l|}{\\textbf{Parámetros entrenables}} & ")
            f.write(f"{model_info['trainable_params']:,} \\\\\n")
            f.write("\\multicolumn{3}{|l|}{\\textbf{Parámetros no entrenables}} & ")
            f.write(f"{model_info['non_trainable_params']:,} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"✅ Tabla LaTeX guardada en: {latex_path}")
    
    def create_comparison_table(self):
        """Crea tabla comparativa entre arquitecturas custom y transfer learning."""
        custom_info = self.generate_model_summary('custom')
        transfer_info = self.generate_model_summary('transfer_learning')
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.axis('off')
        
        # Datos de comparación
        comparison_data = [
            ['Característica', 'Custom CNN', 'Transfer Learning (EfficientNet)'],
            ['Total de capas', str(custom_info['total_layers']), str(transfer_info['total_layers'])],
            ['Parámetros totales', f"{custom_info['total_params']:,}", f"{transfer_info['total_params']:,}"],
            ['Parámetros entrenables', f"{custom_info['trainable_params']:,}", f"{transfer_info['trainable_params']:,}"],
            ['Entrada', '224×224×5', '224×224×5'],
            ['Salida', 'Sigmoid (1)', 'Sigmoid (1)'],
            ['Tiempo de entrenamiento', 'Medio', 'Rápido (Fine-tuning)'],
            ['Datos necesarios', '≥200 imágenes', '≥50 imágenes'],
            ['Precisión esperada', '85-90%', '90-95%']
        ]
        
        # Crear tabla
        table = ax.table(cellText=comparison_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.3, 0.35, 0.35])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estilo de la tabla
        for i, row in enumerate(comparison_data):
            for j in range(len(row)):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4caf50')
                    cell.set_text_props(weight='bold', color='white')
                elif i % 2 == 0:
                    cell.set_facecolor('#f5f5f5')
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1.5)
        
        plt.title('Comparación: Custom CNN vs Transfer Learning',
                 fontsize=16, fontweight='bold', pad=20)
        
        output_path = self.output_dir / 'architecture_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Tabla comparativa guardada en: {output_path}")
        plt.close()


def main():
    """Función principal."""
    print("=" * 80)
    print("Visualización de Arquitectura del Modelo CNN de Geotermia")
    print("=" * 80)
    print()
    
    visualizer = ArchitectureVisualizer()
    
    try:
        # 1. Generar resúmenes de modelos
        print("📊 Generando resúmenes de modelos...")
        custom_info = visualizer.generate_model_summary('custom')
        transfer_info = visualizer.generate_model_summary('transfer_learning')
        
        # 2. Generar diagramas visuales
        print("\n🎨 Generando diagramas de arquitectura...")
        visualizer.visualize_architecture_diagram('custom')
        visualizer.visualize_architecture_diagram('transfer_learning')
        
        # 3. Generar tablas LaTeX
        print("\n📝 Generando tablas LaTeX para tesis...")
        visualizer.create_latex_table(custom_info)
        visualizer.create_latex_table(transfer_info)
        
        # 4. Generar tabla comparativa
        print("\n📊 Generando tabla comparativa...")
        visualizer.create_comparison_table()
        
        print("\n" + "=" * 80)
        print("✅ Visualización completada exitosamente!")
        print(f"📁 Archivos guardados en: {visualizer.output_dir}")
        print("=" * 80)
        
        # Resumen de lo generado
        print("\n📄 Archivos generados:")
        for file_path in sorted(visualizer.output_dir.glob('*')):
            print(f"   - {file_path.name}")
    
    except Exception as e:
        print(f"\n❌ Error durante la visualización: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
