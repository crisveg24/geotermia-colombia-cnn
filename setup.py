"""
Script de Configuración Automatizada
Proyecto: Análisis de Potencial Geotérmico en Colombia

Este script automatiza la configuración del entorno de desarrollo.
"""
import subprocess
import sys
import os

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(number, text):
    """Imprime un paso numerado"""
    print(f"📌 Paso {number}: {text}")

def run_command(command, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n⚙️  {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"   Detalles: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    print_step(1, "Verificando versión de Python")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro} detectado")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Versión de Python compatible")
        return True
    else:
        print("❌ Se requiere Python 3.8 o superior")
        return False

def install_requirements():
    """Instala las dependencias del proyecto"""
    print_step(2, "Instalando dependencias")
    
    if not os.path.exists("requirements.txt"):
        print("❌ Archivo requirements.txt no encontrado")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Instalación de dependencias"
    )

def verify_earthengine():
    """Verifica la instalación de Earth Engine"""
    print_step(3, "Verificando Google Earth Engine")
    
    try:
        import ee
        print("✅ earthengine-api está instalado")
        return True
    except ImportError:
        print("❌ earthengine-api no está instalado")
        return False

def authenticate_earthengine():
    """Guía para autenticar Earth Engine"""
    print_step(4, "Configuración de Google Earth Engine")
    
    try:
        import ee
        ee.Initialize()
        print("✅ Google Earth Engine ya está autenticado")
        return True
    except:
        print("\n⚠️  Autenticación necesaria")
        print("\n   Para autenticarte, ejecuta en Python:")
        print("   >>> import ee")
        print("   >>> ee.Authenticate()")
        print("\n   O en la terminal:")
        print("   $ earthengine authenticate")
        return False

def main():
    """Función principal de configuración"""
    print_header("Configuración del Proyecto - Análisis Geotérmico Colombia")
    
    # Paso 1: Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Paso 2: Instalar dependencias
    print("\n")
    if not install_requirements():
        print("\n⚠️  Error al instalar dependencias")
        sys.exit(1)
    
    # Paso 3: Verificar Earth Engine
    print("\n")
    if not verify_earthengine():
        print("\n⚠️  Ejecuta: pip install earthengine-api")
        sys.exit(1)
    
    # Paso 4: Verificar autenticación
    print("\n")
    authenticated = authenticate_earthengine()
    
    # Resumen final
    print_header("Resumen de Configuración")
    print("✅ Python configurado")
    print("✅ Dependencias instaladas")
    print("✅ Earth Engine instalado")
    
    if authenticated:
        print("✅ Earth Engine autenticado")
        print("\n🎉 ¡Configuración completada con éxito!")
        print("\n📝 Siguiente paso:")
        print("   - Ejecuta: jupyter notebook descargarimagenes.ipynb")
    else:
        print("⚠️  Earth Engine requiere autenticación")
        print("\n📝 Siguientes pasos:")
        print("   1. Autentica Earth Engine (ver instrucciones arriba)")
        print("   2. Ejecuta este script nuevamente")
        print("   3. Luego: jupyter notebook descargarimagenes.ipynb")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
