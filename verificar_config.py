"""
Script para verificar la configuración de Google Earth Engine
y guiar al usuario en el proceso de autenticación.
"""
import sys

def check_earthengine():
    """Verifica si Earth Engine está instalado y autenticado"""
    print("🔍 Verificando instalación de Google Earth Engine...\n")
    
    # Verificar instalación
    try:
        import ee
        print("✅ earthengine-api está instalado correctamente")
    except ImportError:
        print("❌ earthengine-api no está instalado")
        print("   Ejecuta: pip install earthengine-api geemap")
        return False
    
    # Verificar autenticación
    print("\n🔐 Verificando autenticación...")
    try:
        ee.Initialize()
        print("✅ Google Earth Engine está autenticado y listo para usar")
        print("\n🎉 ¡Todo configurado correctamente!")
        return True
    except Exception as e:
        print("❌ No estás autenticado en Google Earth Engine")
        print("\n📝 Para autenticarte, ejecuta uno de estos comandos:")
        print("\n   Opción 1 (Python):")
        print("   python -c \"import ee; ee.Authenticate()\"")
        print("\n   Opción 2 (CLI):")
        print("   earthengine authenticate")
        print("\n   Luego configura tu proyecto:")
        print("   earthengine set_project TU-PROYECTO-ID")
        print(f"\n   Error específico: {str(e)}")
        return False

def test_visualization():
    """Prueba básica de visualización"""
    print("\n🗺️ Probando visualización básica...")
    try:
        import ee
        import geemap
        
        ee.Initialize()
        
        # Cargar imagen ASTER
        aster = ee.Image("NASA/ASTER_GED/AG100_003")
        
        # Definir región de interés (Volcán Galeras)
        roi = ee.Geometry.Point([-77.36, 1.22]).buffer(5000)
        
        print("✅ Imagen ASTER cargada correctamente")
        print("✅ Región de interés definida: Volcán Galeras")
        print("\n💡 Para visualizar el mapa, usa el notebook: descargarimagenes.ipynb")
        print("   O ejecuta: jupyter notebook descargarimagenes.ipynb")
        
        return True
    except Exception as e:
        print(f"❌ Error en la visualización: {str(e)}")
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("   Verificador de Configuración - Proyecto Geotermia")
    print("=" * 60)
    print()
    
    # Verificar Earth Engine
    if check_earthengine():
        # Si está autenticado, probar visualización
        test_visualization()
    
    print("\n" + "=" * 60)
    print("📚 Para más información, consulta el README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
