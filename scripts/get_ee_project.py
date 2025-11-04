"""
Script para obtener el proyecto de Earth Engine configurado
"""
import ee

try:
    # Inicializar Earth Engine
    ee.Initialize()
    print("✅ Earth Engine inicializado correctamente")
    print(f"Proyecto activo: {ee.data.getProjectID() if hasattr(ee.data, 'getProjectID') else 'No disponible'}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nIntenta configurar el proyecto manualmente:")
    print("earthengine set_project YOUR-PROJECT-ID")
