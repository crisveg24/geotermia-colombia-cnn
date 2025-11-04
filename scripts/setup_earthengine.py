"""
Script para configurar Google Earth Engine
Autor: Cristian Camilo Vega Sánchez
"""

import ee

print("="*80)
print("CONFIGURACIÓN DE GOOGLE EARTH ENGINE")
print("="*80)

try:
    # Intentar autenticar
    print("\n🔐 Iniciando proceso de autenticación...")
    print("Se abrirá una ventana del navegador para autorizar el acceso.")
    print("Por favor, inicia sesión con tu cuenta de Google.")
    
    ee.Authenticate()
    
    print("\n✅ Autenticación completada exitosamente!")
    print("\nAhora puedes ejecutar el script de descarga:")
    print("  python scripts/download_dataset.py")
    
except Exception as e:
    print(f"\n❌ Error durante la autenticación: {e}")
    print("\nSi ya estás autenticado, intenta ejecutar directamente:")
    print("  python scripts/download_dataset.py")
