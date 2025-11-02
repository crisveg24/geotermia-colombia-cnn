@echo off
echo ============================================================
echo   Script de Ayuda para Crear Repositorio en GitHub
echo ============================================================
echo.
echo PASO 1: Abre tu navegador y ve a: https://github.com/new
echo.
echo PASO 2: Configura el repositorio:
echo   - Nombre: g_earth_geotermia-proyect
echo   - Descripcion: Analisis de potencial geotermico en Colombia
echo   - NO marques ninguna opcion adicional
echo   - Click en "Create repository"
echo.
echo PASO 3: Copia tu nombre de usuario de GitHub
set /p username="Ingresa tu nombre de usuario de GitHub: "
echo.
echo ============================================================
echo   Comandos para ejecutar:
echo ============================================================
echo.
echo git remote add origin https://github.com/%username%/g_earth_geotermia-proyect.git
echo git branch -M main
echo git push -u origin main
echo.
echo ============================================================
echo Copiando comandos al portapapeles...
echo.
echo git remote add origin https://github.com/%username%/g_earth_geotermia-proyect.git ^&^& git branch -M main ^&^& git push -u origin main | clip
echo.
echo ✅ Comandos copiados! Pegalos en la terminal de PowerShell
echo.
pause
