# 🚀 Instrucciones para Crear el Repositorio en GitHub

## Paso 1: Crear el Repositorio en GitHub (Web)

1. **Ve a GitHub**: https://github.com/new
2. **Configura el repositorio**:
   - **Nombre**: `g_earth_geotermia-proyect`
   - **Descripción**: `Análisis de potencial geotérmico en Colombia usando Google Earth Engine y Deep Learning`
   - **Visibilidad**: Público o Privado (tu elección)
   - ⚠️ **NO** marques: "Add a README file"
   - ⚠️ **NO** marques: "Add .gitignore"
   - ⚠️ **NO** marques: "Choose a license" (o elige MIT si prefieres)

3. Click en **"Create repository"**

## Paso 2: Conectar tu Repositorio Local

Una vez creado el repositorio en GitHub, GitHub te mostrará las instrucciones. Usa estas:

### Opción A: Si es un nuevo repositorio (recomendado)

```bash
# Ya tienes los commits hechos, solo necesitas conectar y hacer push
cd c:\Users\crsti\proyectos\g_earth_geotermia-proyect

# Configura el remote (reemplaza TU_USUARIO con tu nombre de usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/g_earth_geotermia-proyect.git

# Renombra la rama a main (si no lo está)
git branch -M main

# Sube los cambios
git push -u origin main
```

### Opción B: Usando SSH (si tienes configurado SSH)

```bash
git remote add origin git@github.com:TU_USUARIO/g_earth_geotermia-proyect.git
git branch -M main
git push -u origin main
```

## Paso 3: Verificar

Ve a tu repositorio en GitHub: `https://github.com/TU_USUARIO/g_earth_geotermia-proyect`

¡Deberías ver todos tus archivos!

## 📝 Comando Completo de Ejemplo

Reemplaza `crisveg24` con tu nombre de usuario de GitHub:

```bash
cd c:\Users\crsti\proyectos\g_earth_geotermia-proyect
git remote add origin https://github.com/crisveg24/g_earth_geotermia-proyect.git
git branch -M main
git push -u origin main
```

## 🔑 Si te pide autenticación

GitHub puede pedirte credenciales. Usa un **Personal Access Token** en lugar de tu contraseña:

1. Ve a: https://github.com/settings/tokens
2. Click en "Generate new token" → "Generate new token (classic)"
3. Selecciona los permisos: `repo` (todos los sub-permisos)
4. Copia el token generado
5. Cuando Git te pida la contraseña, pega el token

## 🎉 Estado Actual

✅ Repositorio Git inicializado
✅ 2 commits realizados:
   - Initial commit con el código base
   - Segundo commit con mejoras en documentación
✅ Archivos incluidos:
   - README.md completo
   - requirements.txt
   - .gitignore
   - verificar_config.py
   - main.py
   - descargarimagenes.ipynb
   - etiquetas_imagenesgeotermia.xlsx

## 📂 Archivos Excluidos (.gitignore)

Por el .gitignore, estos archivos NO se subirán:
- ❌ Archivos .tif (imágenes grandes)
- ❌ __pycache__
- ❌ .venv (entorno virtual)
- ❌ .ipynb_checkpoints

Esto es correcto porque las imágenes son muy pesadas para GitHub.

## 🔄 Futuras Actualizaciones

Después de hacer cambios:

```bash
git add .
git commit -m "Descripción de los cambios"
git push
```

## 🆘 Problemas Comunes

### Error: "remote origin already exists"
```bash
git remote remove origin
# Luego vuelve a agregar el remote
git remote add origin https://github.com/TU_USUARIO/g_earth_geotermia-proyect.git
```

### Error: Authentication failed
- Usa un Personal Access Token en lugar de tu contraseña
- O configura SSH: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

---

¡Tu proyecto está listo para ser compartido! 🎉
