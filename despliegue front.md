# Despliegue Frontend — Geotermia CNN Colombia

## Arquitectura del despliegue

```
┌──────────────────────────┐       HTTPS        ┌──────────────────────────┐
│   Frontend (React+Vite)  │  ───────────────►  │  Backend (Flask API)     │
│   Vercel (estático)      │  ◄───────────────  │  Render / Railway / VPS  │
│   vercel.app             │    JSON responses  │  Puerto 5000             │
└──────────────────────────┘                    └──────────────────────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────────┐
                                                 │  Google Earth      │
                                                 │  Engine + ASTER    │
                                                 │  NASA AG100_003    │
                                                 └────────────────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────────┐
                                                 │  Modelo CNN        │
                                                 │  EfficientNetB0    │
                                                 │  .keras (TF 2.21)  │
                                                 └────────────────────┘
```

**Frontend** se despliega como sitio estático en **Vercel** (React + Vite).  
**Backend** (Flask + TensorFlow + GEE) necesita un servidor con Python — se recomienda **Render**, **Railway** o un VPS.

---

## 1. Despliegue del Frontend en Vercel

### Prerrequisitos

- Cuenta en [vercel.com](https://vercel.com)
- Repositorio en GitHub con la rama `front`

### Pasos

1. **Importar proyecto** en Vercel → _"Add New Project"_ → seleccionar el repo.

2. **Configurar el directorio raíz** del frontend:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

3. **Variables de entorno** (Settings → Environment Variables):

   | Variable | Valor (producción) | Descripción |
   |---|---|---|
   | `VITE_API_URL` | `https://api-geotermia.onrender.com` | URL del backend Flask |

4. **Deploy** → Vercel construye y publica automáticamente.

### Archivo `vercel.json`

Ya incluido en `frontend/vercel.json`. Configura:
- **SPA Rewrites**: Todas las rutas redirigen a `index.html` (necesario para React Router).
- **Security Headers**: CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy.

### Dominios personalizados

En Vercel → _Settings_ → _Domains_ se puede agregar un dominio propio.  
Vercel provee HTTPS automáticamente.

---

## 2. Despliegue del Backend (Flask API)

### Opción A: Render (recomendado, tiene free tier)

El archivo `render.yaml` en la raíz del proyecto facilita el despliegue automático.

1. Crear cuenta en [render.com](https://render.com)
2. _New_ → _Blueprint_ → Conectar repo → Render detecta `render.yaml` automáticamente.
   - O crear un _Web Service_ manual con:
     - **Root Directory**: `.` (raíz del proyecto)
     - **Build Command**: `pip install -r requirements-api.txt`
     - **Start Command**: `gunicorn api:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
     - **Python Version**: 3.10.14

3. **Variables de entorno** (en Render → Environment):

   | Variable | Valor | Descripción |
   |---|---|---|
   | `CORS_ORIGINS` | `https://tu-proyecto.vercel.app` | Orígenes permitidos por CORS |
   | `GEE_PROJECT` | `alpine-air-469115-f0` | Proyecto de Google Earth Engine |
   | `GEE_SERVICE_ACCOUNT_KEY` | `{"type":"service_account",...}` | JSON completo de la service account (ver abajo) |
   | `PYTHON_VERSION` | `3.10.14` | Versión de Python |

4. **Autenticación GEE en producción (Service Account)**:
   - Ir a [Google Cloud Console](https://console.cloud.google.com) → IAM → Service Accounts.
   - Crear una service account (o usar la existente del proyecto `alpine-air-469115-f0`).
   - Darle el rol **Earth Engine Resource Viewer** (o Editor).
   - Crear una clave JSON → descargar el archivo `.json`.
   - Copiar **todo el contenido** del JSON y pegarlo como valor de la variable `GEE_SERVICE_ACCOUNT_KEY` en Render.
   - Registrar el email de la service account en [Earth Engine](https://signup.earthengine.google.com/#!/service_accounts).

### Opción B: Railway

Similar a Render. Soporta Python nativamente.

### Variables de entorno del backend

| Variable | Valor | Descripción |
|---|---|---|
| `CORS_ORIGINS` | `https://tu-proyecto.vercel.app` | Orígenes permitidos por CORS |
| `GEE_PROJECT` | `alpine-air-469115-f0` | Proyecto de Google Earth Engine |
| `GEE_SERVICE_ACCOUNT_KEY` | `{...json...}` | JSON de la Service Account de GCP |
| `FLASK_ENV` | `production` | No usar `debug=True` en producción |

---

## 3. Estructura de rutas (React Router)

| Ruta | Página | Descripción |
|---|---|---|
| `/` | `HomePage` | Landing page, estadísticas rápidas del modelo |
| `/prediccion` | `PrediccionPage` | Mapa interactivo + panel de predicción CNN |
| `/metricas` | `MetricasPage` | Métricas reales: accuracy, F1, ROC, confusion matrix |
| `/arquitectura` | `ArquitecturaPage` | Arquitectura CNN, hiperparámetros, fases de entrenamiento |
| `/proyecto` | `ProyectoPage` | Equipo de investigación, objetivos, tecnologías |

Navbar fijo horizontal con glassmorphism. Responsive con menú hamburguesa en móvil.

---

## 4. Cumplimiento OWASP Top 10 (2025)

### A01 – Broken Access Control
- La API no tiene endpoints protegidos que requieran autenticación (es de consulta pública).
- `X-Frame-Options: DENY` previene clickjacking.
- CORS restringido solo a orígenes permitidos (`CORS_ORIGINS`).

### A02 – Cryptographic Failures
- Todo el tráfico va por HTTPS (Vercel y Render lo proveen automáticamente).
- No se almacenan datos sensibles del usuario.
- No hay cookies ni tokens de sesión.

### A03 – Injection
- Validación estricta de coordenadas en `api.py`:
  - `lat` y `lon` se parsean como `float` (no se concatenan a queries).
  - Límites geográficos de Colombia: lat ∈ [-5, 14], lon ∈ [-82, -66].
- No hay SQL, shell commands ni templates dinámicos.
- CSP impide ejecución de scripts inline no autorizados.

### A04 – Insecure Design
- El modelo CNN es read-only; no se puede modificar desde la API.
- Fallback a proximidad documentado y transparente (`metodo` en respuesta).
- Rate limiting implementado (30 req/min por IP).

### A05 – Security Misconfiguration
- Headers de seguridad en ambos lados (Vercel `vercel.json` + Flask `@after_request`).
- `debug=False` en producción.
- No se exponen stack traces ni mensajes internos al usuario.

### A06 – Vulnerable and Outdated Components
- Dependencias con versiones fijas en `requirements.txt` y `package.json`.
- Auditar periódicamente: `npm audit` y `pip audit`.

### A07 – Identification and Authentication Failures
- No aplica: la API es pública y no maneja cuentas de usuario.

### A08 – Software and Data Integrity Failures
- Vercel verifica integridad del build desde GitHub.
- El modelo `.keras` se carga localmente (no se descarga en runtime).

### A09 – Security Logging and Monitoring Failures
- `logging` de Python registra eventos clave: carga de modelo, errores GEE, errores de predicción.
- Rate limit logging implícito.
- En producción: conectar a un servicio de monitoreo (Sentry, Datadog, etc.).

### A10 – Server-Side Request Forgery (SSRF)
- Las únicas peticiones externas son a Google Earth Engine con coordenadas validadas.
- No se aceptan URLs como input del usuario.
- Los límites geográficos impiden solicitar imágenes fuera de Colombia.

---

## 5. Variables de entorno — Resumen

### Frontend (`frontend/.env`)

```env
VITE_API_URL=http://127.0.0.1:5000          # desarrollo
# VITE_API_URL=https://api-geotermia.onrender.com  # producción
```

### Backend (entorno del servidor)

```env
CORS_ORIGINS=https://tu-proyecto.vercel.app
GEE_PROJECT=alpine-air-469115-f0
FLASK_ENV=production
```

---

## 6. Comandos útiles

```bash
# --- Frontend ---
cd frontend
npm install              # Instalar dependencias
npm run dev              # Servidor de desarrollo (localhost:5173)
npm run build            # Build de producción → dist/
npx vercel               # Deploy manual a Vercel

# --- Backend ---
cd ..                    # Raíz del proyecto
pip install -r requirements.txt
python api.py            # Iniciar API en localhost:5000

# --- Auditorías de seguridad ---
cd frontend && npm audit
pip audit                # Requiere pip-audit instalado
```

---

## 7. Checklist pre-deploy

- [ ] `VITE_API_URL` apunta a la URL de producción del backend
- [ ] `CORS_ORIGINS` incluye la URL de Vercel
- [ ] Google Earth Engine autenticado en el servidor backend
- [ ] Modelo `.keras` disponible en `models/saved_models/`
- [ ] `npm run build` construye sin errores
- [ ] `npm audit` sin vulnerabilidades críticas
- [ ] Probar `/predict` con coordenadas válidas e inválidas
- [ ] Probar `/health` para verificar estado del modelo
- [ ] Verificar headers de seguridad con [securityheaders.com](https://securityheaders.com)

---

## 8. Endpoints de la API

| Método | Ruta | Descripción | Body |
|---|---|---|---|
| `POST` | `/predict` | Predicción geotérmica | `{"lat": 4.89, "lon": -75.32}` |
| `GET` | `/zonas` | Lista de zonas geotérmicas conocidas | — |
| `GET` | `/health` | Estado del servidor y modelo cargado | — |

### Respuesta de `/predict`

```json
{
  "porcentaje": 87.3,
  "zona_cercana": "Nevado del Ruiz",
  "distancia_km": 12.45,
  "metodo": "cnn",
  "modelo": "geotermia_v7_phase2_best.keras",
  "tiempos": {
    "descarga": 4.21,
    "prediccion": 0.83,
    "total": 5.12
  }
}
```

Si la CNN no está disponible, `metodo` será `"proximidad"` y no incluirá `tiempos`.

---

*Documento generado para el proyecto de grado — Universidad de San Buenaventura, Bogotá.*
