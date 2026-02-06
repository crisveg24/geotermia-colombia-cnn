# 📊 GUÍA DE MONITOREO DEL ENTRENAMIENTO

**Fecha de inicio:** 3 de noviembre de 2025 - 18:50  
**Estado:** ✅ Entrenamiento en progreso  
**Terminal ID:** f0d3a017-04e8-4240-b69c-c8ab613413c8  
**Corrección aplicada:** Rutas absolutas basadas en ubicación del script

---

## 🎯 INFORMACIÓN DEL ENTRENAMIENTO

### Configuración Actual
```yaml
Script: scripts/train_model.py
Dataset: 5,518 imágenes ASTER procesadas
Training Set: 3,862 imágenes (70%)
Validation Set: 828 imágenes (15%)
Test Set: 828 imágenes (15%)

Modelo:
  Arquitectura: CNN personalizada ResNet-inspired
  Capas: 52
  Parámetros: 5,025,409
  Input Shape: (224, 224, 5)
  Output: Clasificación binaria

Hardware:
  Modo: CPU con optimizaciones oneDNN
  Precision: Mixed precision (float16/float32)
  Python: 3.10.11
  TensorFlow: 2.20.0
```

### Hiperparámetros
```python
batch_size = 32
epochs = 100  # Con EarlyStopping
learning_rate = 0.001
optimizer = Adam
loss = binary_crossentropy
class_weights = {0: 2.2247, 1: 0.6450}
```

---

## ⏱️ TIEMPO ESTIMADO

### Cálculos Basados en Dataset
```
Imágenes por época: 3,862 training + 828 validation
Batch size: 32
Steps por época: ~120 training + ~26 validation

Tiempo estimado por época: 1-2 minutos en CPU
Épocas esperadas: 30-50 (con EarlyStopping)
Tiempo total estimado: 2-3 horas
```

### Hitos Esperados
- ⏱️ **20 min:** ~10 épocas completadas
- ⏱️ **40 min:** ~20 épocas completadas
- ⏱️ **1 hora:** ~30 épocas completadas
- ⏱️ **2 horas:** ~50-70 épocas completadas
- ⏱️ **2.5-3 horas:** Entrenamiento completo (EarlyStopping)

---

## 📈 CÓMO MONITOREAR EL PROGRESO

### Opción 1: Salida del Terminal (VS Code)
La salida del terminal mostrará:
```
Epoch 1/100
120/120 [==============================] - 85s 710ms/step - loss: 0.6543 - accuracy: 0.7234 - val_loss: 0.5432 - val_accuracy: 0.7823
Epoch 2/100
120/120 [==============================] - 82s 683ms/step - loss: 0.5234 - accuracy: 0.7856 - val_loss: 0.4987 - val_accuracy: 0.8145
...
```

**Métricas a observar:**
- `loss`: Error en training (debe disminuir)
- `accuracy`: Precisión en training (debe aumentar)
- `val_loss`: Error en validation (debe disminuir y ser similar a loss)
- `val_accuracy`: Precisión en validation (debe aumentar)

**Señales de buen entrenamiento:**
- ✅ Loss disminuye constantemente
- ✅ Accuracy aumenta gradualmente
- ✅ val_loss similar a loss (no overfitting)
- ✅ val_accuracy cercana a accuracy

**Señales de problemas:**
- ⚠️ val_loss aumenta mientras loss disminuye → OVERFITTING
- ⚠️ Loss no disminuye → Learning rate muy bajo o modelo estancado
- ⚠️ Loss explota (NaN) → Learning rate muy alto

### Opción 2: TensorBoard (Recomendado)

**Iniciar TensorBoard:**
```powershell
# En una nueva terminal (Ctrl+Shift+`)
cd C:\Users\crsti\proyectos\g_earth_geotermia-proyect
C:/Users/crsti/proyectos/.venv/Scripts/python.exe -m tensorboard --logdir=logs/tensorboard
```

**Acceder:**
- Abre navegador en: http://localhost:6006
- Se actualizará automáticamente cada 30 segundos

**Visualizaciones disponibles:**
- **SCALARS:** Gráficos de loss y accuracy en tiempo real
- **GRAPHS:** Arquitectura del modelo
- **DISTRIBUTIONS:** Distribución de pesos
- **HISTOGRAMS:** Histogramas de activaciones

### Opción 3: Archivo CSV de Historial

El entrenamiento genera un archivo CSV en tiempo real:
```
models/training_history.csv
```

Puedes abrirlo con Excel o pandas para ver:
```python
import pandas as pd
df = pd.read_csv('models/training_history.csv')
print(df.tail())  # Últimas 5 épocas
```

Columnas:
- `epoch`: Número de época
- `loss`: Error de entrenamiento
- `accuracy`: Precisión de entrenamiento
- `val_loss`: Error de validación
- `val_accuracy`: Precisión de validación
- `lr`: Learning rate actual (cambia con ReduceLROnPlateau)

---

## 🔔 CALLBACKS ACTIVOS

### 1. EarlyStopping
```python
patience = 15 épocas
monitor = 'val_loss'
restore_best_weights = True
```

**¿Qué hace?**
- Detiene el entrenamiento si `val_loss` no mejora por 15 épocas consecutivas
- Restaura automáticamente los pesos del mejor modelo
- Previene desperdicio de tiempo en entrenamiento innecesario

**Mensaje esperado:**
```
Restoring model weights from the end of the best epoch: 35.
Epoch 50: early stopping
```

### 2. ModelCheckpoint
```python
filepath = 'models/best_model.keras'
monitor = 'val_loss'
save_best_only = True
```

**¿Qué hace?**
- Guarda el modelo cada vez que `val_loss` mejora
- Solo mantiene la mejor versión (sobrescribe anteriores)
- Garantiza que tendremos el mejor modelo al final

**Mensaje esperado:**
```
Epoch 12: val_loss improved from 0.4532 to 0.4321, saving model to models/best_model.keras
```

### 3. ReduceLROnPlateau
```python
factor = 0.5
patience = 5 épocas
monitor = 'val_loss'
min_lr = 0.00001
```

**¿Qué hace?**
- Reduce learning rate a la mitad si `val_loss` no mejora por 5 épocas
- Ayuda a refinar el aprendizaje en fases avanzadas
- Mínimo: 0.00001 (no baja más)

**Mensaje esperado:**
```
Epoch 18: ReduceLROnPlateau reducing learning rate to 0.0005
```

### 4. TensorBoard
```python
log_dir = 'logs/tensorboard/[timestamp]'
update_freq = 'epoch'
```

**¿Qué hace?**
- Registra métricas cada época
- Permite visualización en tiempo real
- Guarda gráficos de arquitectura

### 5. CSVLogger
```python
filename = 'models/training_history.csv'
append = False
```

**¿Qué hace?**
- Guarda métricas en CSV cada época
- Permite análisis posterior con pandas/Excel
- Backup independiente de TensorBoard

---

## 📁 ARCHIVOS GENERADOS DURANTE ENTRENAMIENTO

### En `models/`
```
best_model.keras              - Mejor modelo guardado (se actualiza)
training_history.csv          - Métricas por época (se actualiza)
```

### En `logs/tensorboard/`
```
[timestamp]/
  ├── train/
  │   └── events.out.tfevents...    - Métricas de entrenamiento
  └── validation/
      └── events.out.tfevents...    - Métricas de validación
```

---

## 🚨 SEÑALES DE ALERTA

### Problemas Comunes y Soluciones

#### 1. Overfitting (Sobreajuste)
**Síntomas:**
- `val_loss` aumenta mientras `loss` disminuye
- `val_accuracy` << `accuracy` (diferencia >10%)

**Causa:** Modelo memoriza training data
**Solución automática:** EarlyStopping detendrá el entrenamiento

#### 2. Underfitting (Subajuste)
**Síntomas:**
- Tanto `loss` como `val_loss` se mantienen altos
- `accuracy` y `val_accuracy` < 75%

**Causa:** Modelo muy simple o LR muy bajo
**Solución:** Ya se está usando modelo complejo, esperar más épocas

#### 3. Loss Explosiva
**Síntomas:**
- `loss` se vuelve NaN
- Accuracy baja a 0% o 100%

**Causa:** Learning rate muy alto
**Solución automática:** ReduceLROnPlateau reducirá LR

#### 4. Entrenamiento Muy Lento
**Síntomas:**
- Cada época toma >5 minutos

**Causa:** CPU sin optimizaciones o batch size muy grande
**Solución:** Ya se están usando optimizaciones oneDNN

---

## 🎓 INTERPRETACIÓN DE RESULTADOS

### Métricas Objetivo
```
✅ EXCELENTE:
   - accuracy > 90%
   - val_accuracy > 85%
   - val_loss < 0.3

✅ BUENO:
   - accuracy > 85%
   - val_accuracy > 80%
   - val_loss < 0.4

⚠️ ACEPTABLE:
   - accuracy > 80%
   - val_accuracy > 75%
   - val_loss < 0.5

❌ NECESITA MEJORA:
   - accuracy < 80%
   - val_accuracy < 75%
   - val_loss > 0.5
```

### Balance Loss vs Accuracy
```
Ideal:
  loss ≈ val_loss          → Buen balance, no overfitting
  accuracy ≈ val_accuracy  → Generalización correcta

Overfitting:
  loss << val_loss         → Modelo memoriza training
  accuracy >> val_accuracy → No generaliza a nuevos datos

Underfitting:
  loss ≈ val_loss ≈ alto   → Modelo muy simple
  accuracy ≈ val_accuracy < 80% → Capacidad insuficiente
```

---

## 🔍 COMANDOS ÚTILES DURANTE ENTRENAMIENTO

### Ver progreso en tiempo real (PowerShell)
```powershell
# Ver últimas líneas del CSV
Get-Content models/training_history.csv -Tail 5

# Ver tamaño del modelo guardado
Get-ChildItem models/best_model.keras | Select-Object Name, Length, LastWriteTime

# Verificar que el proceso esté corriendo
Get-Process python
```

### Monitorear recursos del sistema
```powershell
# CPU y memoria
Get-Process python | Select-Object CPU, WorkingSet, Name

# Uso de disco
Get-PSDrive C
```

### Si necesitas detener el entrenamiento
```
1. Presiona Ctrl+C en la terminal donde corre el entrenamiento
2. El modelo guardará automáticamente el mejor checkpoint hasta ese momento
3. Puedes reanudar desde el mejor modelo guardado
```

---

## 📊 ANÁLISIS POST-ENTRENAMIENTO

### Al Completarse el Entrenamiento

El script mostrará un resumen final:
```
===============================================
        TRAINING COMPLETED SUCCESSFULLY
===============================================
Best model saved at: models/best_model.keras
Training history saved at: models/training_history.json

Final Metrics:
  - Training Loss:        0.XXXX
  - Training Accuracy:    XX.XX%
  - Validation Loss:      0.XXXX
  - Validation Accuracy:  XX.XX%

Total Training Time:     XX:XX:XX
Total Epochs:            XX
Best Epoch:              XX
===============================================
```

### Archivos Finales Generados
```
models/
  ├── best_model.keras          - Modelo listo para usar (~20 MB)
  ├── training_history.json     - Historial completo
  └── training_history.csv      - Métricas tabuladas

logs/
  └── tensorboard/              - Logs completos
      └── [timestamp]/
```

---

## ⏭️ PRÓXIMOS PASOS DESPUÉS DEL ENTRENAMIENTO

1. **Evaluar en Test Set (10 min)**
   ```bash
   python scripts/evaluate_model.py
   ```
   - Calcula métricas finales en 828 imágenes de prueba
   - Genera matriz de confusión
   - Calcula ROC AUC, Precision, Recall, F1

2. **Generar Visualizaciones (10 min)**
   ```bash
   python scripts/visualize_results.py
   ```
   - Curvas de entrenamiento
   - Matriz de confusión (300 DPI)
   - Curva ROC
   - Predicciones de muestra

3. **Actualizar Documentación (15 min)**
   - Agregar métricas finales a README.md
   - Completar REGISTRO_PROCESO.md
   - Preparar presentación de resultados

4. **Commit a GitHub (10 min)**
   ```bash
   git add models/ results/
   git commit -m "feat: Modelo CNN entrenado - Accuracy XX%"
   git push origin main
   ```

---

## 📞 CONTACTO Y SOPORTE

**Si el entrenamiento se detiene o hay errores:**

1. Captura el mensaje de error completo
2. Verifica logs en `logs/tensorboard/`
3. Revisa `models/training_history.csv` para última época exitosa
4. Consulta con el equipo de desarrollo

**Desarrolladores:**
- Cristian Camilo Vega Sánchez
- Daniel Santiago Arévalo Rubiano

**Asesor:**
- Prof. Yeison Eduardo Conejo Sandoval

---

**Última actualización:** 3 de noviembre de 2025 - 18:42  
**Estado:** 🟢 Entrenamiento en progreso  
**Revisión siguiente:** Al completar entrenamiento
