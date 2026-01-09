# Guía del Proyecto FlightOnTime - Explicación Simple

**Equipo**: MODELS THAT MATTER - Grupo 59  
**Proyecto**: FlightOnTime ✈️ — Predicción de Retrasos de Vuelos  
**Fecha**: Diciembre 2025

---

## 🎯 ¿Qué es este proyecto?

Este es un **sistema inteligente** que predice si un vuelo se va a retrasar. 

**Analogía simple**: Es como tener un meteorólogo experto, pero en vez de predecir el clima, predice retrasos de vuelos basándose en patrones históricos.

---

## 📁 Estructura del Proyecto (¿Qué hay en cada carpeta?)

```
flight-delay-prediction/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📂 data/
├── 📂 notebooks/
├── 📂 src/
├── 📂 models/
└── 📂 outputs/
```

---

## 📄 Archivos en la Raíz (Los documentos principales)

### 1. README.md - "Manual de Usuario"
**¿Qué es?** El manual de instrucciones del proyecto.

**Analogía**: Como el manual que viene con un electrodoméstico nuevo.

**Contiene**:
- Qué hace el proyecto
- Cómo instalarlo
- Cómo usarlo
- Resultados del modelo (78.55% de precisión)
- Contactos del equipo

**¿Para quién?** Cualquier persona que quiera entender o usar el proyecto.

---

### 2. requirements.txt - "Lista de Ingredientes"
**¿Qué es?** Lista de programas/librerías que necesita el proyecto para funcionar.

**Analogía**: Como una lista de ingredientes para una receta. Si quieres cocinar el platillo (ejecutar el proyecto), necesitas todos los ingredientes.

**Ejemplo de contenido**:
```
numpy>=1.24.0          # Herramienta para matemáticas
pandas>=2.0.0          # Herramienta para datos
xgboost>=2.0.0         # El "cerebro" del modelo
```

**¿Para quién?** Técnicos que van a instalar el proyecto.

---

### 3. .gitignore - "Lista de lo que NO Compartir"
**¿Qué es?** Le dice al sistema de control de versiones (Git) qué archivos NO subir al repositorio.

**Analogía**: Como decirle a alguien "cuando hagas la copia de seguridad, NO incluyas mi carpeta de descargas".

**¿Por qué?** Para no compartir:
- Datos sensibles
- Archivos muy grandes
- Configuraciones personales

**¿Para quién?** Programadores que usan Git.

---

## 📂 data/ - "La Bodega de Datos"

**¿Qué es?** Carpeta donde se guardan todos los datos del proyecto.

**Analogía**: Como el almacén de un supermercado donde guardan la mercancía.

### Subcarpetas:

#### 📁 data/raw/ - "Productos sin procesar"
- **Contiene**: Datos originales, tal como llegaron
- **Ejemplo**: `flights_with_weather_complete.parquet` (7+ millones de vuelos)
- **Regla**: NUNCA modificar estos archivos, son la fuente original
- **Analogía**: Como los ingredientes frescos que acabas de comprar

#### 📁 data/processed/ - "Productos listos para usar"
- **Contiene**: Datos ya limpios y preparados
- **Ejemplo**: `train.parquet`, `test.parquet`
- **Función**: Datos listos para entrenar el modelo
- **Analogía**: Como verduras ya lavadas y cortadas, listas para cocinar

---

## 📂 notebooks/ - "Cuadernos de Laboratorio"

**¿Qué es?** Carpeta para análisis interactivos y exploraciones.

**Analogía**: Como un cuaderno de científico donde anota experimentos, hace gráficas y prueba ideas.

**Tipo de archivos**: `.ipynb` (Jupyter Notebooks)

**¿Qué se hace aquí?**
1. **Exploración de datos** (EDA):
   - Ver cómo se ven los datos
   - Hacer gráficas
   - Buscar patrones
   
2. **Experimentos**:
   - Probar ideas nuevas
   - Entrenar modelos de prueba
   - Comparar diferentes enfoques

**¿Para quién?** 
- Data scientists que quieren experimentar
- Personas que prefieren trabajar de forma visual e interactiva

**Estado actual**: Vacío (listo para que crees tus notebooks)

---

## 📂 src/ - "El Código Organizado" (La Cocina)

**¿Qué es?** El corazón del proyecto. Todo el código Python organizado en módulos.

**Analogía**: Como una cocina profesional donde cada ingrediente tiene su lugar y cada chef tiene su estación.

**¿Por qué organizar así?**
- ✅ Fácil de entender
- ✅ Fácil de mantener
- ✅ Reutilizable
- ✅ Profesional

### Los 5 Módulos (Archivos Python):

#### 1. `config.py` - "El Panel de Control"
**¿Qué hace?** Define TODAS las configuraciones del proyecto en un solo lugar.

**Analogía**: Como el panel de control en un avión - todos los ajustes importantes en un solo lugar.

**Contiene**:
- 📍 Rutas de archivos
- 🎯 Parámetros del modelo
- 🔧 Configuraciones de entrenamiento
- 📊 Umbrales de predicción

**Ejemplo**:
```python
DELAY_THRESHOLD = 15        # Cuántos minutos = retraso
SAMPLE_SIZE = 500000        # Cuántos datos usar
CLASSIFICATION_THRESHOLD = 0.25  # Sensibilidad del modelo
```

**Ventaja**: Si quieres cambiar algo, solo editas este archivo, no 20 archivos diferentes.

---

#### 2. `preprocessing.py` - "La Limpieza"
**¿Qué hace?** Limpia y prepara los datos brutos.

**Analogía**: Como lavar y pelar las verduras antes de cocinar.

**Funciones principales**:
- `load_data()` → Cargar datos desde archivo
- `clean_data()` → Eliminar datos malos/inválidos
- `impute_missing_values()` → Rellenar huecos (datos faltantes)
- `split_data()` → Dividir en entrenamiento/prueba

**¿Por qué es necesario?**
- Los datos reales siempre tienen errores
- Hay valores faltantes
- Necesitamos datos limpios para entrenar

**Ejemplo de lo que hace**:
```
Datos sucios:
- Vuelo sin hora de salida ❌
- Clima = "N/A" ❌
- Vuelo cancelado (pero queremos solo operados) ❌

↓ [preprocessing.py hace su magia]

Datos limpios:
- Solo vuelos completos ✅
- Clima rellenado con promedio ✅
- Solo vuelos operados ✅
```

---

#### 3. `features.py` - "El Creativo"
**¿Qué hace?** Crea características nuevas a partir de los datos originales.

**Analogía**: Como un chef que combina ingredientes básicos para crear nuevos sabores.

**Funciones principales**:
- `create_temporal_features()` → Extrae hora, día, mes, etc.
- `create_derived_features()` → Crea diferencias (ej: clima origen - clima destino)
- `encode_categorical_features()` → Convierte texto en números

**Ejemplo de transformaciones**:

**Dato original**:
```
Fecha: "2024-06-15"
```

**Features creadas**:
```
→ month: 6 (Junio)
→ day_of_week: 5 (Viernes)
→ quarter: 2 (Q2)
→ is_weekend: 1 (Sí)
→ is_summer: 1 (Temporada alta)
```

**¿Por qué?** El modelo aprende mejor con estas características derivadas.

---

#### 4. `modeling.py` - "El Cerebro"
**¿Qué hace?** Entrena y usa el modelo de predicción.

**Analogía**: Como entrenar a un perro para detectar algo - le enseñas con ejemplos hasta que aprende.

**Funciones principales**:
- `create_model()` → Crea el modelo XGBoost
- `train_model()` → Entrena con datos históricos
- `save_model()` → Guarda el modelo entrenado
- `load_model()` → Carga modelo guardado
- `predict()` → Hace predicciones

**El proceso de entrenamiento**:
```
1. Recibe 500,000 vuelos históricos
2. Aprende patrones:
   - "Vuelos viernes noche → más retrasos"
   - "Lluvia fuerte → más retrasos"
   - "Temporada navideña → más retrasos"
3. Se vuelve experto en predecir
```

**Resultado**: Un modelo que detecta 78 de cada 100 retrasos.

---

#### 5. `evaluation.py` - "El Inspector de Calidad"
**¿Qué hace?** Mide qué tan bueno es el modelo.

**Analogía**: Como un inspector de calidad que prueba productos y da calificaciones.

**Funciones principales**:
- `calculate_metrics()` → Calcula precisión, recall, F1, etc.
- `plot_confusion_matrix()` → Gráfica de aciertos/errores
- `plot_roc_curve()` → Curva de rendimiento
- `plot_feature_importance()` → Qué es más importante
- `evaluate_model()` → Evaluación completa

**Métricas que calcula**:
```
✅ Accuracy: 80.39% (aciertos totales)
✅ Precision: 50.73% (de las alarmas, cuántas son correctas)
✅ Recall: 78.55% (de los retrasos reales, cuántos detecta)
✅ F1-Score: 0.6165 (balance general)
```

**Visualizaciones que genera**:
- Matriz de confusión (aciertos vs errores)
- Curva ROC (rendimiento general)
- Importancia de features (qué factores pesan más)

---

## 📂 models/ - "El Producto Final"

**¿Qué es?** Carpeta donde se guardan los modelos entrenados.

**Analogía**: Como el freezer donde guardas la comida ya preparada, lista para calentar y servir.

**Archivos guardados**:

### 1. `model.joblib` - "El Cerebro Entrenado"
- **Tamaño**: ~50-100 MB
- **Contiene**: El modelo XGBoost completamente entrenado
- **Analogía**: Como un chef experto congelado - cuando lo "descongelas" (cargas), ya sabe cocinar

### 2. `scaler.pkl` - "El Traductor de Números"
- **Función**: Normaliza los números para que el modelo entienda mejor
- **Ejemplo**: Convierte distancia 2500km y temperatura 25°C a la misma escala

### 3. `label_encoders.pkl` - "El Diccionario"
- **Función**: Convierte texto en números
- **Ejemplo**: 
  ```
  "American Airlines" → 0
  "Delta" → 1
  "United" → 2
  ```

### 4. `metadata.json` - "La Ficha Técnica"
- **Contiene**: Información del modelo
  - Fecha de creación
  - Versión
  - Parámetros usados
  - Métricas de rendimiento
  - Lista de features

**Analogía**: Como la etiqueta de un producto que dice ingredientes, fecha de vencimiento, etc.

---

## 📂 outputs/ - "Los Resultados"

**¿Qué es?** Carpeta donde se guardan todos los resultados del modelo.

**Analogía**: Como el portafolio de un fotógrafo donde muestra su trabajo.

### Subcarpetas:

#### 📁 outputs/figures/ - "Las Fotos"
**Contiene**: Gráficas y visualizaciones

**Ejemplos**:
- `confusion_matrix.png` → Matriz de aciertos/errores
- `roc_curve.png` → Curva de rendimiento
- `feature_importance.png` → Qué factores son más importantes

**¿Para qué?** Presentaciones, reportes, entender visualmente cómo funciona el modelo.

#### 📁 outputs/metrics/ - "Los Números"
**Contiene**: Métricas en formato JSON

**Ejemplo** (`model_metrics.json`):
```json
{
  "timestamp": "2025-12-29 10:00:00",
  "threshold": 0.25,
  "metrics": {
    "accuracy": 0.8039,
    "precision": 0.5073,
    "recall": 0.7855,
    "f1_score": 0.6165
  }
}
```

**¿Para qué?** Seguimiento histórico, comparaciones, reportes automatizados.

---

## 🔄 ¿Cómo Funcionan Juntos Todos los Componentes?

### El Flujo Completo (De Inicio a Fin):

```
1. 📂 data/raw/
   ↓
   [Datos originales de vuelos]
   
2. 📂 src/preprocessing.py
   ↓
   [Limpia y prepara datos]
   
3. 📂 src/features.py
   ↓
   [Crea características útiles]
   
4. 📂 data/processed/
   ↓
   [Datos listos para entrenar]
   
5. 📂 src/modeling.py
   ↓
   [Entrena el modelo]
   
6. 📂 models/
   ↓
   [Modelo guardado]
   
7. 📂 src/evaluation.py
   ↓
   [Evalúa rendimiento]
   
8. 📂 outputs/
   ↓
   [Resultados finales: gráficas + métr icas]
```

### Como una Receta de Cocina:

1. **Ingredientes** (`data/raw/`) → Datos crudos
2. **Preparación** (`preprocessing.py`) → Lavar y cortar
3. **Mezcla** (`features.py`) → Combinar ingredientes
4. **Cocción** (`modeling.py`) → Entrenar el modelo
5. **Producto Final** (`models/`) → Platillo listo
6. **Prueba de Sabor** (`evaluation.py`) → ¿Qué tal quedó?
7. **Presentación** (`outputs/`) → Servir en plato bonito

---

## 🎓 Para Diferentes Audiencias

### Si eres Gerente/Directivo:
**Lee**:
- ✅ Este documento (GUÍA_PROYECTO.md)
- ✅ README.md (sección de resultados)
- ✅ `outputs/figures/` (ver gráficas)

**Ignora**:
- ❌ Archivos .py (código técnico)
- ❌ requirements.txt
- ❌ .gitignore

---

### Si eres Analista/Usuario de Negocio:
**Lee**:
- ✅ Este documento
- ✅ README.md
- ✅ `outputs/metrics/` (números de rendimiento)

**Explora**:
- 🔍 `notebooks/` (si quieres hacer análisis propios)

---

### Si eres Desarrollador/Data Scientist:
**Lee**:
- ✅ Todo este documento
- ✅ README.md (documentación técnica)
- ✅ Código en `src/` (entender la implementación)

**Usa**:
- 🔧 `src/config.py` (ajustar parámetros)
- 🔧 `notebooks/` (experimentar)
- 🔧 Todos los módulos

---

## 📊 Resumen Visual

```
┌────────────────────────────────────────────┐
│  flight-delay-prediction/                 │
│  (El Sistema Completo)                     │
└────────────────────────────────────────────┘
        │
        ├─── 📄 Documentos (README, etc.)
        │    └─ "Manual de usuario"
        │
        ├─── 📂 data/
        │    ├─ raw/ → "Ingredientes crudos"
        │    └─ processed/ → "Ingredientes preparados"
        │
        ├─── 📂 src/ (El código organizado)
        │    ├─ config.py → "Panel de control"
        │    ├─ preprocessing.py → "Limpieza"
        │    ├─ features.py → "Creativo"
        │    ├─ modeling.py → "Cerebro"
        │    └─ evaluation.py → "Inspector"
        │
        ├─── 📂 models/ → "Producto final"
        │    └─ model.joblib (78.55% de recall!)
        │
        ├─── 📂 notebooks/ → "Laboratorio"
        │    └─ Para experimentar
        │
        └─── 📂 outputs/ → "Resultados"
             ├─ figures/ → Gráficas
             └─ metrics/ → Números
```

---

## ❓ Preguntas Frecuentes

**P: ¿Qué archivo es el más importante?**  
R: `src/modeling.py` (el cerebro) y `models/model.joblib` (el modelo entrenado).

**P: ¿Puedo borrar la carpeta `outputs/`?**  
R: Sí, se puede regenerar. Pero `models/` NO - contiene el modelo que tomó horas entrenar.

**P: ¿Necesito entender todo el código?**  
R: No. Solo necesitas entender:
- Qué hace cada carpeta (este documento)
- Cómo usar el modelo (README.md)

**P: ¿Cómo empiezo a usar esto?**  
R: 
1. Lee README.md
2. Instala dependencias (`pip install -r requirements.txt`)
3. Usa el modelo cargándolo desde `models/`

**P: ¿Dónde están los notebooks mencionados?**  
R: No existen aún. Son sugerencias de lo que puedes crear. Ve a `notebooks/README.md` para instrucciones.

---

## 🚀 Conclusión

Este proyecto es como una **fábrica bien organizada**:

- **Entrada**: Datos de vuelos (históricos)
- **Proceso**: Limpieza → Features → Entrenamiento
- **Salida**: Modelo que predice retrasos con 78.55% de precisión

**Fortalezas**:
- ✅ Código profesional y organizado
- ✅ Fácil de mantener
- ✅ Bien documentado
- ✅ Listo para producción

**¿Siguiente paso?**  
Lee el `README.md` principal para instrucciones de uso detalladas.

---

**Documento creado por**: MODELS THAT MATTER - Grupo 59  
**Última actualización**: 29 de Diciembre, 2025  
**Versión**: 1.0
