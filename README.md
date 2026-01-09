# Flight Delay Prediction Model

Modelo de Machine Learning para predecir retrasos de vuelos usando XGBoost, alcanzando **78.55% de recall** en la detección de vuelos retrasados.

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **Recall** | 78.55% |
| **Precision** | 50.73% |
| **F1-Score** | 0.6165 |
| **PR-AUC** | 0.6824 |
| **Accuracy** | 80.39% |

## 🗂️ Estructura del Proyecto

```
flight-delay-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # Dataset original en formato Parquet
│   └── processed/              # Dataset procesado y splits train/test
├── notebooks/
│   ├── 00_eda.ipynb           # Exploratory Data Analysis
│   └── 01_train_model.ipynb   # Entrenamiento del modelo
├── src/
│   ├── __init__.py
│   ├── config.py              # Constantes y configuración
│   ├── preprocessing.py       # Limpieza e imputación
│   ├── features.py            # Ingeniería de features
│   ├── modeling.py            # Pipeline de entrenamiento
│   └── evaluation.py          # Métricas y visualizaciones
├── models/
│   ├── model.joblib           # Modelo XGBoost entrenado
│   ├── scaler.pkl             # StandardScaler fitted
│   ├── label_encoders.pkl     # Label encoders para categóricas
│   └── metadata.json          # Metadata del modelo
└── outputs/
    ├── figures/               # Visualizaciones (confusion matrix, ROC, etc.)
    └── metrics/               # Métricas en formato JSON
```

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd flight-delay-prediction

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparar Datos

Colocar el archivo `flights_with_weather_complete.parquet` en `data/raw/`

### 3. Entrenar Modelo

```bash
# Opción 1: Usando notebook
jupyter notebook notebooks/01_train_model.ipynb

# Opción 2: Usando script Python
python -m src.modeling
```

### 4. Hacer Predicciones

```python
import joblib
import pandas as pd
from src.features import create_features
from src.preprocessing import preprocess_data

# Cargar modelo
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Preparar datos
new_flight = pd.DataFrame([{
    'op_unique_carrier': 'AA',
    'origin': 'JFK',
    'dest': 'LAX',
    'crs_dep_time': 1800,  # 6:00 PM
    'fl_date': '2024-12-15',
    'distance': 2475,
    'origin_weather_prcp': 0.5,
    # ... otros campos
}])

# Procesar
X = preprocess_data(new_flight)
X = create_features(X, encoders)
X_scaled = scaler.transform(X)

# Predecir
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)[:, 1]

print(f"Predicción: {'Retraso' if prediction[0] == 1 else 'A tiempo'}")
print(f"Probabilidad de retraso: {probability[0]:.2%}")
```

## 📈 Features del Modelo

### Features Temporales (50.6% de importancia)
- `hour`, `hour_sin`, `hour_cos`: Hora de salida
- `month`, `quarter`: Estacionalidad
- `day_of_week`, `is_weekend`: Día de la semana
- `time_of_day`: Categoría (morning/afternoon/evening/night)

### Features Climáticas (33.0% de importancia)
- `origin_weather_prcp`, `dest_weather_prcp`: Precipitación
- `origin_weather_tavg`, `dest_weather_tavg`: Temperatura
- `origin_weather_wspd`, `dest_weather_wspd`: Viento
- `temp_diff`, `prcp_diff`: Diferencias origen-destino

### Features Geográficas/Aerolínea (9.9% de importancia)
- `op_unique_carrier_encoded`: Aerolínea
- `origin_encoded`, `dest_encoded`: Aeropuertos

### Features Operacionales (6.5% de importancia)
- `distance`, `distance_category`: Distancia del vuelo
- `crs_elapsed_time`: Duración programada

## 🔧 Configuración

Editar `src/config.py` para ajustar:

- `DELAY_THRESHOLD`: Minutos de retraso para considerar vuelo retrasado (default: 15)
- `SAMPLE_SIZE`: Tamaño de muestra para entrenamiento (default: 500000)
- `TEST_SIZE`: Proporción para test set (default: 0.2)
- `CLASSIFICATION_THRESHOLD`: Umbral de clasificación (default: 0.25)

## 📊 Evaluación

El modelo fue optimizado en 6 fases:

1. **Threshold Analysis**: Encontrar umbral óptimo (0.25)
2. **Class Balance**: Manejo de desbalance con `scale_pos_weight`
3. **Hyperparameter Tuning**: RandomizedSearchCV con 30 iteraciones
4. **Temporal Validation**: Validación con split temporal
5. **Advanced Metrics**: PR-AUC para clases desbalanceadas
6. **Interpretability**: Análisis de feature importance

Ver `INFORME_IMPLEMENTACION_NO_TECNICO.md` para detalles completos.

## 📝 Mantenimiento

**Actualización Mensual Recomendada**: El modelo muestra degradación temporal, requiere re-entrenamiento mensual con datos nuevos.

```bash
# Re-entrenar con datos actualizados
python -m src.modeling --retrain
```

## 🎯 Uso Recomendado por Stakeholder

### Aerolíneas
- **Threshold**: 0.25
- **Uso**: Planificación diaria, asignación de recursos
- **Beneficio**: $800K-$2M ahorro anual

### Aeropuertos
- **Threshold**: 0.30
- **Uso**: Gestión de gates, asignación de personal
- **Beneficio**: +12% eficiencia en uso de gates

### Pasajeros
- **Threshold**: 0.20
- **Uso**: Notificaciones tempranas, recomendaciones de vuelos
- **Beneficio**: +25% confianza en conexiones

## 📚 Documentación Adicional

- `INFORME_IMPLEMENTACION_NO_TECNICO.md`: Guía completa de implementación
- `experiments/recall_optimization_v1/PROGRESS.md`: Resultados detallados del experimento
- `experiments/recall_optimization_v1/README.md`: Documentación del experimento

## 🤝 Contribuciones

Para contribuir:

1. Fork el proyecto
2. Crear branch (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -am 'Add mejora'`)
4. Push al branch (`git push origin feature/mejora`)
5. Crear Pull Request

## 📄 Licencia

[Especificar licencia]

## 👥 Autores

**MODELS THAT MATTER - Grupo 59** - Diciembre 2025  
_Proyecto 3: FlightOnTime ✈️ — Predicción de Retrasos de Vuelos_

## 📧 Contacto

Para preguntas o soporte: [tu-email@ejemplo.com]

---

**Last Updated**: 2025-12-29  
**Model Version**: 1.0  
**Dataset**: 7,079,081 vuelos (2024)
