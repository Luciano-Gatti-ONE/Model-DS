"""
Configuración global del proyecto.

Define constantes, parámetros del modelo y paths.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Root del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Directorios de datos
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Directorios de salida
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
METRICS_DIR = OUTPUTS_DIR / 'metrics'

# Asegurar que existen los directorios
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET
# ============================================================================

# Archivo de entrada
RAW_DATA_FILE = 'flights_with_weather_complete.parquet'
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILE

# Archivos procesados
TRAIN_DATA_FILE = 'train.parquet'
TEST_DATA_FILE = 'test.parquet'
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / TRAIN_DATA_FILE
TEST_DATA_PATH = PROCESSED_DATA_DIR / TEST_DATA_FILE

# ============================================================================
# MODELO
# ============================================================================

# Target
TARGET_COLUMN = 'is_delayed'
DELAY_THRESHOLD = 15  # minutos

# Features a eliminar (alta proporción de nulos)
COLS_TO_DROP = [
    'dest_weather_wdir',
    'origin_weather_wdir',
    'dest_weather_wpgt',
    'origin_weather_wpgt',
    'origin_weather_tsun',
    'dest_weather_tsun',
    'cancellation_code'
]

# Features numéricas
NUMERIC_FEATURES = [
    'distance',
    'crs_elapsed_time',
    'month',
    'day_of_week',
    'day_of_month',
    'quarter',
    'is_weekend',
    'hour',
    'hour_sin',
    'hour_cos',
    'origin_weather_tavg',
    'origin_weather_prcp',
    'origin_weather_wspd',
    'origin_weather_pres',
    'dest_weather_tavg',
    'dest_weather_prcp',
    'dest_weather_wspd',
    'dest_weather_pres',
    'temp_diff',
    'prcp_diff'
]

# Features categóricas
CATEGORICAL_FEATURES = [
    'op_unique_carrier',
    'origin',
    'dest',
    'time_of_day',
    'distance_category'
]

# Features climáticas para imputación
CLIMATE_FEATURES = [
    'origin_weather_tavg',
    'origin_weather_tmin',
    'origin_weather_tmax',
    'origin_weather_prcp',
    'origin_weather_snow',
    'origin_weather_wspd',
    'origin_weather_pres',
    'dest_weather_tavg',
    'dest_weather_tmin',
    'dest_weather_tmax',
    'dest_weather_prcp',
    'dest_weather_snow',
    'dest_weather_wspd',
    'dest_weather_pres'
]

# Columnas críticas (no pueden tener nulos)
CRITICAL_COLUMNS = [
    'arr_delay',
    'dep_delay',
    'distance',
    'crs_elapsed_time'
]

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

# Sampling
SAMPLE_SIZE = 500000  # Número de registros para entrenar (None = todos)
RANDOM_STATE = 2024

# Split
TEST_SIZE = 0.2  # 80% train, 20% test
STRATIFY = True  # Estratificar por target

# Modelo
MODEL_TYPE = 'xgboost'

# XGBoost con scale_pos_weight (mejor modelo según experimentación)
XGBOOST_PARAMS = {
    'max_depth': 10,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

# Calcular scale_pos_weight automáticamente basado en desbalance
# scale_pos_weight = (n_negatives / n_positives)
USE_SCALE_POS_WEIGHT = True

# ============================================================================
# PREDICCIÓN
# ============================================================================

# Threshold de clasificación
# 0.50 = default, 0.25 = optimizado para recall
CLASSIFICATION_THRESHOLD = 0.25

# Thresholds recomendados por stakeholder
THRESHOLDS = {
    'airlines': 0.25,     # Balance precision-recall
    'airports': 0.30,     # Priorizar precision
    'passengers': 0.20    # Priorizar recall
}

# ============================================================================
# ARCHIVOS DEL MODELO
# ============================================================================

MODEL_FILE = 'model.joblib'
SCALER_FILE = 'scaler.pkl'
ENCODERS_FILE = 'label_encoders.pkl'
METADATA_FILE = 'metadata.json'

MODEL_PATH = MODELS_DIR / MODEL_FILE
SCALER_PATH = MODELS_DIR / SCALER_FILE
ENCODERS_PATH = MODELS_DIR / ENCODERS_FILE
METADATA_PATH = MODELS_DIR / METADATA_FILE

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

# Tamaño de figuras
FIGURE_SIZE = (10, 6)
LARGE_FIGURE_SIZE = (14, 10)

# Estilo
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# DPI para guardar figuras
DPI = 100

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
