"""
Módulo de ingeniería de features.

Crea features temporales, derivadas y codifica variables categóricas.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
import logging
import joblib

from . import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT)
logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features temporales a partir de la fecha.
    
    Args:
        df: DataFrame con columna 'fl_date'
    
    Returns:
        DataFrame con features temporales añadidas
    """
    logger.info("Creando features temporales...")
    
    df_fe = df.copy()
    
    # Convertir a datetime
    df_fe['fl_date'] = pd.to_datetime(df_fe['fl_date'])
    
    # Extraer componentes temporales
    df_fe['month'] = df_fe['fl_date'].dt.month
    df_fe['day_of_week'] = df_fe['fl_date'].dt.dayofweek
    df_fe['day_of_month'] = df_fe['fl_date'].dt.day
    df_fe['quarter'] = df_fe['fl_date'].dt.quarter
    df_fe['is_weekend'] = (df_fe['day_of_week'] >= 5).astype(int)
    
    # Hora de salida
    df_fe['hour'] = df_fe['crs_dep_time'] // 100
    
    # Componentes ciclicas de hora (sin/cos para capturar circularidad)
    df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
    
    logger.info("  ✓ Features temporales creadas")
    
    return df_fe


def categorize_time_of_day(hour: int) -> str:
    """
    Categoriza la hora del día.
    
    Args:
        hour: Hora en formato 0-23
    
    Returns:
        Categoría: 'morning', 'afternoon', 'evening', 'night'
    """
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features derivadas (diferencias, categorías, etc.).
    
    Args:
        df: DataFrame con features base
    
    Returns:
        DataFrame con features derivadas añadidas
    """
    logger.info("Creando features derivadas...")
    
    df_fe = df.copy()
    
    # Categoría de hora del día
    df_fe['time_of_day'] = df_fe['hour'].apply(categorize_time_of_day)
    
    # Diferencias climáticas
    df_fe['temp_diff'] = df_fe['dest_weather_tavg'] - df_fe['origin_weather_tavg']
    df_fe['prcp_diff'] = df_fe['dest_weather_prcp'] - df_fe['origin_weather_prcp']
    
    # Categoría de distancia
    df_fe['distance_category'] = pd.cut(
        df_fe['distance'],
        bins=[0, 500, 1000, 2000, 5000],
        labels=['short', 'medium', 'long', 'very_long']
    )
    
    logger.info("  ✓ Features derivadas creadas")
    
    return df_fe


def encode_categorical_features(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Codifica features categóricas usando LabelEncoder.
    
    Args:
        df: DataFrame con features categóricas
        encoders: Diccionario de encoders (si ya están fitted)
        fit: Si True, fit los encoders. Si False, solo transform
    
    Returns:
        Tupla (DataFrame con features codificadas, diccionario de encoders)
    """
    logger.info(f"Codificando features categóricas (fit={fit})...")
    
    df_encoded = df.copy()
    
    if encoders is None:
        encoders = {}
    
    for col in config.CATEGORICAL_FEATURES:
        if col not in df_encoded.columns:
            logger.warning(f"  ⚠ Feature {col} no encontrada, saltando...")
            continue
        
        if fit:
            # Crear y fit encoder
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            logger.info(f"  ✓ {col}: {len(le.classes_)} clases únicas")
        else:
            # Usar encoder existente
            le = encoders[col]
            # Manejar valores nuevos no vistos
            df_encoded[f'{col}_encoded'] = df_encoded[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df_encoded, encoders


def feature_engineering_pipeline(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder] = None,
    fit_encoders: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Pipeline completo de feature engineering.
    
    Args:
        df: DataFrame preprocesado
        encoders: Encoders existentes (opcional)
        fit_encoders: Si se deben fit los encoders
    
    Returns:
        Tupla (DataFrame con features, diccionario de encoders)
    """
    logger.info("="*80)
    logger.info("INICIANDO FEATURE ENGINEERING")
    logger.info("="*80)
    
    # 1. Features temporales
    df_fe = create_temporal_features(df)
    
    # 2. Features derivadas
    df_fe = create_derived_features(df_fe)
    
    # 3. Codificar categóricas
    df_fe, encoders = encode_categorical_features(df_fe, encoders, fit=fit_encoders)
    
    logger.info("="*80)
    logger.info(f"FEATURE ENGINEERING COMPLETADO: {df_fe.shape[1]} columnas totales")
    logger.info("=" *80)
    
    return df_fe, encoders


def get_feature_columns() -> list:
    """
    Retorna la lista de features finales para el modelo.
    
    Returns:
        Lista de nombres de columnas
    """
    # Features numéricas
    numeric = config.NUMERIC_FEATURES.copy()
    
    # Features categóricas codificadas
    categorical_encoded = [f'{col}_encoded' for col in config.CATEGORICAL_FEATURES]
    
    # Todas las features
    all_features = numeric + categorical_encoded
    
    return all_features


def save_encoders(encoders: Dict[str, LabelEncoder], file_path: str = None):
    """
    Guarda los encoders en disco.
    
    Args:
        encoders: Diccionario de encoders
        file_path: Ruta donde guardar (None = usar config)
    """
    if file_path is None:
        file_path = config.ENCODERS_PATH
    
    joblib.dump(encoders, file_path)
    logger.info(f"✓ Encoders guardados en {file_path}")


def load_encoders(file_path: str = None) -> Dict[str, LabelEncoder]:
    """
    Carga los encoders desde disco.
    
    Args:
        file_path: Ruta de donde cargar (None = usar config)
    
    Returns:
        Diccionario de encoders
    """
    if file_path is None:
        file_path = config.ENCODERS_PATH
    
    encoders = joblib.load(file_path)
    logger.info(f"✓ Encoders cargados desde {file_path}")
    
    return encoders


if __name__ == '__main__':
    # Test del módulo
    from src.preprocessing import preprocess_pipeline
    
    df = preprocess_pipeline()
    df_fe, encoders = feature_engineering_pipeline(df)
    
    features = get_feature_columns()
    
    print("\n✓ Feature engineering exitoso")
    print(f"  Features totales: {len(features)}")
    print(f"  Features numéricas: {len(config.NUMERIC_FEATURES)}")
    print(f"  Features categóricas: {len(config.CATEGORICAL_FEATURES)}")
