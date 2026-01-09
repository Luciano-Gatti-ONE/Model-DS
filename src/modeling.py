"""
Módulo de modelado.

Incluye entrenamiento, guardado y carga del modelo.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json
from datetime import datetime
from typing import Tuple, Dict, Any
import logging

from . import config
from .preprocessing import preprocess_pipeline, split_data
from .features import feature_engineering_pipeline, get_feature_columns, save_encoders

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT)
logger = logging.getLogger(__name__)


def create_model(scale_pos_weight: float = None) -> xgb.XGBClassifier:
    """
    Crea un modelo XGBoost con los parámetros configurados.
    
    Args:
        scale_pos_weight: Peso para clase positiva (si None, se calcula automáticamente)
    
    Returns:
        Modelo XGBoost sin entrenar
    """
    params = config.XGBOOST_PARAMS.copy()
    
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
        logger.info(f"Creando modelo XGBoost con scale_pos_weight={scale_pos_weight:.2f}")
    else:
        logger.info("Creando modelo XGBoost (sin scale_pos_weight)")
    
    model = xgb.XGBClassifier(**params)
    
    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[xgb.XGBClassifier, StandardScaler, Dict]:
    """
    Entrena el modelo completo con escalado de features.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        Tupla (modelo entrenado, scaler, encoders)
    """
    logger.info("="*80)
    logger.info("INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info("="*80)
    
    # Obtener features finales
    feature_cols = get_feature_columns()
    
    # Verificar que todas las features existen
    missing_features = set(feature_cols) - set(X_train.columns)
    if missing_features:
        logger.error(f"Features faltantes: {missing_features}")
        raise ValueError(f"Features faltantes en X_train: {missing_features}")
    
    X_train_features = X_train[feature_cols].copy()
    
    # Rellenar NaNs restantes con 0
    X_train_features = X_train_features.fillna(0)
    
    logger.info(f"Features para modelo: {len(feature_cols)}")
    logger.info(f"Registros de entrenamiento: {len(X_train_features):,}")
    
    # Calcular scale_pos_weight si está configurado
    if config.USE_SCALE_POS_WEIGHT:
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive
        logger.info(f"Clases: Negativa={n_negative:,}, Positiva={n_positive:,}")
        logger.info(f"Scale pos weight calculado: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = None
    
    # Crear modelo
    model = create_model(scale_pos_weight)
    
    # Entrenar modelo
    logger.info("Entrenando modelo...")
    start_time = datetime.now()
    
    model.fit(X_train_features, y_train)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"✓ Modelo entrenado en {duration:.1f} segundos")
    
    # Crear y fit scaler (para features numéricas)
    logger.info("Creando escalador de features...")
    scaler = StandardScaler()
    numeric_features = [f for f in config.NUMERIC_FEATURES if f in feature_cols]
    X_train_features[numeric_features] = scaler.fit_transform(X_train_features[numeric_features])
    logger.info(f"✓ Scaler fitted para {len(numeric_features)} features numéricas")
    
    logger.info("="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    
    return model, scaler


def save_model(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    train_metrics: Dict = None,
    feature_columns: list = None
):
    """
    Guarda el modelo, scaler y metadata.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler fitted
        train_metrics: Métricas de entrenamiento (opcional)
        feature_columns: Lista de features (opcional)
    """
    logger.info("Guardando modelo y artifacts...")
    
    # Guardar modelo
    joblib.dump(model, config.MODEL_PATH)
    logger.info(f"✓ Modelo guardado en {config.MODEL_PATH}")
    
    # Guardar scaler
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"✓ Scaler guardado en {config.SCALER_PATH}")
    
    # Crear metadata
    if feature_columns is None:
        feature_columns = get_feature_columns()
    
    metadata = {
        'model_type': config.MODEL_TYPE,
        'model_version': '1.0',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'delay_threshold': config.DELAY_THRESHOLD,
            'classification_threshold': config.CLASSIFICATION_THRESHOLD,
            'sample_size': config.SAMPLE_SIZE,
            'test_size': config.TEST_SIZE,
            'use_scale_pos_weight': config.USE_SCALE_POS_WEIGHT
        },
        'features': {
            'total': len(feature_columns),
            'numeric': len(config.NUMERIC_FEATURES),
            'categorical': len(config.CATEGORICAL_FEATURES),
            'columns': feature_columns
        },
        'xgboost_params': config.XGBOOST_PARAMS
    }
    
    if train_metrics:
        metadata['train_metrics'] = train_metrics
    
    # Guardar metadata
    with open(config.METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata guardada en {config.METADATA_PATH}")


def load_model() -> Tuple[xgb.XGBClassifier, StandardScaler, Dict]:
    """
    Carga el modelo, scaler y encoders guardados.
    
    Returns:
        Tupla (modelo, scaler, encoders)
    """
    from .features import load_encoders
    
    logger.info("Cargando modelo y artifacts...")
    
    model = joblib.load(config.MODEL_PATH)
    logger.info(f"✓ Modelo cargado desde {config.MODEL_PATH}")
    
    scaler = joblib.load(config.SCALER_PATH)
    logger.info(f"✓ Scaler cargado desde {config.SCALER_PATH}")
    
    encoders = load_encoders()
    
    return model, scaler, encoders


def predict(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    X: pd.DataFrame,
    threshold: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza predicciones con el modelo.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler fitted
        X: Features de entrada
        threshold: Umbral de clasificación (None = usar config)
    
    Returns:
        Tupla (predicciones, probabilidades)
    """
    if threshold is None:
        threshold = config.CLASSIFICATION_THRESHOLD
    
    # Obtener features y escalar
    feature_cols = get_feature_columns()
    X_features = X[feature_cols].copy().fillna(0)
    
    numeric_features = [f for f in config.NUMERIC_FEATURES if f in feature_cols]
    X_features[numeric_features] = scaler.transform(X_features[numeric_features])
    
    # Predecir probabilidades
    probabilities = model.predict_proba(X_features)[:, 1]
    
    # Aplicar threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities


if __name__ == '__main__':
    # Pipeline completo de entrenamiento
    logger.info("INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO")
    
    # 1. Preprocesar datos
    df = preprocess_pipeline()
    
    # 2. Feature engineering
    df_fe, encoders = feature_engineering_pipeline(df)
    save_encoders(encoders)
    
    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df_fe)
    
    # 4. Entrenar modelo
    model, scaler = train_model(X_train, y_train)
    
    # 5. Evaluar
    from .evaluation import evaluate_model
    metrics = evaluate_model(model, scaler, X_test, y_test)
    
    # 6. Guardar
    save_model(model, scaler, train_metrics=metrics)
    
    logger.info("\n✓ PIPELINE COMPLETADO EXITOSAMENTE")
