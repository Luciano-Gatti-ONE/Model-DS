"""
Lógica de negocio para la API.
"""

from functools import lru_cache
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from fastapi import HTTPException

from .. import config
from ..features import (
    create_derived_features,
    create_temporal_features,
    encode_categorical_features,
    get_feature_columns,
)

REQUIRED_COLUMNS = [
    "op_unique_carrier",
    "origin",
    "dest",
    "crs_dep_time",
    "fl_date",
    "distance",
    "crs_elapsed_time",
    "origin_weather_tavg",
    "origin_weather_prcp",
    "origin_weather_wspd",
    "origin_weather_pres",
    "dest_weather_tavg",
    "dest_weather_prcp",
    "dest_weather_wspd",
    "dest_weather_pres",
]


# Tuple[Any, Any, Dict[str, Any]] = (modelo, scaler, encoders).
# - Any: objeto cargado con joblib (modelo o scaler).
# - Dict[str, Any]: diccionario de encoders por columna.
@lru_cache(maxsize=1)
def load_artifacts() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Carga modelo, scaler y encoders desde disco.

    Retorna:
        (model, scaler, encoders)
    """
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(config.ENCODERS_PATH)
    return model, scaler, encoders


def validate_payload(df: pd.DataFrame) -> None:
    """
    Valida que el payload tenga todas las columnas y sin nulos.

    Args:
        df: DataFrame con un solo vuelo.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas requeridas: {', '.join(missing_columns)}",
        )

    null_columns = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isnull().any()].tolist()
    if null_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Columnas con valores nulos: {', '.join(null_columns)}",
        )


def build_features(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    """
    Ejecuta el pipeline de features para inferencia.

    Args:
        df: DataFrame con el vuelo.
        encoders: encoders entrenados para variables categóricas.

    Returns:
        DataFrame listo para escalar y predecir.
    """
    df_features = create_temporal_features(df)
    df_features = create_derived_features(df_features)
    df_features, _ = encode_categorical_features(df_features, encoders, fit=False)
    return df_features


def predict_from_payload(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Calcula predicción y probabilidad para un solo vuelo.

    Args:
        df: DataFrame con un solo vuelo.

    Returns:
        (prediction, probability)
    """
    validate_payload(df)
    model, scaler, encoders = load_artifacts()
    df_features = build_features(df, encoders)

    feature_columns = get_feature_columns()
    missing_feature_columns = [col for col in feature_columns if col not in df_features.columns]
    if missing_feature_columns:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudieron construir las features: {', '.join(missing_feature_columns)}",
        )

    x_features = df_features[feature_columns].copy().fillna(0)
    numeric_features = [f for f in config.NUMERIC_FEATURES if f in feature_columns]
    x_features[numeric_features] = scaler.transform(x_features[numeric_features])
    probability = float(model.predict_proba(x_features)[0, 1])
    prediction = int(probability >= config.CLASSIFICATION_THRESHOLD)

    return prediction, probability
