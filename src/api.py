"""
API para exponer el modelo de predicción de retrasos.
"""

from functools import lru_cache
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .features import (
    create_derived_features,
    create_temporal_features,
    encode_categorical_features,
    get_feature_columns,
)

# Columnas mínimas que esperamos en el payload.
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


class PredictionRequest(BaseModel):
    """
    Payload de entrada.

    flight: diccionario con las columnas requeridas para el modelo.
    """

    # Dict[str, Any] = diccionario con columnas del vuelo.
    # - str: nombre de columna.
    # - Any: valor de esa columna (número, string, fecha, etc.).
    flight: Dict[str, Any] = Field(..., description="Vuelo a predecir.")


class PredictionResponse(BaseModel):
    """
    Respuesta de salida.

    prediction: 0 = on-time, 1 = delayed.
    probability: probabilidad de retraso.
    threshold: umbral usado para clasificar.
    """

    prediction: int
    probability: float
    threshold: float


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


app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0.0",
    description="API para predecir retrasos de vuelos con el modelo entrenado.",
)


# Retorna Dict[str, str]: clave "status" y valor "ok".
@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint simple para chequear salud del servicio.

    Retorna:
        {"status": "ok"}
    """
    return {"status": "ok"}


# Args:
# - df: pd.DataFrame con un solo vuelo.
# Retorna:
# - None (lanza HTTPException si hay error).
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


# Args:
# - df: pd.DataFrame con un solo vuelo.
# - encoders: Dict[str, Any] con encoders entrenados.
# Retorna:
# - pd.DataFrame con features listas para el modelo.
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


# Entrada: PredictionRequest (un vuelo).
# Salida: PredictionResponse con predicción, probabilidad y threshold.
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Recibe un vuelo y devuelve la predicción.

    Args:
        request: payload con un vuelo.

    Returns:
        PredictionResponse con predicción, probabilidad y threshold.
    """
    df = pd.DataFrame([request.flight])
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

    x_scaled = scaler.transform(df_features[feature_columns])
    probability = float(model.predict_proba(x_scaled)[0, 1])
    prediction = int(probability >= config.CLASSIFICATION_THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        threshold=config.CLASSIFICATION_THRESHOLD,
    )


if __name__ == "__main__":
    import uvicorn

    # Para desarrollo local: python -m src.api
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
