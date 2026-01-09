"""
Controladores (routes) de la API.
"""

from typing import Dict

import pandas as pd
from fastapi import APIRouter

from .schemas import PredictionRequest, PredictionResponse
from .service import predict_from_payload
from .. import config

router = APIRouter()


# Retorna Dict[str, str]: clave "status" y valor "ok".
@router.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint simple para chequear salud del servicio.

    Retorna:
        {"status": "ok"}
    """
    return {"status": "ok"}


# Entrada: PredictionRequest (un vuelo).
# Salida: PredictionResponse con predicción, probabilidad y threshold.
@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Recibe un vuelo y devuelve la predicción.

    Args:
        request: payload con un vuelo.

    Returns:
        PredictionResponse con predicción, probabilidad y threshold.
    """
    df = pd.DataFrame([request.flight.model_dump()])
    prediction, probability = predict_from_payload(df)

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        threshold=config.CLASSIFICATION_THRESHOLD,
    )
