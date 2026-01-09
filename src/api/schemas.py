"""
Schemas/DTOs de entrada y salida para la API.
"""

from pydantic import BaseModel, Field


class FlightPayload(BaseModel):
    """
    DTO con los datos del vuelo para la predicción.
    """

    # Campos categóricos
    op_unique_carrier: str = Field(..., description="Código de la aerolínea (ej: AA).")
    origin: str = Field(..., description="Aeropuerto de origen (ej: JFK).")
    dest: str = Field(..., description="Aeropuerto de destino (ej: LAX).")

    # Campos temporales/operacionales
    crs_dep_time: int = Field(..., description="Hora de salida programada en formato HHMM (ej: 1800).")
    fl_date: str = Field(..., description="Fecha del vuelo en formato YYYY-MM-DD.")
    distance: float = Field(..., description="Distancia del vuelo en millas.")
    crs_elapsed_time: float = Field(..., description="Duración programada del vuelo en minutos.")

    # Clima en origen
    origin_weather_tavg: float = Field(..., description="Temperatura media en origen.")
    origin_weather_prcp: float = Field(..., description="Precipitación en origen.")
    origin_weather_wspd: float = Field(..., description="Velocidad del viento en origen.")
    origin_weather_pres: float = Field(..., description="Presión atmosférica en origen.")

    # Clima en destino
    dest_weather_tavg: float = Field(..., description="Temperatura media en destino.")
    dest_weather_prcp: float = Field(..., description="Precipitación en destino.")
    dest_weather_wspd: float = Field(..., description="Velocidad del viento en destino.")
    dest_weather_pres: float = Field(..., description="Presión atmosférica en destino.")


class PredictionRequest(BaseModel):
    """
    DTO de entrada.
    """

    flight: FlightPayload = Field(..., description="Vuelo a predecir.")


class PredictionResponse(BaseModel):
    """
    DTO de salida.
    """

    prediction: int
    probability: float
    threshold: float
