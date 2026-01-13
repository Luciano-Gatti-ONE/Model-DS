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
    day_of_month: int | None = Field(
        None,
        description="Día del mes (1-31). Si no se envía, se deriva desde fl_date."
    )
    day_of_week: int | None = Field(
        None,
        description="Día de la semana (0=Lunes, 6=Domingo). Si no se envía, se deriva desde fl_date."
    )
    distance: float = Field(..., description="Distancia del vuelo en millas.")
    crs_elapsed_time: float | None = Field(
        None,
        description="Duración programada del vuelo en minutos. Si no se envía, se estima con la distancia."
    )

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
