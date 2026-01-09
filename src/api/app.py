"""
Configuración de la aplicación FastAPI.
"""

from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    """
    Crea y configura la app FastAPI.
    """
    app = FastAPI(
        title="Flight Delay Prediction API",
        version="1.0.0",
        description="API para predecir retrasos de vuelos con el modelo entrenado.",
    )

    app.include_router(router)
    return app
