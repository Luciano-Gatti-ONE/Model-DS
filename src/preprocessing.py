"""
Módulo de preprocesamiento de datos.

Incluye limpieza, imputación de valores faltantes y transformación de tipos.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

from . import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT)
logger = logging.getLogger(__name__)


def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo Parquet.
    
    Args:
        file_path: Ruta al archivo Parquet. Si None, usa config.RAW_DATA_PATH
    
    Returns:
        DataFrame con los datos cargados
    """
    if file_path is None:
        file_path = config.RAW_DATA_PATH
    
    logger.info(f"Cargando datos desde {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset eliminando columnas innecesarias y vuelos cancelados.
    
    Args:
        df: DataFrame original
    
    Returns:
        DataFrame limpio
    """
    logger.info("Iniciando limpieza de datos...")
    
    # Eliminar columnas con alta proporción de nulos
    df_clean = df.drop(columns=config.COLS_TO_DROP, errors='ignore')
    logger.info(f"Eliminadas {len(config.COLS_TO_DROP)} columnas con alta proporción de nulos")
    
    # Filtrar vuelos cancelados
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['cancelled'] == 0].copy()
    logger.info(f"Eliminados {initial_count - len(df_clean):,} vuelos cancelados")
    
    # Crear target
    df_clean[config.TARGET_COLUMN] = (df_clean['arr_delay'] > config.DELAY_THRESHOLD).astype(int)
    logger.info(f"Target creado: {df_clean[config.TARGET_COLUMN].sum():,} vuelos retrasados ({df_clean[config.TARGET_COLUMN].mean():.2%})")
    
    # Eliminar filas con nulos en columnas críticas
    df_clean = df_clean.dropna(subset=config.CRITICAL_COLUMNS)
    logger.info(f"Después de eliminar nulos críticos: {len(df_clean):,} registros")
    
    return df_clean


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en features climáticas usando la mediana.
    
    Args:
        df: DataFrame con valores faltantes
    
    Returns:
        DataFrame con valores imputados
    """
    logger.info("Imputando valores faltantes...")
    
    df_imputed = df.copy()
    
    for col in config.CLIMATE_FEATURES:
        if col in df_imputed.columns:
            null_count = df_imputed[col].isnull().sum()
            if null_count > 0:
                median_value = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_value)
                logger.info(f"  {col}: {null_count} nulos imputados con mediana {median_value:.2f}")
    
    return df_imputed


def sample_data(df: pd.DataFrame, sample_size: int = None, random_state: int = None) -> pd.DataFrame:
    """
    Toma una muestra aleatoria del dataset.
    
    Args:
        df: DataFrame completo
        sample_size: Número de registros a muestrear. Si None, usa config.SAMPLE_SIZE
        random_state: Semilla para reproducibilidad. Si None, usa config.RANDOM_STATE
    
    Returns:
        DataFrame muestreado
    """
    if sample_size is None:
        sample_size = config.SAMPLE_SIZE
    
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    if sample_size is None or sample_size >= len(df):
        logger.info("Usando dataset completo (no se realiza sampling)")
        return df
    
    logger.info(f"Muestreando {sample_size:,} registros (sampling aleatorio)")
    df_sample = df.sample(n=sample_size, random_state=random_state)
    
    # Verificar distribución del target
    target_dist = df_sample[config.TARGET_COLUMN].value_counts(normalize=True)
    logger.info(f"Distribución del target en muestra: On-time={target_dist[0]:.2%}, Delayed={target_dist[1]:.2%}")
    
    return df_sample


def preprocess_pipeline(file_path: str = None, sample_size: int = None) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento.
    
    Args:
        file_path: Ruta al archivo de datos
        sample_size: Tamaño de muestra (None = usar config)
    
    Returns:
        DataFrame preprocesado
    """
    logger.info("="*80)
    logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO")
    logger.info("="*80)
    
    # 1. Cargar datos
    df = load_data(file_path)
    
    # 2. Limpiar datos
    df_clean = clean_data(df)
    
    # 3. Imputar valores faltantes
    df_imputed = impute_missing_values(df_clean)
    
    # 4. Sampling (si aplica)
    df_final = sample_data(df_imputed, sample_size)
    
    logger.info("="*80)
    logger.info(f"PREPROCESAMIENTO COMPLETADO: {len(df_final):,} registros finales")
    logger.info("="*80)
    
    return df_final


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en train y test.
    
    Args:
        df: DataFrame preprocesado
    
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Dividiendo datos en train/test (test_size={config.TEST_SIZE})")
    
    # Separar features y target
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN, 'arr_delay', 'dep_delay', 'cancelled'], errors='ignore')
    
    # Split estratificado
    if config.STRATIFY:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        logger.info("Split estratificado por target")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
    
    logger.info(f"Train: {len(X_train):,} registros ({y_train.mean():.2%} delayed)")
    logger.info(f"Test:  {len(X_test):,} registros ({y_test.mean():.2%} delayed)")
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Test del módulo
    df = preprocess_pipeline()
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("\n✓ Preprocesamiento exitoso")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
