"""
Módulo de evaluación.

Incluye cálculo de métricas, matrices de confusión y visualizaciones.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, Tuple
import logging

from . import config
from .modeling import predict

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT)
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_proba: Probabilidades predichas
    
    Returns:
        Diccionario con métricas
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba))
    }
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Métricas del Modelo"):
    """
    Imprime las métricas de forma legible.
    
    Args:
        metrics: Diccionario de métricas
        title: Título para el reporte
    """
    logger.info("\n" + "="*60)
    logger.info(title.center(60))
    logger.info("="*60)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    logger.info(f"Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})")
    logger.info(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']:.2%})")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    logger.info("="*60 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    show: bool = False
):
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        save_path: Ruta donde guardar (None = usar config)
        show: Si mostrar el plot
    """
    if save_path is None:
        save_path = config.FIGURES_DIR / 'confusion_matrix.png'
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['On-time', 'Delayed'],
                yticklabels=['On-time', 'Delayed'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"✓ Confusion matrix guardada en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = None,
    show: bool = False
):
    """
    Genera y guarda la curva ROC.
    
    Args:
        y_true: Etiquetas verdaderas
        y_proba: Probabilidades predichas
        save_path: Ruta donde guardar (None = usar config)
        show: Si mostrar el plot
    """
    if save_path is None:
        save_path = config.FIGURES_DIR / 'roc_curve.png'
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"✓ ROC curve guardada en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    model,
    save_path: str = None,
    top_n: int = 20,
    show: bool = False
):
    """
    Genera y guarda el gráfico de feature importance.
    
    Args:
        model: Modelo XGBoost entrenado
        save_path: Ruta donde guardar (None = usar config)
        top_n: Número de features a mostrar
        show: Si mostrar el plot
    """
    if save_path is None:
        save_path = config.FIGURES_DIR / 'feature_importance.png'
    
    from .features import get_feature_columns
    
    feature_cols = get_feature_columns()
    importance = model.feature_importances_
    
    # Crear DataFrame y ordenar
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], align='center')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"✓ Feature importance guardada en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_metrics(metrics: Dict, save_path: str = None):
    """
    Guarda las métricas en formato JSON.
    
    Args:
        metrics: Diccionario de métricas
        save_path: Ruta donde guardar (None = usar config)
    """
    if save_path is None:
        save_path = config.METRICS_DIR / 'model_metrics.json'
    
    from datetime import datetime
    
    metrics_with_metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'threshold': config.CLASSIFICATION_THRESHOLD,
        'metrics': metrics
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    
    logger.info(f"✓ Métricas guardadas en {save_path}")


def evaluate_model(
    model,
    scaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = None,
    save_plots: bool = True
) -> Dict:
    """
    Evaluación completa del modelo.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler fitted
        X_test: Features de test
        y_test: Target de test
        threshold: Umbral de clasificación
        save_plots: Si guardar visualizaciones
    
    Returns:
        Diccionario con métricas
    """
    logger.info("="*80)
    logger.info("EVALUANDO MODELO")
    logger.info("="*80)
    
    # Predicciones
    y_pred, y_proba = predict(model, scaler, X_test, threshold)
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Imprimir métricas
    print_metrics(metrics, "Métricas en Test Set")
    
    # Guardar métricas
    save_metrics(metrics)
    
    if save_plots:
        logger.info("Generando visualizaciones...")
        
        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred)
        
        # ROC curve
        plot_roc_curve(y_test, y_proba)
        
        # Feature importance
        plot_feature_importance(model)
    
    logger.info("="*80)
    logger.info("EVALUACIÓN COMPLETADA")
    logger.info("="*80)
    
    return metrics


if __name__ == '__main__':
    # Test del módulo
    from .preprocessing import preprocess_pipeline, split_data
    from .features import feature_engineering_pipeline
    from .modeling import train_model
    
    # Pipeline
    df = preprocess_pipeline()
    df_fe, encoders = feature_engineering_pipeline(df)
    X_train, X_test, y_train, y_test = split_data(df_fe)
    
    # Entrenar
    model, scaler = train_model(X_train, y_train)
    
    # Evaluar
    metrics = evaluate_model(model, scaler, X_test, y_test)
    
    print("\n✓ Evaluación exitosa")
