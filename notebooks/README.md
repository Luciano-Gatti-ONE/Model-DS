# Notebooks

Este directorio está preparado para contener notebooks de Jupyter para análisis exploratorio y entrenamiento.

## 📝 Notebooks Sugeridos (A Crear)

### 00_eda.ipynb (Recomendado)
Análisis Exploratorio de Datos (EDA):
- Cargar dataset desde `data/raw/`
- Visualizaciones de distribuciones
- Análisis de correlaciones
- Identificación de patrones temporales
- Análisis de features climáticas

### 01_train_model.ipynb (Recomendado)
Pipeline completo de entrenamiento:
```python
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.modeling import train_model, save_model
from src.evaluation import evaluate_model

# Pipeline completo en notebook interactivo
```

## 🚀 Crear Notebooks

### Opción 1: Crear desde cero
```bash
cd notebooks
jupyter notebook
# Crear nuevo notebook con el nombre deseado
```

### Opción 2: Copiar notebooks existentes
Si ya tienes notebooks de EDA o entrenamiento en otra parte del proyecto:
```bash
copy "..\tu_notebook.ipynb" "notebooks\00_eda.ipynb"
```

## 💡 Ventaja de usar los módulos src/

Los notebooks pueden ser **muy simples** porque toda la lógica está en `src/`:

```python
# Ejemplo de notebook limpio
import sys
sys.path.append('..')

from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline  
from src.modeling import train_model
from src.evaluation import evaluate_model

# Todo el código complejo está encapsulado en módulos
df = preprocess_pipeline()
df_fe, encoders = feature_engineering_pipeline(df)
# ... etc
```

## 📂 Estado Actual

**Notebooks en este directorio**: Ninguno (directorio vacío)

Para empezar, crea tu primer notebook con:
```bash
jupyter notebook
```
