# ==============================
# utils.py - Auxiliary functions
# ==============================

import os
import joblib
import pandas as pd

# ==============================
# Block 1: Load trained pipeline
# ==============================
def load_model(model_path=None):
    """
    Load the trained model pipeline from disk.
    If model_path is None, it uses 'model_pipeline.pkl' in the same folder as this file.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el archivo {model_path}. Asegúrate de entrenar y guardar el pipeline primero.")
        return None
    
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"❌ Error cargando el pipeline: {e}")
        return None

# ==============================
# Block 2: Preprocess input (pipeline already handles preprocessing)
# ==============================
def preprocess_input(df):
    """
    Returns the DataFrame as-is because preprocessing is included in the pipeline.
    """
    return df

# ==============================
# Block 3: Predict price
# ==============================
def predict_price(pipeline, input_df):
    """
    Predict the property price using the trained pipeline.
    Returns a single value if prediction is length 1, otherwise the full prediction array.
    """
    if pipeline is None:
        return "❌ Modelo no cargado."

    try:
        prediction = pipeline.predict(input_df)
        if hasattr(prediction, "__len__") and len(prediction) == 1:
            return prediction[0]
        return prediction
    except Exception as e:
        return f"❌ Error en la predicción: {e}"
