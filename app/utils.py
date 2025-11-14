# ==============================
# utils.py - Auxiliary functions
# ==============================

import os
import joblib
import requests

# ==============================
# Block 1: Download model from Google Drive
# ==============================
def download_model_from_drive(file_id, destination):
    """
    Download a file from Google Drive using the file ID.
    """
    URL = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(URL, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file from Drive, status code {response.status_code}")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            f.write(chunk)

# ==============================
# Block 2: Load trained pipeline
# ==============================
def load_model(model_path=None, drive_file_id=None):
    """
    Load the trained model pipeline from disk.
    If not found locally, download it from Google Drive.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
    
    if not os.path.exists(model_path):
        if drive_file_id is None:
            print(f"❌ Model not found locally and no Drive ID provided.")
            return None
        print("⬇️ Model not found locally. Downloading from Google Drive...")
        download_model_from_drive(drive_file_id, model_path)

    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"❌ Error loading the pipeline: {e}")
        return None

# ==============================
# Block 3: Preprocess input (pipeline handles preprocessing)
# ==============================
def preprocess_input(df):
    """
    Return the DataFrame as-is; preprocessing is handled inside the pipeline.
    """
    return df

# ==============================
# Block 4: Predict price
# ==============================
def predict_price(pipeline, input_df):
    """
    Predict property price using the trained pipeline.
    Returns a single value if the prediction length is 1.
    """
    if pipeline is None:
        return "❌ Model not loaded."
    try:
        prediction = pipeline.predict(input_df)
        if hasattr(prediction, "__len__") and len(prediction) == 1:
            return prediction[0]
        return prediction
    except Exception as e:
        return f"❌ Prediction error: {e}"
