import os
import sys
import streamlit as st
import pandas as pd
import requests

# ==============================
# Ensure app folder is in sys.path
# ==============================
sys.path.append(os.path.dirname(__file__))

# ==============================
# Imports from utils
# ==============================
import utils  # using utils.load_model, utils.preprocess_input, utils.predict_price

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Predicción de Precios de Viviendas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Title and Description
# ==============================
st.title("🏠 Predicción de Precios de Viviendas - Madrid")
st.markdown("""
Predict property prices in Madrid using a trained machine learning model.  
Enter the property features to get an estimated price.
""")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Property Features")
rooms = st.sidebar.number_input("Number of rooms", min_value=1, max_value=10, value=3)
size = st.sidebar.number_input("Size (m²)", min_value=10, max_value=1000, value=70)
bathrooms = st.sidebar.number_input("Number of bathrooms", min_value=1, max_value=5, value=1)
district = st.sidebar.selectbox("District", ["Centro", "Chamartín", "Salamanca", "Retiro", "Latina"])

# ==============================
# Google Drive model ID
# ==============================
MODEL_ID = "1P1W_vC38Jl8Gdrtv-8rl9WR8_PjBGuve"  # Replace with actual file ID
DOWNLOAD_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# ==============================
# Function to download model if missing
# ==============================
def download_model(path, url):
    if not os.path.exists(path):
        st.info("Downloading model from Google Drive...")
        r = requests.get(url, allow_redirects=True)
        with open(path, "wb") as f:
            f.write(r.content)
        st.success("Model downloaded successfully.")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def get_model():
    pipeline_path = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
    
    # Download if not exists
    download_model(pipeline_path, DOWNLOAD_URL)
    
    pipeline = utils.load_model(pipeline_path)
    if pipeline is None:
        st.error("❌ Could not load the model.")
    return pipeline

model = get_model()

# ==============================
# Make Prediction
# ==============================
if st.button("Predict Price"):
    input_data = {
        "habitaciones": rooms,
        "metros": size,
        "baños": bathrooms,
        "zona": district
    }
    input_df = pd.DataFrame([input_data])

    # Reorder columns according to model expectations
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_.tolist()
        input_df = input_df[expected_cols]

    try:
        processed_input = utils.preprocess_input(input_df)
        predicted_price = utils.predict_price(model, processed_input)
        if isinstance(predicted_price, str):
            st.error(predicted_price)
        else:
            st.success(f"🏷️ Estimated price: €{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# ==============================
# Optional: Show Feature Info
# ==============================
with st.expander("ℹ️ Property Features Info"):
    st.write("""
    - **Number of rooms**: Total rooms in the property  
    - **Size (m²)**: Total surface in square meters  
    - **Number of bathrooms**: Total bathrooms  
    - **District**: Area of Madrid
    """)
