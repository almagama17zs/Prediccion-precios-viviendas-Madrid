import os
import sys
import streamlit as st
import pandas as pd
import gdown  # to download the model from Google Drive

# ==============================
# Ensure app folder is in sys.path
# ==============================
sys.path.append(os.path.dirname(__file__))

# ==============================
# Imports from utils
# ==============================
import utils  # using utils.load_model, utils.preprocess_input, utils.predict_price

# ==============================
# Load Custom CSS (from assets/style.css)
# ==============================
def load_css():
    """Load external CSS file for sidebar and layout styling."""
    css_path = os.path.join(os.path.dirname(__file__), "assets/style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

load_css()

# ==============================
# Sidebar Logo
# ==============================
logo_path = os.path.join(os.path.dirname(__file__), "assets/logo.png")
if os.path.exists(logo_path):
    # Show logo at the top of the sidebar
    st.sidebar.image(logo_path, use_container_width=True)

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
Predice el precio de viviendas en Madrid utilizando un modelo de machine learning entrenado.  
Introduce las características de la propiedad para obtener un precio estimado.
""")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Características de la Propiedad")
rooms = st.sidebar.number_input("Número de habitaciones", min_value=1, max_value=10, value=3)
size = st.sidebar.number_input("Tamaño (m²)", min_value=10, max_value=1000, value=70)
bathrooms = st.sidebar.number_input("Número de baños", min_value=1, max_value=5, value=1)
district = st.sidebar.selectbox("Distrito", ["Centro", "Chamartín", "Salamanca", "Retiro", "Latina"])

# ==============================
# Google Drive model ID and paths
# ==============================
MODEL_ID = "1Nn59vaxw_arH-KBgN53wbnH9R68L9SSu"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
DOWNLOAD_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# ==============================
# Function to download model if missing
# ==============================
def download_model(path, url):
    """Download model file only if not already found locally."""
    if not os.path.exists(path):
        st.info("Descargando el modelo desde Google Drive…")
        gdown.download(url, path, quiet=False)
        st.success("Modelo descargado correctamente.")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def get_model():
    """Load the trained ML model from local file or download it."""
    download_model(MODEL_PATH, DOWNLOAD_URL)
    pipeline = utils.load_model(MODEL_PATH)
    if pipeline is None:
        st.error("❌ No se pudo cargar el modelo.")
    return pipeline

model = get_model()

# ==============================
# Make Prediction
# ==============================
if st.button("Predecir precio"):
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
            st.success(f"🏷️ Precio estimado: €{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")

# ==============================
# Optional: Show Feature Info
# ==============================
with st.expander("ℹ️ Información sobre las características"):
    st.write("""
    - **Número de habitaciones**: Total de habitaciones en la propiedad  
    - **Tamaño (m²)**: Superficie total en metros cuadrados  
    - **Número de baños**: Total de baños  
    - **Distrito**: Zona de Madrid
    """)
