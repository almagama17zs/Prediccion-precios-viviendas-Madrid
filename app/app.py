import os
import sys
import streamlit as st
import pandas as pd

# ==============================
# Ensure app folder is in sys.path
# ==============================
sys.path.append(os.path.dirname(__file__))  # permite importar utils.py

# ==============================
# Imports from utils
# ==============================
import utils  # usamos utils.load_model, utils.preprocess_input, utils.predict_price

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
Introduce las características de la propiedad y obtén un precio estimado.
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
# Load Model
# ==============================
@st.cache_resource
def get_model():
    pipeline_path = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
    pipeline = utils.load_model(pipeline_path)
    if pipeline is None:
        st.error("❌ No se pudo cargar el modelo. Verifica que 'model_pipeline.pkl' exista en la carpeta app.")
    return pipeline

model = get_model()

# ==============================
# Make Prediction
# ==============================
if st.button("Predecir precio"):

    if model is None:
        st.error("❌ Modelo no disponible. No se puede predecir.")
    else:
        # 🔹 Mapear inputs a los nombres que espera el pipeline
        input_data = {
            "habitaciones": rooms,
            "metros": size,
            "baños": bathrooms,
            "zona": district
        }
        input_df = pd.DataFrame([input_data])

        # 🔹 Reordenar columnas según lo que espera el pipeline
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
# Optional: Show Data / Feature Info
# ==============================
with st.expander("ℹ️ Información sobre las características"):
    st.write("""
    - **Número de habitaciones**: Total de habitaciones en la propiedad  
    - **Tamaño (m²)**: Superficie total en metros cuadrados  
    - **Número de baños**: Total de baños  
    - **Distrito**: Zona de Madrid
    """)
