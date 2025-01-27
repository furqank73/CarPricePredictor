import streamlit as st
import joblib
import pandas as pd

# Load Model and Encoders
data = joblib.load('random_forest_model.pkl')
model = data["model"]
encoders = data["encoders"]

# Streamlit Interface Styling
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Title and Subtitle
st.title("🚗 Car Price Prediction")
st.markdown("### Enter car details to predict its price in lacs.")

# Layout with columns for better spacing
col1, col2 = st.columns([1, 3])

# Dropdowns using Encoders (Column 1)
with col1:
    st.markdown("#### Select Car Details")
    brand_model = st.selectbox('Brand Model', encoders['brand_model'].classes_, key="brand_model")
    city = st.selectbox('City', encoders['city'].classes_, key="city")
    transmission = st.selectbox('Transmission', encoders['Transmission'].classes_, key="transmission")
    price_type = st.selectbox('Price Type', encoders['price_type'].classes_, key="price_type")

# Numerical Inputs (Column 2)
with col2:
    st.markdown("#### Enter Car Specifications")
    model_year = st.number_input('Model Year', min_value=1990, max_value=2025, step=1, key="model_year")
    mileage_in_km = st.number_input('Mileage (in km)', min_value=0, max_value=500000, step=1000, key="mileage_in_km")
    engine_in_cc = st.number_input('Engine Size (in cc)', min_value=800, max_value=5000, step=100, key="engine_in_cc")

# Prediction Button
if st.button('🔮 Predict Price'):
    # Prepare Input Data
    input_data = pd.DataFrame({
        'brand_model': [brand_model],
        'city': [city],
        'model': [model_year],
        'mileage_in_km': [mileage_in_km],
        'engine_in_cc': [engine_in_cc],
        'Transmission': [transmission],
        'price_type': [price_type]
    })

    # Encode Inputs
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)
    
    # Display Prediction with Colorful Format
    st.markdown(f"### Predicted Price: **{prediction[0]:.2f} lacs**", unsafe_allow_html=True)
    st.success("Prediction Successful! 🎉", icon="✅")

# Styling the page (optional)
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #0073e6;
            color: white;
            font-weight: bold;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
    </style>
    """, unsafe_allow_html=True
)
