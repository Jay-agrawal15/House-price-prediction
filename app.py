import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="California House Price Predictor 🏠",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
            color: #111827;
        }

        /* Title */
        .title {
            font-size: 2.3rem;
            font-weight: 700;
            color: #1e3a8a;
            text-align: center;
            margin-bottom: 1rem;
        }

        /* Subheader */
        .subtitle {
            font-size: 1.1rem;
            text-align: center;
            color: #374151;
            margin-bottom: 2rem;
        }

        /* Card style for inputs */
        .input-card {
            background-color: #ffffffcc;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        /* Button styling */
        div.stButton > button:first-child {
            background-color: #1d4ed8;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }

        div.stButton > button:first-child:hover {
            background-color: #2563eb;
            transform: scale(1.05);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1e3a8a;
            color: white;
        }
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #f9fafb;
            margin-bottom: 10px;
        }
        .sidebar-text {
            color: #dbeafe;
            font-size: 0.95rem;
        }
        .sidebar-link a {
            color: #93c5fd;
            text-decoration: none;
        }
        .sidebar-link a:hover {
            text-decoration: underline;
            color: #bfdbfe;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Make input labels dark */
    label[data-baseweb="label"] {
        color: #111827 !important;
        font-weight: 500;
    }

    /* Optional: make placeholder text dark too */
    input {
        color: #111827 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============ LOAD MODEL FILES ============
with open('label_encoder_ocean.pkl', 'rb') as file:
    label_encoder_ocean = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('xgb_model.pkl', 'rb') as file:
    xg_regressor = pickle.load(file)

# ============ SIDEBAR ============
st.sidebar.image('self.png', width=150)
st.sidebar.markdown("<div class='sidebar-title'>👨‍💻 Developed by</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-text'>**Jay Agrawal**</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-link'>[🌐 GitHub](https://github.com/Jay-agrawal15)</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-link'>[💼 LinkedIn](https://linkedin.com/in/jay-agrawal-87321a215)</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-title'>📘 About this App</div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class='sidebar-text'>
This app predicts the **median house price** in California using an
XGBoost regression model.  
<br><br>
Enter property details, and the model estimates the value based on
location, age, and demographics.  
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-text'>✨ Built with Streamlit & Machine Learning</div>", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown("<div class='title'>🏡 California House Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter property details below to estimate the median house value</div>", unsafe_allow_html=True)

# ============ INPUT SECTION ============
with st.container():
    #st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("📍 Longitude", format="%.4f")
        latitude = st.number_input("📍 Latitude", format="%.4f")
        house_median_age = st.slider("🏠 House Median Age", 0, 50)
        total_rooms = st.number_input("🛏️ Total Rooms", min_value=0, max_value=10000)

    with col2:
        total_bedrooms = st.number_input("🛌 Total Bedrooms", min_value=0, max_value=2000)
        population = st.number_input("👨‍👩‍👧 Population", min_value=0)
        households = st.number_input("🏠 Households", min_value=0)
        median_income = st.number_input("💰 Median Income", format="%.2f")

    #st.markdown("</div>", unsafe_allow_html=True)

# ============ OCEAN PROXIMITY ============
#st.markdown("<div class='input-card'>", unsafe_allow_html=True)
ocean_prox = st.selectbox("🌊 Ocean Proximity", label_encoder_ocean.classes_)
#st.markdown("</div>", unsafe_allow_html=True)

# Encode input
ocean_encoded = label_encoder_ocean.transform([ocean_prox])[0]

# ============ DATA PREPARATION ============
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [house_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_encoded],
})

scaled_input = scaler.transform(input_data)

# ============ PREDICTION ============
if st.button("🔍 Predict House Price"):
    prediction = xg_regressor.predict(scaled_input)
    predicted_value = prediction[0] * 100000

    st.success(f"💵 **Predicted Median House Value:** ${predicted_value:,.2f}")

    # Celebration animation
    st.balloons()

    # Display input summary below
    st.markdown("### 📊 Input Summary")
    st.dataframe(input_data.style.highlight_max(axis=0, color="#bbf7d0"))
