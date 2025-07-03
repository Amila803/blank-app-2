import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="centered")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("Predict your travel costs based on destination, duration, and other factors.")

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).parent / "models" / "travel_cost_predictor.pkl"
        
        if not model_path.exists():
            st.error(f"❌ Model file not found at {model_path}")
            return None
            
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.stop()  # Don't proceed if model failed to load

# Define possible options
NATIONALITIES = ['American', 'British', 'Canadian', 'Australian', 'Japanese', 'Chinese', 'Indian', 'German']
DESTINATIONS = ['London', 'Paris', 'Tokyo', 'New York', 'Bali', 'Sydney', 'Bangkok', 'Rome']
ACCOMMODATION_TYPES = ['Hotel', 'Airbnb', 'Resort', 'Hostel', 'Villa']
TRANSPORTATION_TYPES = ['Flight', 'Train', 'Bus', 'Car rental']

# User inputs form
with st.form("travel_form"):
    st.subheader("Trip Details")
    
    col1, col2 = st.columns(2)
    with col1:
        nationality = st.selectbox("Nationality", NATIONALITIES)
        destination = st.selectbox("Destination", DESTINATIONS)
        start_date = st.date_input("Start Date", datetime.today())
    with col2:
        days = st.number_input("Duration (days)", min_value=1, max_value=90, value=7)
        accommodation_type = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES)
        transportation_type = st.selectbox("Transportation Type", TRANSPORTATION_TYPES)
    
    submitted = st.form_submit_button("Calculate Costs")
    reset = st.form_submit_button("Reset")

# Reset functionality
if reset:
    st.experimental_rerun()

# Prediction logic
if submitted and model is not None:
    try:
        # Prepare input data
        input_data = {
            'Destination': [destination],
            'Traveler nationality': [nationality],
            'Duration (days)': [days],
            'Accommodation type': [accommodation_type],
            'Transportation type': [transportation_type],
            'Start month': [start_date.month],
            'Is_peak_season': [1 if start_date.month in [6, 7, 8, 12] else 0]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Make prediction (assuming model predicts total cost directly)
        predicted_cost = model.predict(input_df)[0]
        
        # Display results
        st.success("### Cost Estimation Results")
        st.metric("Total Estimated Cost", f"${predicted_cost:,.2f}")
        
        # Optional: Add more visualizations if your model provides breakdown
        st.subheader("Cost Breakdown")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Total Cost"], [predicted_cost], color='#1f77b4')
        ax.set_ylabel('USD')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.code("Please check if your input data matches the model's training format")
