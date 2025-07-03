import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
Predict accommodation and transportation costs for your trips based on historical data.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Trip Details")
    destination = st.selectbox("Destination", ["London", "Paris", "Tokyo", "New York", "Bali", "Sydney", "Bangkok"])
    start_date = st.date_input("Start Date", datetime.today())
    duration = st.slider("Duration (days)", 1, 30, 7)
    traveler_age = st.slider("Traveler Age", 18, 80, 35)
    traveler_gender = st.selectbox("Traveler Gender", ["Male", "Female", "Other"])
    traveler_nationality = st.selectbox("Traveler Nationality", ["American", "British", "Canadian", "Australian", "Japanese", "Chinese"])
    accommodation_type = st.selectbox("Accommodation Type", ["Hotel", "Airbnb", "Resort", "Hostel", "Villa"])
    transportation_type = st.selectbox("Transportation Type", ["Flight", "Train", "Bus", "Car rental"])

# Load and preprocess data
@st.cache_data
def load_data():
    # This would be your actual data loading code
    # For demo, we'll create a sample dataset
    data = pd.read_csv('Travel_details_dataset.csv')
    
    # Data cleaning
    for col in ['Accommodation cost', 'Transportation cost']:
        data[col] = data[col].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
    
    # Feature engineering
    data['Start date'] = pd.to_datetime(data['Start date'])
    data['Start month'] = data['Start date'].dt.month
    data['Is_peak_season'] = data['Start month'].isin([6, 7, 8, 12]).astype(int)
    data['Total cost'] = data['Accommodation cost'] + data['Transportation cost']
    
    # Target encoding for destinations
    destination_means = data.groupby('Destination')['Total cost'].mean().to_dict()
    data['Destination_encoded'] = data['Destination'].map(destination_means)
    
    return data

@st.cache_resource
def train_models(data):
    # Define features and target
    features = ['Destination_encoded', 'Duration (days)', 'Traveler age', 
                'Traveler gender', 'Traveler nationality', 'Accommodation type',
                'Transportation type', 'Is_peak_season']
    
    X = data[features]
    y_accom = data['Accommodation cost']
    y_trans = data['Transportation cost']
    y_total = data['Total cost']
    
    # Preprocessing
    numeric_features = ['Destination_encoded', 'Duration (days)', 'Traveler age', 'Is_peak_season']
    categorical_features = ['Traveler gender', 'Traveler nationality', 
                          'Accommodation type', 'Transportation type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Train models
    gbm_accom = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gbm_trans = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    rf_total = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    
    pipeline_accom = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', gbm_accom)])
    pipeline_trans = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', gbm_trans)])
    pipeline_total = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', rf_total)])
    
    pipeline_accom.fit(X, y_accom)
    pipeline_trans.fit(X, y_trans)
    pipeline_total.fit(X, y_total)
    
    return pipeline_accom, pipeline_trans, pipeline_total

# Load data and train models
data = load_data()
pipeline_accom, pipeline_trans, pipeline_total = train_models(data)

# Prepare input for prediction
def prepare_input():
    # Get destination encoding (fallback to mean if new destination)
    destination_mean = data['Total cost'].mean()
    dest_encoded = data.groupby('Destination')['Total cost'].mean().get(destination, destination_mean)
    
    # Determine peak season
    month = start_date.month
    is_peak = 1 if month in [6, 7, 8, 12] else 0
    
    input_data = {
        'Destination_encoded': [dest_encoded],
        'Duration (days)': [duration],
        'Traveler age': [traveler_age],
        'Traveler gender': [traveler_gender],
        'Traveler nationality': [traveler_nationality],
        'Accommodation type': [accommodation_type],
        'Transportation type': [transportation_type],
        'Is_peak_season': [is_peak]
    }
    
    return pd.DataFrame(input_data)

# Make predictions
if st.button("Predict Costs"):
    input_df = prepare_input()
    
    accom_pred = pipeline_accom.predict(input_df)[0]
    trans_pred = pipeline_trans.predict(input_df)[0]
    total_pred = pipeline_total.predict(input_df)[0]
    
    # Display results
    st.success("### Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accommodation Cost", f"${accom_pred:,.2f}")
    with col2:
        st.metric("Transportation Cost", f"${trans_pred:,.2f}")
    with col3:
        st.metric("Total Estimated Cost", f"${total_pred:,.2f}", delta_color="off")
    
    # Show cost breakdown
    fig, ax = plt.subplots(figsize=(8, 4))
    costs = [accom_pred, trans_pred]
    labels = ['Accommodation', 'Transportation']
    ax.bar(labels, costs, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Cost Breakdown')
    ax.set_ylabel('USD')
    st.pyplot(fig)

# Data visualization section
st.markdown("---")
st.header("Data Insights")

tab1, tab2, tab3 = st.tabs(["Cost Distributions", "Top Destinations", "Seasonal Trends"])

with tab1:
    st.subheader("Cost Distributions")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data['Accommodation cost'], bins=30, ax=ax[0], kde=True)
    ax[0].set_title('Accommodation Costs')
    sns.histplot(data['Transportation cost'], bins=30, ax=ax[1], kde=True)
    ax[1].set_title('Transportation Costs')
    st.pyplot(fig)

with tab2:
    st.subheader("Top Destinations by Average Cost")
    top_dests = data.groupby('Destination')['Total cost'].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_dests.sort_values().plot(kind='barh', ax=ax, color='#2ca02c')
    ax.set_xlabel('Average Total Cost (USD)')
    st.pyplot(fig)

with tab3:
    st.subheader("Seasonal Cost Variations")
    seasonal_data = data.groupby('Is_peak_season')['Total cost'].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    seasonal_data.plot(kind='bar', ax=ax, color=['#9467bd', '#8c564b'])
    ax.set_xticklabels(['Off-Peak', 'Peak'], rotation=0)
    ax.set_ylabel('Average Total Cost (USD)')
    st.pyplot(fig)

# Model performance section
st.markdown("---")
st.header("Model Performance")

# This would be replaced with your actual model metrics
performance_data = {
    'Model': ['Accommodation', 'Transportation', 'Total'],
    'MAE': [343.10, 266.22, 525.13],
    'R² Score': [-0.23, 0.09, -0.11]
}

st.table(pd.DataFrame(performance_data))
st.warning("Note: These are example metrics. Your actual model performance may vary based on data quality and feature engineering.")
