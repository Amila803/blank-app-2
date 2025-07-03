import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import os
import sys
import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


# Constants
MODEL_PATH = 'travel_cost_model.joblib'
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Travel_details_dataset.csv"

# Destination list extracted from the dataset
DESTINATIONS = [
    'London', 'Phuket', 'Bali', 'New York', 'Tokyo', 'Paris', 'Sydney',
    'Rio de Janeiro', 'Amsterdam', 'Dubai', 'Cancun', 'Barcelona',
    'Honolulu', 'Berlin', 'Marrakech', 'Edinburgh', 'Rome', 'Bangkok',
    'Cape Town', 'Vancouver', 'Seoul', 'Los Angeles', 'Santorini',
    'Phnom Penh', 'Athens', 'Auckland'
]

# Nationality list extracted from the dataset
NATIONALITIES = sorted(list(set([
    'American', 'Canadian', 'Korean', 'British', 'Vietnamese', 'Australian',
    'Brazilian', 'Dutch', 'Emirati', 'Mexican', 'Spanish', 'Chinese',
    'German', 'Moroccan', 'Scottish', 'Indian', 'Italian', 'South Korean',
    'Taiwanese', 'South African', 'French', 'Japanese', 'Cambodia', 'Greece',
    'United Arab Emirates', 'Hong Kong', 'Singapore', 'Indonesia', 'USA',
    'UK', 'China', 'New Zealander'
])))

ACCOMMODATION_TYPES = [
    'Hotel', 'Resort', 'Villa', 'Airbnb', 'Hostel', 'Riad',
    'Guesthouse', 'Vacation rental'
]

TRANSPORTATION_TYPES = [
    'Flight', 'Train', 'Plane', 'Bus', 'Car rental', 'Subway',
    'Ferry', 'Car', 'Airplane'
]

def clean_cost(value):
    """Clean cost values that might contain currency symbols, commas, or text"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove currency symbols and text like 'USD'
        cleaned = re.sub(r'[^\d.]', '', value.split('USD')[0].strip())
        return float(cleaned) if cleaned else np.nan
    return float(value)

def clean_destination(dest):
    """Clean destination names by extracting the main location"""
    if pd.isna(dest):
        return np.nan
    dest = str(dest).split(',')[0].strip()
    # Handle special cases
    if dest == 'New York City':
        return 'New York'
    elif dest == 'Sydney, Aus' or dest == 'Sydney, AUS':
        return 'Sydney'
    elif dest == 'Bangkok, Thai':
        return 'Bangkok'
    elif dest == 'Phuket, Thai':
        return 'Phuket'
    elif dest == 'Cape Town, SA':
        return 'Cape Town'
    return dest

def load_data():
    """Load the dataset with file upload fallback"""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    else:
        upload = st.file_uploader("Upload Travel_details_dataset.csv", type="csv")
        if upload:
            df = pd.read_csv(upload)
        else:
            st.warning("Please upload the dataset to continue")
            st.stop()
    
    # Basic data validation
    df.columns = df.columns.str.strip()
    if "Destination" not in df.columns:
        st.error("Column `Destination` not found. Available: " + ", ".join(df.columns))
        st.stop()
    
    with st.expander("📋 Show sample data"):
        st.dataframe(df.head(10))
    
    return df

def preprocess_data(df):
    """Preprocess the loaded dataset"""
    # Data cleaning - remove empty rows
    df = df.dropna(how='all')
    
    # Clean cost columns
    df['Accommodation cost'] = df['Accommodation cost'].apply(clean_cost)
    df['Transportation cost'] = df['Transportation cost'].apply(clean_cost)
    
    # Calculate total cost
    df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']
    
    # Clean and standardize columns
    df['Destination'] = df['Destination'].apply(clean_destination)
    df['Traveler nationality'] = df['Traveler nationality'].str.split().str[0].str.strip()
    df['Accommodation type'] = df['Accommodation type'].str.strip()
    df['Transportation type'] = df['Transportation type'].str.strip().replace({
        'Plane': 'Flight',
        'Airplane': 'Flight'
    })
    
    # Date handling
    df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df = df.dropna(subset=['Start date', 'End date'])
    
    # Feature engineering
    df['Year'] = df['Start date'].dt.year
    df['Month'] = df['Start date'].dt.month
    df['Season'] = df['Start date'].dt.month % 12 // 3 + 1
    df['Duration (days)'] = (df['End date'] - df['Start date']).dt.days
    
    # Additional features
    df['Is_peak_season'] = df['Month'].isin([6, 7, 8, 12]).astype(int)
    df['Is_long_trip'] = (df['Duration (days)'] > 14).astype(int)
    df['Is_domestic'] = (df['Traveler nationality'] == df['Destination']).astype(int)
    
    # Select features and target
    features = ['Destination', 'Traveler nationality', 'Duration (days)', 
               'Accommodation type', 'Transportation type', 'Year', 
               'Month', 'Season', 'Is_peak_season', 'Is_long_trip', 'Is_domestic']
    target = 'Total cost'
    
    # Final cleaning
    df = df.dropna(subset=features + [target])
    df = df[(df['Total cost'] > 0) & (df['Total cost'] < 50000)]
    df = df[(df['Duration (days)'] > 0) & (df['Duration (days)'] <= 90)]
    
    return df[features], df[target]

def evaluate_model(model, X, y):
    """Evaluate model performance using cross-validation"""
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        mae_scores = -scores
        st.write(f"Mean Absolute Error (CV): ${mae_scores.mean():.2f} (± {mae_scores.std():.2f})")
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        st.write(f"R² Score (CV): {scores.mean():.2f} (± {scores.std():.2f})")
    except Exception as e:
        st.warning(f"Could not perform cross-validation: {str(e)}")

def train_model():
    df = load_data()
    X, y = preprocess_data(df)
    
    if X is None or y is None:
        st.error("Cannot train model due to data issues.")
        return None
        
    categorical_features = ['Destination', 'Traveler nationality', 'Accommodation type', 'Transportation type']
    numerical_features = ['Duration (days)', 'Year', 'Month', 'Season', 'Is_peak_season', 'Is_long_trip', 'Is_domestic']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    # Define base models for stacking
    base_models = [
        ('random_forest', RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )),
        ('svr', SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1
        )),
        ('ridge', Ridge(
            alpha=1.0,
            random_state=42
        ))
    ]
    
    # Define meta-model
    meta_model = LinearRegression()

    param_grid = {
    # Random Forest
    'regressor__estimators__random_forest__n_estimators': [100, 200, 300],
    'regressor__estimators__random_forest__max_depth': [10, 15, None],
    'regressor__estimators__random_forest__min_samples_split': [2, 5, 10],

    # SVR
    'regressor__estimators__svr__C': [0.1, 1.0, 10.0],
    'regressor__estimators__svr__epsilon': [0.01, 0.1, 1.0],

    # Ridge
    'regressor__estimators__ridge__alpha': [0.1, 1.0, 10.0],

    # Meta-model (LinearRegression has no hyperparameters, but if you switch to Ridge for meta you could tune its alpha here)
    # 'regressor__final_estimator__alpha': [0.1, 1.0, 10.0],
}

    Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=True
    ))
])

    
    # Create stacking ensemble
    stacked_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            n_jobs=-1,
            passthrough=True
        ))
    ])
    
    # Train model
    with st.spinner("Training model (this may take a few minutes)..."):
        stacked_model.fit(X, y)
    
    # Evaluate model
    st.subheader("Model Evaluation")
    evaluate_model(stacked_model, X, y)
    
    try:
        joblib.dump(stacked_model, MODEL_PATH)
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        
    return stacked_model

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found. Training a new model...")
        return train_model()
        
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Attempting to train a new model...")
        return train_model()

def predict_cost(model, nationality, destination, start_date, duration, accommodation, transportation):
    if model is None:
        st.error("Cannot make predictions - no model available")
        return None
        
    try:
        # Calculate additional features
        month = start_date.month
        season = (month % 12) // 3 + 1
        is_peak_season = 1 if month in [6, 7, 8, 12] else 0
        is_long_trip = 1 if duration > 14 else 0
        is_domestic = 1 if nationality == destination else 0
        
        # Standardize transportation type
        transportation = 'Flight' if transportation in ['Plane', 'Airplane'] else transportation
        
        input_data = pd.DataFrame({
            'Destination': [destination],
            'Traveler nationality': [nationality],
            'Duration (days)': [duration],
            'Accommodation type': [accommodation],
            'Transportation type': [transportation],
            'Year': [start_date.year],
            'Month': [month],
            'Season': [season],
            'Is_peak_season': [is_peak_season],
            'Is_long_trip': [is_long_trip],
            'Is_domestic': [is_domestic]
        })
        
        prediction = model.predict(input_data)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title('Travel Cost Estimator')
    st.write("Estimate the total cost of your trip based on your nationality, destination, dates, and accommodation type.")
    
    # Sidebar for model management
    with st.sidebar:
        st.header("Model Management")
        if st.button("Retrain Model"):
            model = train_model()
        st.info("Click the button above to retrain the model with the latest data.")
    
    # Load or train model
    model = load_model()
    
    if model is None:
        st.error("Failed to initialize model. Cannot continue.")
        return
    
    with st.form("travel_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nationality = st.selectbox('Your Nationality', NATIONALITIES, index=0)
            destination = st.selectbox('Destination', DESTINATIONS)
            start_date = st.date_input('Start Date', min_value=datetime.today())
        
        with col2:
            duration = st.number_input('Duration (days)', min_value=1, max_value=90, value=7)
            accommodation = st.selectbox('Accommodation Type', ACCOMMODATION_TYPES)
            transportation = st.selectbox('Transportation Type', TRANSPORTATION_TYPES)
        
        submitted = st.form_submit_button("Estimate Cost")
    
    if submitted:
        total_cost = predict_cost(
            model, nationality, destination, start_date, duration, accommodation, transportation
        )
        
        if total_cost is not None:
            st.subheader("Estimated Cost")
            st.metric(label="Total Estimated Cost", value=f"${total_cost:,.2f}")
            
            st.write("**Estimated Cost Breakdown:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Accommodation", value=f"${total_cost * 0.7:,.2f}")
            with col2:
                st.metric(label="Transportation", value=f"${total_cost * 0.3:,.2f}")
            
            # Show if this is a domestic trip
            if nationality == destination:
                st.info("This is a domestic trip (same nationality and destination)")
            else:
                st.info("This is an international trip")

if __name__ == '__main__':
    main()
