import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs using a unified RandomForest model for both accommodation and transportation.
""")

# Load data function
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig')
        data = data.dropna(how='all')
        
        def clean_currency(value):
            if isinstance(value, str):
                value = value.replace('USD', '').replace(',', '').replace('$', '').strip()
                try:
                    return float(value)
                except:
                    return None
            return value
        
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
        
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce', format='mixed')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce', format='mixed')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        transport_mapping = {
            'Plane': 'Flight', 'Airplane': 'Flight', 'Car': 'Car rental',
            'Subway': 'Train', 'Bus': 'Bus', 'Train': 'Train', 'Ferry': 'Ferry'
        }
        data['Transportation type'] = data['Transportation type'].replace(transport_mapping)
        
        data = data.rename(columns={
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation type': 'TransportType',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate'
        })
        
        data = data[[
            'Destination', 'Duration', 'StartDate', 'AccommodationType',
            'TravelerNationality', 'Cost', 'TransportType', 'TransportCost'
        ]].dropna(subset=['Cost', 'TransportCost'])
        
        # Remove extreme outliers
        cost_q1, cost_q3 = data['Cost'].quantile([0.05, 0.95])
        transport_q1, transport_q3 = data['TransportCost'].quantile([0.05, 0.95])
        data = data[
            (data['Cost'] >= cost_q1) & (data['Cost'] <= cost_q3) &
            (data['TransportCost'] >= transport_q1) & (data['TransportCost'] <= transport_q3)
        ]
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def engineer_features(df):
    df = df.copy()
    df['Year'] = df['StartDate'].dt.year
    df['Month'] = df['StartDate'].dt.month
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
    return df

# Load data
data = load_data()

if data is not None:
    engineered_data = engineer_features(data)
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())

    # Create unified preprocessor
    def create_preprocessor(categorical_features):
        return ColumnTransformer([
            ('num', StandardScaler(), ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ])

    # Unified model training function
    def train_model(X, y, model_name):
        categorical_features = list(X.select_dtypes(include=['object']).columns)
        preprocessor = create_preprocessor(categorical_features)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        
        # Save model
        joblib.dump(best_model, f'{model_name}_model.pkl')
        
        return best_model

    # Train models button
    if st.button("Train Models"):
        with st.spinner("Training models with hyperparameter tuning..."):
            # Train accommodation model
            X_accom = engineered_data[['Destination', 'Duration', 'AccommodationType', 
                                     'TravelerNationality', 'Month', 'IsWeekend', 'IsPeakSeason']]
            y_accom = engineered_data['Cost']
            accom_model = train_model(X_accom, y_accom, 'accom')
            
            # Train transport model
            transport_data = data[['Destination', 'Duration', 'TransportType', 
                                 'TravelerNationality', 'StartDate']].copy()
            transport_data['PeakSeason'] = transport_data['StartDate'].dt.month.isin([6,7,8,12]).astype(int)
            X_trans = transport_data[['Destination', 'Duration', 'TransportType', 
                                    'TravelerNationality', 'PeakSeason']]
            y_trans = transport_data['TransportCost']
            trans_model = train_model(X_trans, y_trans, 'trans')
            
            st.success("Both models trained and saved!")
            
            # Evaluate accommodation model
            X_train, X_test, y_train, y_test = train_test_split(
                X_accom, y_accom, test_size=0.2, random_state=42)
            y_pred = accom_model.predict(X_test)
            
            st.subheader("Accommodation Model Evaluation")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            
            with col2:
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                ax.set_xlabel('Actual Cost')
                ax.set_ylabel('Predicted Cost')
                st.pyplot(fig)

    # Prediction Interface
    st.header("Cost Prediction")

    with st.form("prediction_form"):
        st.subheader("Enter Trip Details")
        
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", DESTINATIONS)
            duration = st.number_input("Duration (days)", min_value=1, max_value=90, value=7)
            accommodation = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES)
            nationality = st.selectbox("Nationality", NATIONALITIES)
        
        with col2:
            start_date = st.date_input("Start Date", datetime.today())
            month = start_date.month
            day_of_week = start_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_season = 1 if month in [6,7,8,12] else 0
            transport_type = st.selectbox("Transportation Type", TRANSPORT_TYPES)
        
        submitted = st.form_submit_button("Calculate Costs")

    if submitted:
        try:
            # Load models
            accom_model = joblib.load('accom_model.pkl')
            trans_model = joblib.load('trans_model.pkl')
            
            # Accommodation prediction
            accom_input = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'AccommodationType': accommodation,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': is_weekend,
                'IsPeakSeason': is_peak_season
            }])
            accom_pred = accom_model.predict(accom_input)[0]
            
            # Transport prediction
            trans_input = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'TransportType': transport_type,
                'TravelerNationality': nationality,
                'PeakSeason': is_peak_season
            }])
            trans_pred = trans_model.predict(trans_input)[0]
            
            total_cost = accom_pred + trans_pred
            
            # Display results
            st.success(f"## Total Estimated Trip Cost: ${total_cost:,.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accommodation Cost", f"${accom_pred:,.2f}")
                st.write(f"Type: {accommodation}")
                st.write(f"Duration: {duration} days")
            
            with col2:
                st.metric("Transportation Cost", f"${trans_pred:,.2f}")
                st.write(f"Type: {transport_type}")
                st.write(f"Season: {'Peak' if is_peak_season else 'Off-peak'}")
            
            # Cost breakdown
            st.subheader("Cost Breakdown")
            fig, ax = plt.subplots()
            costs = [accom_pred, trans_pred]
            labels = ['Accommodation', 'Transportation']
            ax.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please train the models first by clicking the 'Train Models' button")

else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists in the same directory.")
