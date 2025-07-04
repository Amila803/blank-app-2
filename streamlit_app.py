import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
from sklearn.feature_selection import SelectFromModel

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs quickly using optimized models.
""")

# Data loading (unchanged)
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
                    return np.nan
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
        
        # Remove top/bottom 5% outliers
        for col in ['Cost', 'TransportCost']:
            q1 = data[col].quantile(0.05)
            q3 = data[col].quantile(0.95)
            data = data[(data[col] >= q1) & (data[col] <= q3)]
        
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Simplified feature engineering
def engineer_features(df):
    df = df.copy()
    df['Month'] = df['StartDate'].dt.month
    df['IsWeekend'] = df['StartDate'].dt.dayofweek.isin([5,6]).astype(int)
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

    # Create preprocessor
    def create_preprocessor():
        numeric_features = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason']
        categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])
        
        return ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ])

    # Faster model training with RandomizedSearchCV
    def train_fast_model(X, y, model_name):
        preprocessor = create_preprocessor()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        
        # Reduced parameter space with more reasonable defaults
        param_dist = {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['sqrt', None]
        }
        
        # Using RandomizedSearchCV with fewer iterations
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=10,  # Reduced from exhaustive search
            cv=3,       # Fewer folds
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y)
        best_model = search.best_estimator_
        
        joblib.dump(best_model, f'travel_cost_model.pkl')
        
        return best_model, search.best_params_

    # Train models button
    if st.button("Train Models (Fast)"):
        with st.spinner("Training optimized models (this will be faster)..."):
            # Train accommodation model
            X_accom = engineered_data[['Destination', 'Duration', 'AccommodationType',
                                     'TravelerNationality', 'Month', 'IsWeekend', 'IsPeakSeason']]
            y_accom = engineered_data['Cost']
            accom_model, accom_params = train_fast_model(X_accom, y_accom, 'accom')
            
            # Train transport model
            transport_data = data[['Destination', 'Duration', 'TransportType',
                                 'TravelerNationality', 'StartDate']].copy()
            transport_data['PeakSeason'] = transport_data['StartDate'].dt.month.isin([6,7,8,12]).astype(int)
            transport_data = engineer_features(transport_data)
            X_trans = transport_data[['Destination', 'Duration', 'TransportType',
                                    'TravelerNationality', 'PeakSeason']]
            y_trans = transport_data['TransportCost']
            trans_model, trans_params = train_fast_model(X_trans, y_trans, 'trans')
            
            st.success("Models trained successfully!")
            
            # Quick evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_accom, y_accom, test_size=0.2, random_state=42)
            y_pred = accom_model.predict(X_test)
            
            st.subheader("Quick Evaluation")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accommodation MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
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
        
        with col2:
            nationality = st.selectbox("Nationality", NATIONALITIES)
            start_date = st.date_input("Start Date", datetime.today())
            transport_type = st.selectbox("Transportation Type", TRANSPORT_TYPES)
        
        submitted = st.form_submit_button("Calculate Costs")

    if submitted:
        try:
            # Load models
            accom_model = joblib.load('travel_cost_model.pkl')
            trans_model = joblib.load('travel_cost_model.pkl')
            
            # Prepare input
            month = start_date.month
            is_weekend = start_date.weekday() >= 5
            is_peak_season = month in [6,7,8,12]
            
            # Accommodation prediction
            accom_input = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'AccommodationType': accommodation,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': int(is_weekend),
                'IsPeakSeason': int(is_peak_season)
            }])
            accom_pred = accom_model.predict(accom_input)[0]
            
            # Transport prediction
            trans_input = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'TransportType': transport_type,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': int(is_weekend),
                'IsPeakSeason': int(is_peak_season)
            }])
            trans_pred = trans_model.predict(trans_input)[0]
            
            total_cost = accom_pred + trans_pred
            
            # Display results
            st.success(f"## Total Estimated Cost: ${total_cost:,.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accommodation", f"${accom_pred:,.2f}")
                st.write(f"Type: {accommodation}")
                st.write(f"Duration: {duration} days")
            
            with col2:
                st.metric("Transportation", f"${trans_pred:,.2f}")
                st.write(f"Type: {transport_type}")
                st.write(f"Season: {'Peak' if is_peak_season else 'Off-peak'}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please train the models first by clicking the 'Train Models' button")

else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists in the same directory.")
