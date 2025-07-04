import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from datetime import datetime

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs based on actual travel data with improved modeling techniques.
""")

# Load data function with enhanced cleaning
@st.cache_data
def load_data():
    try:
        # Load the dataset with proper encoding
        data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig')
        
        # Remove completely empty rows
        data = data.dropna(how='all')
        
        # Function to clean currency values
        def clean_currency(value):
            if isinstance(value, str):
                # Remove USD, commas, and whitespace
                value = value.replace('USD', '').replace(',', '').replace('$', '').strip()
                try:
                    return float(value)
                except:
                    return None
            return value
        
        # Clean cost columns
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
        
        # Clean destination names (remove countries)
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        
        # Convert dates - handle multiple date formats
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce', format='mixed')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce', format='mixed')
        
        # Calculate duration
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        # Standardize transport types
        transport_mapping = {
            'Plane': 'Flight',
            'Airplane': 'Flight',
            'Car': 'Car rental',
            'Subway': 'Train',
            'Bus': 'Bus',
            'Train': 'Train',
            'Ferry': 'Ferry'
        }
        data['Transportation type'] = data['Transportation type'].replace(transport_mapping)
        
        # Rename columns
        data = data.rename(columns={
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation type': 'TransportType',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate'
        })
        
        # Filter and clean data
        data = data[[
            'Destination', 'Duration', 'StartDate', 'AccommodationType',
            'TravelerNationality', 'Cost', 'TransportType', 'TransportCost'
        ]].dropna(subset=['Cost', 'TransportCost'])
        
        # Remove extreme outliers
        cost_q1 = data['Cost'].quantile(0.05)
        cost_q3 = data['Cost'].quantile(0.95)
        transport_q1 = data['TransportCost'].quantile(0.05)
        transport_q3 = data['TransportCost'].quantile(0.95)
        
        data = data[
            (data['Cost'] >= cost_q1) & (data['Cost'] <= cost_q3) &
            (data['TransportCost'] >= transport_q1) & (data['TransportCost'] <= transport_q3)
        ]
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Feature engineering function
def engineer_features(df):
    df = df.copy()
    # Extract date features
    df['Year'] = df['StartDate'].dt.year
    df['Month'] = df['StartDate'].dt.month
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek  # Monday=0, Sunday=6
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
    
    # Additional features
    df['DurationSquared'] = df['Duration'] ** 2
    df['WeekendDuration'] = df['IsWeekend'] * df['Duration']
    df['PeakDuration'] = df['IsPeakSeason'] * df['Duration']
    
    return df

# Load data
data = load_data()

if data is not None:
    # Feature engineering
    engineered_data = engineer_features(data)
    
    # Update dropdown options based on actual data
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())
    
    # Show data relationships
    st.header("Data Relationships")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accommodation Cost vs Duration")
        fig, ax = plt.subplots()
        sns.regplot(data=engineered_data, x='Duration', y='Cost', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Transport Cost by Type")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='TransportType', y='TransportCost', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Create shared preprocessor
    def create_preprocessor():
        numeric_features = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason']
        categorical_features = ['Destination', 'TravelerNationality']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        return ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ])

    # Model training function
    def train_models():
        # Prepare accommodation data
        X_accom = engineered_data[['Destination', 'Duration', 'TravelerNationality',
                                 'Month', 'IsWeekend', 'IsPeakSeason']]
        y_accom = np.log1p(engineered_data['Cost'])  # Log transform
        
        # Prepare transport data
        transport_data = data[['Destination', 'Duration', 'TravelerNationality',
                             'TransportType', 'TransportCost', 'StartDate']].copy()
        transport_data['PeakSeason'] = transport_data['StartDate'].dt.month.isin([6,7,8,12]).astype(int)
        X_trans = transport_data[['Destination', 'Duration', 'TravelerNationality', 'PeakSeason']]
        y_trans = np.log1p(transport_data['TransportCost'])  # Log transform
        
        # Create models
        preprocessor = create_preprocessor()
        
        # Accommodation model (Gradient Boosting)
        accom_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Transport model (XGBoost)
        trans_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        
        # Train models
        accom_model.fit(X_accom, y_accom)
        trans_model.fit(X_trans, y_trans)
        
        return accom_model, trans_model

    # Train models button
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            accom_model, trans_model = train_models()
            
            # Save models
            joblib.dump(accom_model, 'accom_model.pkl')
            joblib.dump(trans_model, 'trans_model.pkl')
            st.success("Models trained and saved!")
            
            # Evaluate accommodation model
            X_train, X_test, y_train, y_test = train_test_split(
                X_accom, y_accom, test_size=0.2, random_state=42)
            y_pred = accom_model.predict(X_test)
            
            # Reverse log transform
            y_test = np.expm1(y_test)
            y_pred = np.expm1(y_pred)
            
            # Evaluation metrics
            st.subheader("Model Evaluation")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accommodation MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
                st.metric("Accommodation R²", f"{r2_score(y_test, y_pred):.2f}")
                
                # Baseline comparison
                baseline_mae = mean_absolute_error(y_test, [np.expm1(y_train).mean()]*len(y_test))
                st.metric("Improvement Over Mean", 
                         f"{100*(baseline_mae - mean_absolute_error(y_test, y_pred))/baseline_mae:.1f}%")
            
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
            day_of_week = start_date.weekday()  # Monday=0, Sunday=6
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_season = 1 if month in [6,7,8,12] else 0
        
        submitted = st.form_submit_button("Calculate Costs")

    if submitted:
        try:
            # Load models
            accom_model = joblib.load('accom_model.pkl')
            trans_model = joblib.load('trans_model.pkl')
            
            # Prepare input data
            input_data = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': is_weekend,
                'IsPeakSeason': is_peak_season
            }])
            
            # Make predictions
            accom_pred = np.expm1(accom_model.predict(input_data))[0]
            trans_pred = np.expm1(trans_model.predict(input_data))[0]
            total_cost = accom_pred + trans_pred
            
            # Display results
            st.success(f"## Total Estimated Trip Cost: ${total_cost:,.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accommodation Cost", f"${accom_pred:,.2f}")
                st.write(f"Duration: {duration} days")
                st.write(f"Type: {accommodation}")
            
            with col2:
                st.metric("Transportation Cost", f"${trans_pred:,.2f}")
                st.write(f"Season: {'Peak' if is_peak_season else 'Off-peak'}")
                st.write(f"Traveler: {nationality}")
            
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
