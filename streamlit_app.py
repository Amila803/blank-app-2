import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")


# Custom constrained model class
class ConstrainedRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )
        self.accommodation_weights = {
            'Hostel': 0.8,  # 20% cheaper
            'Airbnb': 0.95,
            'Hotel': 1.0,   # Baseline
            'Resort': 1.3,  # 30% premium
            'Villa': 1.2,
            'Guesthouse': 0.85,
            'Vacation rental': 0.9
        }

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # Always predicts BASE DAILY RATE
        base_pred = self.model.predict(X)
        
        if isinstance(X, pd.DataFrame) and 'AccommodationType' in X.columns:
            acc_types = X['AccommodationType']
            acc_multipliers = acc_types.map(lambda x: self.accommodation_weights.get(x, 1.0)).values
            base_pred = base_pred * acc_multipliers
        
        return base_pred

# Data loading
@st.cache_resource
def load_data():
    try:
        data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig')
        data = data.dropna(how='all')
        
        def clean_currency(value):
            if isinstance(value, str):
                value = value.replace('USD', '').replace(',', '').replace('$', '').strip()
                return float(value) if value.replace('.','',1).isdigit() else None
            return value
        
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
        
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        transport_mapping = {
            'Plane': 'Flight', 'Airplane': 'Flight', 'Car': 'Car rental',
            'Subway': 'Train', 'Bus': 'Bus', 'Ferry': 'Ferry'
        }
        data['TransportType'] = data['Transportation type'].replace(transport_mapping)
        
        data = data.rename(columns={
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate'
        }).dropna(subset=['Cost', 'TransportCost'])
        
        return data[['Destination', 'Duration', 'StartDate', 'AccommodationType',
                   'TravelerNationality', 'Cost', 'TransportType', 'TransportCost']]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main app
data = load_data()

if data is not None:
    # Prepare data
    def engineer_features(df):
        df = df.copy()
        df['Month'] = df['StartDate'].dt.month
        df['IsWeekend'] = df['StartDate'].dt.dayofweek.isin([5,6]).astype(int)
        df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
        return df

    engineered_data = engineer_features(data)
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())

    # Model training
    st.header("Model Training")
    if st.button("Train Models"):
        with st.spinner("Training..."):
            # Accommodation model
            X_acc = engineered_data[['Destination', 'AccommodationType', 
                                   'TravelerNationality', 'Month', 'IsWeekend', 
                                   'IsPeakSeason']]
            y_acc = engineered_data['Cost'] / engineered_data['Duration']  # Daily rate
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), ['Month', 'IsWeekend', 'IsPeakSeason']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['Destination', 'AccommodationType', 'TravelerNationality'])
            ])
            
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', ConstrainedRandomForest(random_state=42))
            ])
            
            model.fit(X_acc, y_acc)
            joblib.dump(model, 'accommodation_model.pkl')
            
            # Transport model
            X_trans = engineered_data[['Destination', 'TransportType', 
                                     'TravelerNationality', 'IsPeakSeason']]
            y_trans = engineered_data['TransportCost']
            
            trans_model = Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ('regressor', LGBMRegressor())
            ])
            trans_model.fit(X_trans, y_trans)
            joblib.dump(trans_model, 'transport_model.pkl')
            
            st.success("Models trained successfully!")

    # Prediction UI
    st.header("Cost Prediction")
    
    with st.form("accommodation_form"):
        st.subheader("Accommodation Cost")
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", DESTINATIONS)
            accommodation = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES)
        with col2:
            nationality = st.selectbox("Nationality", NATIONALITIES)
            duration = st.number_input("Nights", min_value=1, max_value=90, value=7)
        
        start_date = st.date_input("Start Date", datetime.today())
        month = start_date.month
        is_weekend = start_date.weekday() >= 5
        is_peak = month in [6,7,8,12]
        
        if st.form_submit_button("Calculate"):
            try:
                model = joblib.load('accommodation_model.pkl')
                input_data = pd.DataFrame([{
                    'Destination': destination,
                    'AccommodationType': accommodation,
                    'TravelerNationality': nationality,
                    'Month': month,
                    'IsWeekend': int(is_weekend),
                    'IsPeakSeason': int(is_peak)
                }])
                
                daily_rate = model.predict(input_data)[0]
                total_cost = daily_rate * duration
                
                st.success(f"## Total Accommodation Cost: ${total_cost:,.2f}")
                st.write(f"**Daily rate:** ${daily_rate:,.2f}")
                st.write(f"**{duration} nights × ${daily_rate:,.2f} = ${total_cost:,.2f}**")
                st.session_state['accom_cost'] = total_cost
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Train model first.")

    with st.form("transport_form"):
        st.subheader("Transportation Cost")
        col1, col2 = st.columns(2)
        with col1:
            trans_dest = st.selectbox("Destination", DESTINATIONS, key='trans_dest')
            trans_type = st.selectbox("Type", TRANSPORT_TYPES)
        with col2:
            trans_nat = st.selectbox("Nationality", NATIONALITIES, key='trans_nat')
            is_peak = st.checkbox("Peak Season", value=False)
        
        if st.form_submit_button("Calculate"):
            try:
                model = joblib.load('transport_model.pkl')
                input_data = pd.DataFrame([{
                    'Destination': trans_dest,
                    'TransportType': trans_type,
                    'TravelerNationality': trans_nat,
                    'IsPeakSeason': int(is_peak)
                }])
                
                trans_cost = model.predict(input_data)[0]
                st.success(f"## Transportation Cost: ${trans_cost:,.2f}")
                st.session_state['trans_cost'] = trans_cost
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Train model first.")

    # Combined total
    if 'accom_cost' in st.session_state and 'trans_cost' in st.session_state:
        st.header("💵 Total Trip Cost")
        total = st.session_state['accom_cost'] + st.session_state['trans_cost']
        st.success(f"## Grand Total: ${total:,.2f}")
        st.write(f"- Accommodation: ${st.session_state['accom_cost']:,.2f}")
        st.write(f"- Transportation: ${st.session_state['trans_cost']:,.2f}")

else:
    st.error("Failed to load data. Check your CSV file.")
