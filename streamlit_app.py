import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This optimized app quickly predicts travel costs based on actual travel data.
""")

# Load data function with optimizations
@st.cache_data
def load_data():
    try:
        # Load with optimized parameters
        data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig', engine='python')
        
        # Remove empty rows and duplicates
        data = data.dropna(how='all').drop_duplicates()
        
        # Vectorized currency cleaning
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = (
                data[cost_col]
                .astype(str)
                .str.replace('USD', '')
                .str.replace(',', '')
                .str.strip()
                .replace('', np.nan)
                .astype(float)
            )
        
        # Vectorized destination cleaning
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        
        # Faster date parsing
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        # Optimized transport type mapping
        transport_map = {'Plane': 'Flight', 'Airplane': 'Flight', 
                        'Car': 'Car rental', 'Subway': 'Train'}
        data['Transportation type'] = data['Transportation type'].map(transport_map).fillna(data['Transportation type'])
        
        # Column renaming
        data = data.rename(columns={
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation type': 'TransportType',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate'
        })
        
        # Filter and drop NA
        cols = ['Destination', 'Duration', 'StartDate', 'AccommodationType',
                'TravelerNationality', 'Cost', 'TransportType', 'TransportCost']
        return data[cols].dropna(subset=['Cost', 'TransportCost'])
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load data
data = load_data()

if data is not None:
    # Cache dropdown options
    @st.cache_data
    def get_dropdown_options(data):
        return {
            'DESTINATIONS': sorted(data['Destination'].unique().tolist()),
            'TRANSPORT_TYPES': sorted(data['TransportType'].dropna().unique().tolist()),
            'NATIONALITIES': sorted(data['TravelerNationality'].dropna().unique().tolist()),
            'ACCOMMODATION_TYPES': sorted(data['AccommodationType'].dropna().unique().tolist())
        }
    
    options = get_dropdown_options(data)
    
    # Feature Engineering with vectorized operations
    @st.cache_data
    def engineer_features(df):
        df = df.copy()
        df['Year'] = df['StartDate'].dt.year
        df['Month'] = df['StartDate'].dt.month
        df['DayOfWeek'] = df['StartDate'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
        return df

    engineered_data = engineer_features(data)

    # Show data relationships
    st.header("Data Relationships")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost vs Duration")
        fig, ax = plt.subplots()
        sns.regplot(data=engineered_data, x='Duration', y='Cost', ax=ax, scatter_kws={'alpha':0.3})
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Average Cost by Month")
        fig, ax = plt.subplots()
        engineered_data.groupby('Month')['Cost'].mean().plot(kind='bar', ax=ax)
        st.pyplot(fig, use_container_width=True)

    # --- TRANSPORTATION COST PREDICTION ---
    st.header("🚆 Transportation Cost Prediction")

    # Train optimized transport model
    @st.cache_resource
    def train_transport_model():
        transport_data = data[['Destination', 'TransportType', 'TravelerNationality', 'TransportCost']].copy()
        transport_data['PeakSeason'] = pd.to_datetime(data['StartDate']).dt.month.isin([6,7,8,12]).astype(int)
        
        X = transport_data[['Destination', 'TransportType', 'TravelerNationality', 'PeakSeason']]
        y = transport_data['TransportCost']
        
        # Simplified preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 ['Destination', 'TransportType', 'TravelerNationality'])
            ])
        
        # Optimized LGBM model with sensible defaults
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])
        
        model.fit(X, y)
        return model

    transport_model = train_transport_model()

    # Show transportation relationships
    st.subheader("Cost Patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Average Cost by Transport Type**")
        fig, ax = plt.subplots()
        sns.barplot(data=data, x='TransportType', y='TransportCost', ax=ax, estimator='mean')
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.write("**Nationality Preferences**")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='TravelerNationality', hue='TransportType', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)

    # --- ACCOMMODATION COST PREDICTION ---
    st.header("Cost Prediction")

    # Prepare 
