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
st.set_page_config(page_title="Fast Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Fast Travel Cost Predictor")
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

    # Prepare features and target
    features = ['Destination', 'Duration', 'AccommodationType', 'TravelerNationality', 
                'Month', 'IsWeekend', 'IsPeakSeason']
    target = 'Cost'

    X = engineered_data[features]
    y = engineered_data[target]

    # Simplified preprocessing
    categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
    numeric_features = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # Faster model with feature selection
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(LGBMRegressor(n_estimators=100, random_state=42))),
        ('regressor', XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster training method
        ))
    ])

    # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Faster training with early stopping
            model.fit(X_train, y_train,
                     regressor__early_stopping_rounds=10,
                     regressor__eval_set=[(X_test, y_test)],
                     regressor__verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
                st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            
            with col2:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, ax=ax)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                st.pyplot(fig, use_container_width=True)

            # Save model
            joblib.dump(model, 'travel_cost_model.pkl')
            st.success("Model trained and saved!")

    # Prediction Interface
    st.header("Cost Prediction")

    with st.form("prediction_form"):
        st.subheader("Calculate Accommodation Costs")
        
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", options['DESTINATIONS'])
            duration = st.number_input("Duration (days)", min_value=1, max_value=90, value=7)
            accommodation = st.selectbox("Accommodation Type", options['ACCOMMODATION_TYPES'])
            nationality = st.selectbox("Nationality", options['NATIONALITIES'])
        
        with col2:
            start_date = st.date_input("Start Date", datetime.today())
            month = start_date.month
            is_weekend = 1 if start_date.weekday() >= 5 else 0
            is_peak_season = 1 if month in [6,7,8,12] else 0
        
        submitted = st.form_submit_button("Calculate Accommodation Cost")

    if submitted:
        try:
            model = joblib.load('travel_cost_model.pkl')
            
            input_data = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'AccommodationType': accommodation,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': is_weekend,
                'IsPeakSeason': is_peak_season
            }])
            
            prediction = model.predict(input_data)[0]
            st.success(f"## Predicted Cost: ${prediction:,.2f}")
            st.session_state['accom_pred'] = prediction
            
            # Show cost breakdown
            st.write(f"**Daily rate:** ${prediction/duration:,.2f}")
            if is_peak_season:
                st.write("⚠️ Peak season pricing")
            if is_weekend:
                st.write("⚠️ Weekend pricing")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # Transport Prediction interface
    with st.form("transport_form"):
        st.subheader("Calculate Transportation Costs")
        
        col1, col2 = st.columns(2)
        with col1:
            trans_destination = st.selectbox("Destination", options['DESTINATIONS'], key='trans_dest')
            trans_type = st.selectbox("Transportation Type", options['TRANSPORT_TYPES'], key='trans_type')
        with col2:
            trans_nationality = st.selectbox("Nationality", options['NATIONALITIES'], key='trans_nat')
            is_peak = st.checkbox("Peak Season Travel", value=False)
        
        submitted = st.form_submit_button("Calculate Transport Cost")

    if submitted:
        input_data = pd.DataFrame([{
            'Destination': trans_destination,
            'TransportType': trans_type,
            'TravelerNationality': trans_nationality,
            'PeakSeason': int(is_peak)
        }])
        
        pred_cost = transport_model.predict(input_data)[0]
        st.success(f"### Estimated Transportation Cost: ${pred_cost:.2f}")
        st.session_state['trans_pred'] = pred_cost

    # Combined Cost Prediction
    st.header("💵 Combined Cost Prediction")

    if 'accom_pred' in st.session_state and 'trans_pred' in st.session_state:
        total_cost = st.session_state['accom_pred'] + st.session_state['trans_pred']
        st.success(f"## Total Estimated Trip Cost: ${total_cost:,.2f}")
        st.write(f"- Accommodation: ${st.session_state['accom_pred']:,.2f}")
        st.write(f"- Transportation: ${st.session_state['trans_pred']:,.2f}")
else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists.")
