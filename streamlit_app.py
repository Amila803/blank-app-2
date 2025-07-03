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
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

# Set page config
st.set_page_config(page_title="Enhanced Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Complete Travel Cost Predictor")
st.markdown("Predicting both accommodation and transportation costs with destination-specific relationships")

# Load or generate sample data
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    
    destinations = ['London', 'Paris', 'Tokyo', 'New York', 'Bali']
    transport_types = ['Flight', 'Train', 'Bus', 'Car']
    
    # Base costs with destination-specific transportation patterns
    base_accom_costs = {'London': 150, 'Paris': 130, 'Tokyo': 200, 'New York': 180, 'Bali': 120}
    transport_cost_factors = {
        'London': {'Flight': 1.2, 'Train': 0.8, 'Bus': 0.5, 'Car': 0.7},
        'Paris': {'Flight': 1.1, 'Train': 0.9, 'Bus': 0.6, 'Car': 0.8},
        'Tokyo': {'Flight': 1.4, 'Train': 1.1, 'Bus': 0.7, 'Car': 1.0},
        'New York': {'Flight': 1.3, 'Train': 0.7, 'Bus': 0.4, 'Car': 0.9},
        'Bali': {'Flight': 1.5, 'Train': 0.2, 'Bus': 0.3, 'Car': 0.6}
    }
    
    # Nationality preferences for transportation
    nationality_prefs = {
        'American': {'Flight': 0.9, 'Car': 1.2},
        'British': {'Train': 1.1, 'Flight': 1.0},
        'Canadian': {'Flight': 1.0, 'Car': 1.1},
        'Japanese': {'Train': 1.3, 'Bus': 1.1},
        'Australian': {'Flight': 1.2, 'Car': 0.9}
    }
    
    data = pd.DataFrame({
        'Destination': np.random.choice(destinations, n_samples),
        'Duration': np.random.randint(1, 30, n_samples),
        'StartDate': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2023-12-31'), n_samples)),
        'AccommodationType': np.random.choice(['Hotel', 'Airbnb', 'Resort'], n_samples),
        'TransportationType': np.random.choice(transport_types, n_samples),
        'TravelerNationality': np.random.choice(list(nationality_prefs.keys()), n_samples)
    })
    
    # Calculate accommodation cost (unchanged from your working version)
    data['AccommodationCost'] = (
        data['Destination'].map(base_accom_costs) * data['Duration'] *
        (1 + 0.2 * data['StartDate'].dt.month.isin([6,7,8,12])) *
        (1 + 0.1 * data['StartDate'].dt.dayofweek.isin([4,5]))) * np.random.normal(1, 0.1, n_samples)
    
    # Calculate transportation cost with destination and nationality relationships
    base_transport_cost = 100  # Base cost for transportation
    data['TransportationCost'] = (
        base_transport_cost *
        data['Destination'].apply(lambda x: transport_cost_factors[x][data.loc[data['Destination']==x, 'TransportationType']]) *
        data['TravelerNationality'].apply(lambda x: nationality_prefs[x].get(data.loc[data['TravelerNationality']==x, 'TransportationType'], 1.0)) *
        (1 + 0.15 * data['StartDate'].dt.month.isin([6,7,8,12]))  # Peak season transport markup
    )
    
    data['TotalCost'] = data['AccommodationCost'] + data['TransportationCost']
    return data.round(2)

data = load_data()

# Feature Engineering
def engineer_features(df):
    df = df.copy()
    df['Year'] = df['StartDate'].dt.year
    df['Month'] = df['StartDate'].dt.month
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
    return df

engineered_data = engineer_features(data)

# Show transportation relationships
st.header("Transportation Cost Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Cost by Transportation Type")
    fig, ax = plt.subplots()
    sns.boxplot(data=engineered_data, x='TransportationType', y='TransportationCost', hue='Destination', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("Nationality Preferences")
    fig, ax = plt.subplots()
    sns.countplot(data=engineered_data, x='TravelerNationality', hue='TransportationType', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Model Training
st.header("Model Training")

# Accommodation Model (unchanged from your working version)
st.subheader("Accommodation Cost Model")
accom_features = ['Destination', 'Duration', 'AccommodationType', 'Month', 'IsWeekend']
accom_target = 'AccommodationCost'

X_accom = engineered_data[accom_features]
y_accom = engineered_data[accom_target]

# Transportation Model
st.subheader("Transportation Cost Model")
transport_features = ['Destination', 'TransportationType', 'TravelerNationality', 'Month']
transport_target = 'TransportationCost'

X_trans = engineered_data[transport_features]
y_trans = engineered_data[transport_target]

# Preprocessing
numeric_features = ['Duration']
categorical_features = ['Destination', 'AccommodationType', 'TransportationType', 'TravelerNationality']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train models
if st.button("Train Models"):
    with st.spinner("Training models..."):
        # Accommodation model
        X_train_accom, X_test_accom, y_train_accom, y_test_accom = train_test_split(
            X_accom, y_accom, test_size=0.2, random_state=42)
        
        accom_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        accom_model.fit(X_train_accom, y_train_accom)
        
        # Transportation model
        X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(
            X_trans, y_trans, test_size=0.2, random_state=42)
        
        trans_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        trans_model.fit(X_train_trans, y_train_trans)
        
        # Save models
        joblib.dump(accom_model, 'accom_model.pkl')
        joblib.dump(trans_model, 'trans_model.pkl')
        st.success("Models trained and saved!")

        # Evaluation
        st.subheader("Model Performance")
        
        # Accommodation evaluation
        y_pred_accom = accom_model.predict(X_test_accom)
        accom_mae = mean_absolute_error(y_test_accom, y_pred_accom)
        accom_r2 = r2_score(y_test_accom, y_pred_accom)
        
        # Transportation evaluation
        y_pred_trans = trans_model.predict(X_test_trans)
        trans_mae = mean_absolute_error(y_test_trans, y_pred_trans)
        trans_r2 = r2_score(y_test_trans, y_pred_trans)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accommodation MAE", f"${accom_mae:.2f}")
            st.metric("Accommodation R²", f"{accom_r2:.2f}")
        with col2:
            st.metric("Transportation MAE", f"${trans_mae:.2f}")
            st.metric("Transportation R²", f"{trans_r2:.2f}")

# Prediction Interface
st.header("Cost Prediction")

with st.form("prediction_form"):
    st.subheader("Enter Trip Details")
    
    col1, col2 = st.columns(2)
    with col1:
        destination = st.selectbox("Destination", data['Destination'].unique())
        duration = st.number_input("Duration (days)", min_value=1, max_value=90, value=7)
        accommodation = st.selectbox("Accommodation Type", data['AccommodationType'].unique())
        nationality = st.selectbox("Nationality", data['TravelerNationality'].unique())
    
    with col2:
        transport_type = st.selectbox("Transportation Type", data['TransportationType'].unique())
        start_date = st.date_input("Start Date", datetime.today())
        month = start_date.month
        is_weekend = 1 if start_date.weekday() >= 5 else 0
        is_peak_season = 1 if month in [6,7,8,12] else 0
    
    submitted = st.form_submit_button("Predict Costs")

if submitted:
    try:
        # Load models
        accom_model = joblib.load('accom_model.pkl')
        trans_model = joblib.load('trans_model.pkl')
        
        # Prepare input data
        accom_input = pd.DataFrame([{
            'Destination': destination,
            'Duration': duration,
            'AccommodationType': accommodation,
            'Month': month,
            'IsWeekend': is_weekend
        }])
        
        trans_input = pd.DataFrame([{
            'Destination': destination,
            'TransportationType': transport_type,
            'TravelerNationality': nationality,
            'Month': month
        }])
        
        # Make predictions
        accom_pred = accom_model.predict(accom_input)[0]
        trans_pred = trans_model.predict(trans_input)[0]
        total_pred = accom_pred + trans_pred
        
        # Display results
        st.success("## Cost Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accommodation Cost", f"${accom_pred:,.2f}")
        with col2:
            st.metric("Transportation Cost", f"${trans_pred:,.2f}")
        with col3:
            st.metric("Total Estimated Cost", f"${total_pred:,.2f}")
        
        # Show insights
        st.subheader("Cost Insights")
        
        # Transportation cost factors
        st.write(f"**Transportation Analysis for {nationality} travelers to {destination}:**")
        if transport_type == 'Flight':
            st.write("- ✈️ Flight costs are typically higher but faster")
            if destination in ['Tokyo', 'Bali']:
                st.write("- 🌏 Long-haul flights have additional distance charges")
        elif transport_type == 'Train':
            st.write("- 🚆 Train travel offers good value in Europe")
            if destination == 'Japan':
                st.write("- 🚄 Shinkansen (bullet train) provides premium service")
        
        # Peak season notice
        if is_peak_season:
            st.warning("📅 Peak season rates apply (June-August, December)")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Please train the models first using the 'Train Models' button above")
