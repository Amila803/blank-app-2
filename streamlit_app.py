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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

# Set page config
st.set_page_config(page_title="Enhanced Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Enhanced Travel Cost Predictor")
st.markdown("""
This app predicts travel costs with improved date and duration relationships.
""")

# Load or generate sample data
@st.cache_data
def load_data():
    # This is sample data - replace with your actual data
    # Generating synthetic data with clear relationships
    np.random.seed(42)
    n_samples = 1000
    
    # Base cost relationships
    destinations = ['London', 'Paris', 'Tokyo', 'New York', 'Bali']
    base_costs = {'London': 150, 'Paris': 130, 'Tokyo': 200, 'New York': 180, 'Bali': 120}
    
    # Create DataFrame
    data = pd.DataFrame({
        'Destination': np.random.choice(destinations, n_samples),
        'Duration': np.random.randint(1, 30, n_samples),
        'StartDate': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2023-12-31'), n_samples)),
        'AccommodationType': np.random.choice(['Hotel', 'Airbnb', 'Resort'], n_samples),
        'TravelerNationality': np.random.choice(['American', 'British', 'Canadian'], n_samples)
    })
    
    # Calculate cost with clear relationships
   # Calculate cost with clear relationships
    data['Cost'] = (
        data['Destination'].map(base_costs) * data['Duration'] *  # Base cost * duration
        (1 + 0.2 * data['StartDate'].dt.month.isin([6,7,8,12])) *  # Peak season markup
        (1 + 0.1 * data['StartDate'].dt.dayofweek.isin([4,5])))  # Weekend markup
    
    # Add some noise
    data['Cost'] = data['Cost'] * np.random.normal(1, 0.1, n_samples)
    return data.round(2)

data = load_data()

# Feature Engineering
def engineer_features(df):
    df = df.copy()
    # Extract date features
    df['Year'] = df['StartDate'].dt.year
    df['Month'] = df['StartDate'].dt.month
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek  # Monday=0, Sunday=6
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
    return df

engineered_data = engineer_features(data)

# Show data relationships
st.header("Data Relationships")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Cost vs Duration")
    fig, ax = plt.subplots()
    sns.regplot(data=engineered_data, x='Duration', y='Cost', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Average Cost by Month")
    monthly_avg = engineered_data.groupby('Month')['Cost'].mean()
    fig, ax = plt.subplots()
    monthly_avg.plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Model Training
st.header("Model Training")

# Prepare features and target
features = ['Destination', 'Duration', 'AccommodationType', 'TravelerNationality', 
            'Month', 'IsWeekend', 'IsPeakSeason']
target = 'Cost'

X = engineered_data[features]
y = engineered_data[target]

# Preprocessing
numeric_features = ['Duration']
categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train model
if st.button("Train Model"):
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Save model
        joblib.dump(best_model, 'travel_cost_model.pkl')
        st.success("Model trained and saved!")

        # Evaluation
        st.subheader("Model Evaluation")
        y_pred = best_model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_xlabel('Actual Cost')
            ax.set_ylabel('Predicted Cost')
            st.pyplot(fig)

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
        start_date = st.date_input("Start Date", datetime.today())
        month = start_date.month
        day_of_week = start_date.weekday()  # Monday=0, Sunday=6
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_season = 1 if month in [6,7,8,12] else 0
    
    submitted = st.form_submit_button("Predict Cost")

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
        
        # Show cost breakdown
        st.subheader("Cost Breakdown")
        base_cost = prediction / duration
        st.write(f"Base daily cost: ${base_cost:,.2f}")
        st.write(f"Total for {duration} days: ${base_cost * duration:,.2f}")
        
        if is_peak_season:
            st.write("⚠️ Peak season surcharge applied")
        if is_weekend:
            st.write("⚠️ Weekend surcharge applied")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
