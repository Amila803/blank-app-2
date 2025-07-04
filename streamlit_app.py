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
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs with improved date and duration relationships.
""")

# Load or generate sample data
@st.cache_data
# Replace your load_data() function with this:
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
                value = value.replace('USD', '').replace(',', '').strip()
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
        data['Duration (days)'] = (data['End date'] - data['Start date']).dt.days
        
        # Standardize transport types
        transport_mapping = {
            'Plane': 'Flight',
            'Airplane': 'Flight',
            'Car': 'Car rental',
            'Subway': 'Train'
        }
        data['Transportation type'] = data['Transportation type'].replace(transport_mapping)
        
        # Rename columns to match your existing code
        data = data.rename(columns={
            'Duration (days)': 'Duration',
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation type': 'TransportType',
            'Transportation cost': 'TransportCost'
        })
        
        # Filter only needed columns and drop rows with missing critical data
        data = data[[
            'Destination', 'Duration', 'Start date', 'AccommodationType',
            'TravelerNationality', 'Cost', 'TransportType', 'TransportCost'
        ]].dropna(subset=['Cost', 'TransportCost'])
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Fallback to sample data
        return load_sample_data()

def load_sample_data():
    # Your original synthetic data generation code here
    np.random.seed(42)
    n_samples = 1000
    destinations = ['London', 'Paris', 'Tokyo', 'New York', 'Bali']
    base_costs = {'London': 150, 'Paris': 130, 'Tokyo': 200, 'New York': 180, 'Bali': 120}
    
    data = pd.DataFrame({
        'Destination': np.random.choice(destinations, n_samples),
        'Duration': np.random.randint(1, 30, n_samples),
        'Start date': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2023-12-31'), n_samples)),
        'AccommodationType': np.random.choice(['Hotel', 'Airbnb', 'Resort', 'Hostel'], n_samples),
        'TravelerNationality': np.random.choice(['American', 'British', 'Canadian'], n_samples),
        'Cost': 0,  # Will be calculated
        'TransportType': np.random.choice(['Flight', 'Train', 'Bus', 'Car rental'], n_samples),
        'TransportCost': 0  # Will be calculated
    })
    
    accommodation_factors = {'Hostel': 0.2, 'Hotel': 1.0, 'Airbnb': 0.8, 'Resort': 1.5}
    data['Cost'] = (
        data['Destination'].map(base_costs) * 
        data['Duration'] * 
        (1 + 0.2 * data['Start date'].dt.month.isin([6,7,8,12])) * 
        (1 + 0.1 * data['Start date'].dt.dayofweek.isin([4,5])) * 
        data['AccommodationType'].map(accommodation_factors) * 
        np.random.normal(1, 0.1, n_samples)
    
    # Simple transport cost calculation for sample data
    transport_base = {'Flight': 300, 'Train': 100, 'Bus': 50, 'Car rental': 150}
    data['TransportCost'] = data['TransportType'].map(transport_base) * np.random.normal(1, 0.2, n_samples)
    
    return data.round(2)

# Load data
data = load_data()

# Update your dropdown options based on actual data
if data is not None:
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())
data = load_data()


### FEATURE ENGINEERING

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

# --- TRANSPORTATION COST PREDICTION ---
st.header("🚆 Transportation Cost Prediction")


# Transportation type options
TRANSPORT_TYPES = ['Flight', 'Train', 'Bus', 'Car rental', 'Ferry', 'Bike rental', 'Rideshare']
NATIONALITIES = [
    'American', 'British', 'Canadian', 'Australian', 'Japanese',
    'Chinese', 'German', 'French', 'Italian', 'Spanish',
    'Brazilian', 'Indian', 'Russian', 'South Korean', 'Mexican',
    'Dutch', 'Swedish', 'Norwegian', 'Swiss', 'Singaporean',
    'Thai', 'Vietnamese', 'Indonesian', 'Malaysian', 'Emirati'
]
DESTINATIONS = [
    'London', 'Paris', 'Tokyo', 'New York', 'Bali',
    'Sydney', 'Rome', 'Berlin', 'Barcelona', 'Dubai',
    'Singapore', 'Hong Kong', 'Bangkok', 'Seoul', 'Istanbul',
    'Cape Town', 'Rio de Janeiro', 'Toronto', 'Los Angeles', 'Mumbai',
    'Amsterdam', 'Vienna', 'Prague', 'Athens', 'Cairo',
    'Reykjavik', 'Hawaii', 'Santorini', 'Phuket', 'Marrakech'
]
# Load/generate transportation data
@st.cache_data
def load_transport_data():
    np.random.seed(42)
    n_samples = 500
    
    # Base costs by destination and transport type
    transport_costs = {
        'London': {'Flight': 400, 'Train': 150, 'Bus': 80, 'Car rental': 200},
        'Paris': {'Flight': 350, 'Train': 120, 'Bus': 60, 'Car rental': 180},
        'Tokyo': {'Flight': 800, 'Train': 250, 'Bus': 100, 'Car rental': 300},
        'New York': {'Flight': 500, 'Train': 100, 'Bus': 70, 'Car rental': 250},
        'Bali': {'Flight': 700, 'Train': 50, 'Bus': 30, 'Car rental': 150}
    }
    
    # Nationality preferences (multipliers)
    nationality_factors = {
        'American': {'Flight': 1.0, 'Car rental': 1.2},
        'British': {'Train': 1.3, 'Flight': 1.1},
        'Canadian': {'Flight': 1.1, 'Car rental': 1.1},
        'Japanese': {'Train': 1.4, 'Bus': 1.2},
        'Australian': {'Flight': 1.2, 'Car rental': 0.9}
    }
    
    data = pd.DataFrame({
        'Destination': np.random.choice(DESTINATIONS, n_samples),
        'TransportType': np.random.choice(TRANSPORT_TYPES, n_samples),
        'Nationality': np.random.choice(NATIONALITIES, n_samples),
        'BaseCost': [transport_costs[d][t] for d,t in zip(
            np.random.choice(DESTINATIONS, n_samples),
            np.random.choice(TRANSPORT_TYPES, n_samples)
        )],
        'PeakSeason': np.random.choice([0,1], n_samples, p=[0.7,0.3])
    })
    
    # Apply nationality preferences and peak season markup
    data['TransportCost'] = data['BaseCost'] * \
        [nationality_factors[row['Nationality']].get(row['TransportType'], 1.0) 
         for _, row in data.iterrows()] * \
        (1 + data['PeakSeason']*0.2)
    
    return data

transport_data = load_transport_data()

# Show transportation relationships
st.subheader("Cost Patterns")
col1, col2 = st.columns(2)
with col1:
    st.write("**Average Cost by Transport Type**")
    fig, ax = plt.subplots()
    sns.barplot(data=transport_data, x='TransportType', y='TransportCost', 
                hue='Destination', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.write("**Nationality Preferences**")
    fig, ax = plt.subplots()
    sns.countplot(data=transport_data, x='Nationality', hue='TransportType', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Train transportation model
@st.cache_resource
def train_transport_model():
    # Feature engineering
    X = transport_data[['Destination', 'TransportType', 'Nationality', 'PeakSeason']]
    y = transport_data['TransportCost']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Destination', 'TransportType', 'Nationality'])
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    model.fit(X, y)
    return model

transport_model = train_transport_model()




### MODEL TRAINING
st.header("Model Training")

# Prepare features and target
features = ['Destination', 'Duration', 'AccommodationType', 'TravelerNationality', 
            'Month', 'IsWeekend', 'IsPeakSeason']
target = 'Cost'

X = engineered_data[features]
y = engineered_data[target]

# Preprocessing
categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
numeric_features     = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(),        numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
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

# Transport Prediction interface
with st.form("transport_form"):
    st.subheader("Calculate Transportation Costs")
    
    col1, col2 = st.columns(2)
    with col1:
        trans_destination = st.selectbox("Destination", DESTINATIONS, key='trans_dest')
        trans_type = st.selectbox("Transportation Type", TRANSPORT_TYPES, key='trans_type')
    with col2:
        trans_nationality = st.selectbox("Nationality", NATIONALITIES, key='trans_nat')
        is_peak = st.checkbox("Peak Season Travel", value=False)
    
    submitted = st.form_submit_button("Calculate Transport Cost")

if submitted:
    input_data = pd.DataFrame([{
        'Destination': trans_destination,
        'TransportType': trans_type,
        'Nationality': trans_nationality,
        'PeakSeason': int(is_peak)
    }])
    
    pred_cost = transport_model.predict(input_data)[0]
    
    st.success(f"### Estimated Transportation Cost: ${pred_cost:.2f}")
    st.session_state['trans_pred'] = pred_cost

    
    # Show cost factors
    st.write("**Cost Factors:**")
    if is_peak:
        st.write("- 20% peak season surcharge applied")
    if trans_nationality == 'Japanese' and trans_type == 'Train':
        st.write("- Japanese travelers typically prefer trains (higher quality expectation)")
    if trans_destination == 'Bali' and trans_type == 'Train':
        st.warning("Limited train options in Bali - consider flights or car rental")

# --- INTEGRATION WITH ACCOMMODATION MODEL ---
st.header("💵 Combined Cost Prediction")

if 'accom_pred' in st.session_state and 'trans_pred' in st.session_state:
    total_cost = st.session_state['accom_pred'] + st.session_state['trans_pred']
    st.success(f"## Total Estimated Trip Cost: ${total_cost:.2f}")
    st.write(f"- Accommodation: ${st.session_state['accom_pred']:.2f}")
    st.write(f"- Transportation: ${st.session_state['trans_pred']:.2f}")
