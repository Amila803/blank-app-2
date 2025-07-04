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
        'AccommodationType': np.random.choice(['Hotel', 'Airbnb', 'Resort', 'Hostel'], n_samples),
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
col1, col2, col3 = st.columns(3)

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
    ax.set_ylabel("Average Cost")
    st.pyplot(fig)

with col3:
    st.subheader("Average Cost by Accommodation")
    accom_avg = engineered_data.groupby('AccommodationType')['Cost']\
                 .mean().sort_values()
    fig, ax = plt.subplots()
    accom_avg.plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Cost")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- TRANSPORTATION COST PREDICTION ---
st.header("🚆 Transportation Cost Prediction")


# Transportation type options
TRANSPORT_TYPES = ['Flight', 'Train', 'Bus', 'Car rental']
NATIONALITIES = sorted(list(set([
    'American', 'Canadian', 'Korean', 'British', 'Vietnamese', 'Australian',
    'Brazilian', 'Dutch', 'Emirati', 'Mexican', 'Spanish', 'Chinese',
    'German', 'Moroccan', 'Scottish', 'Indian', 'Italian', 'South Korean',
    'Taiwanese', 'South African', 'French', 'Japanese', 'Cambodia', 'Greece',
    'United Arab Emirates', 'Hong Kong', 'Singapore', 'Indonesia', 'USA',
    'UK', 'China', 'New Zealander'
])))

DESTINATIONS = [
    'London', 'Phuket', 'Bali', 'New York', 'Tokyo', 'Paris', 'Sydney',
    'Rio de Janeiro', 'Amsterdam', 'Dubai', 'Cancun', 'Barcelona',
    'Honolulu', 'Berlin', 'Marrakech', 'Edinburgh', 'Rome', 'Bangkok',
    'Cape Town', 'Vancouver', 'Seoul', 'Los Angeles', 'Santorini',
    'Phnom Penh', 'Athens', 'Auckland'
]


# Load/generate transportation data
@st.cache_data
def load_transport_data():
    np.random.seed(42)
    n_samples = 500
    
    # Base costs by destination and transport type
# Base costs by destination and transport type
transport_costs = {
    'London': {'Flight': 400, 'Train': 150, 'Bus': 80, 'Car rental': 200},
    'Paris': {'Flight': 350, 'Train': 120, 'Bus': 60, 'Car rental': 180},
    'Tokyo': {'Flight': 800, 'Train': 250, 'Bus': 100, 'Car rental': 300},
    'New York': {'Flight': 500, 'Train': 100, 'Bus': 70, 'Car rental': 250},
    'Bali': {'Flight': 700, 'Train': 50, 'Bus': 30, 'Car rental': 150},
    'Phuket': {'Flight': 650, 'Train': 40, 'Bus': 35, 'Car rental': 120},
    'Sydney': {'Flight': 750, 'Train': 180, 'Bus': 90, 'Car rental': 220},
    'Rio de Janeiro': {'Flight': 600, 'Train': 90, 'Bus': 50, 'Car rental': 170},
    'Amsterdam': {'Flight': 380, 'Train': 130, 'Bus': 65, 'Car rental': 190},
    'Dubai': {'Flight': 450, 'Train': 60, 'Bus': 40, 'Car rental': 160},
    'Cancun': {'Flight': 550, 'Train': 70, 'Bus': 45, 'Car rental': 140},
    'Barcelona': {'Flight': 370, 'Train': 110, 'Bus': 55, 'Car rental': 175},
    'Honolulu': {'Flight': 720, 'Train': 30, 'Bus': 60, 'Car rental': 210},
    'Berlin': {'Flight': 390, 'Train': 140, 'Bus': 70, 'Car rental': 185},
    'Marrakech': {'Flight': 420, 'Train': 80, 'Bus': 40, 'Car rental': 130},
    'Edinburgh': {'Flight': 410, 'Train': 120, 'Bus': 60, 'Car rental': 180},
    'Rome': {'Flight': 360, 'Train': 100, 'Bus': 50, 'Car rental': 170},
    'Bangkok': {'Flight': 680, 'Train': 45, 'Bus': 30, 'Car rental': 110},
    'Cape Town': {'Flight': 730, 'Train': 85, 'Bus': 55, 'Car rental': 160},
    'Vancouver': {'Flight': 480, 'Train': 95, 'Bus': 65, 'Car rental': 195},
    'Seoul': {'Flight': 780, 'Train': 220, 'Bus': 95, 'Car rental': 280},
    'Los Angeles': {'Flight': 520, 'Train': 110, 'Bus': 75, 'Car rental': 230},
    'Santorini': {'Flight': 430, 'Train': 60, 'Bus': 35, 'Car rental': 125},
    'Phnom Penh': {'Flight': 670, 'Train': 35, 'Bus': 25, 'Car rental': 100},
    'Athens': {'Flight': 440, 'Train': 90, 'Bus': 50, 'Car rental': 155},
    'Auckland': {'Flight': 760, 'Train': 75, 'Bus': 80, 'Car rental': 200}
}

# Nationality preferences (multipliers)
nationality_factors = {
    'American': {'Flight': 1.0, 'Car rental': 1.2},
    'British': {'Train': 1.3, 'Flight': 1.1},
    'Canadian': {'Flight': 1.1, 'Car rental': 1.1},
    'Japanese': {'Train': 1.4, 'Bus': 1.2},
    'Australian': {'Flight': 1.2, 'Car rental': 0.9},
    'Korean': {'Flight': 1.1, 'Train': 1.3},
    'Vietnamese': {'Bus': 1.3, 'Flight': 0.9},
    'Brazilian': {'Flight': 1.0, 'Bus': 1.1},
    'Dutch': {'Bike': 1.5, 'Train': 1.2},  # Assuming bike is an option for Dutch
    'Emirati': {'Flight': 1.3, 'Car rental': 1.4},
    'Mexican': {'Bus': 1.2, 'Car rental': 1.1},
    'Spanish': {'Train': 1.2, 'Bus': 1.1},
    'Chinese': {'Flight': 1.1, 'Train': 1.3},
    'German': {'Train': 1.4, 'Car rental': 1.1},
    'Moroccan': {'Bus': 1.3, 'Train': 1.1},
    'Scottish': {'Train': 1.2, 'Car rental': 1.0},
    'Indian': {'Train': 1.5, 'Flight': 1.0},
    'Italian': {'Train': 1.3, 'Car rental': 1.2},
    'South Korean': {'Flight': 1.2, 'Train': 1.4},
    'Taiwanese': {'Flight': 1.1, 'Bus': 1.2},
    'South African': {'Flight': 1.0, 'Car rental': 1.1},
    'French': {'Train': 1.5, 'Flight': 1.0},
    'Cambodian': {'Bus': 1.4, 'Flight': 0.8},
    'Greek': {'Bus': 1.2, 'Car rental': 1.0},
    'United Arab Emirates': {'Flight': 1.4, 'Car rental': 1.5},
    'Hong Kong': {'Flight': 1.2, 'Train': 1.1},
    'Singapore': {'Flight': 1.3, 'Train': 1.2},
    'Indonesian': {'Bus': 1.3, 'Flight': 0.9},
    'USA': {'Flight': 1.0, 'Car rental': 1.2},  # Same as American
    'UK': {'Train': 1.3, 'Flight': 1.1},  # Same as British
    'China': {'Flight': 1.1, 'Train': 1.3},  # Same as Chinese
    'New Zealander': {'Flight': 1.2, 'Car rental': 0.8}
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




# Model Training
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
    ('num', StandardScaler(),       ['Duration','Month','IsWeekend','IsPeakSeason']),
    ('cat', OneHotEncoder(),        ['Destination','AccommodationType','TravelerNationality'])
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
