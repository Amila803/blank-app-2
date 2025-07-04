
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor



# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs based on actual travel data.
""")

# Load data function
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
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
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
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation type': 'TransportType',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate'
        })
        
        # Filter only needed columns and drop rows with missing critical data
        data = data[[
            'Destination', 'Duration', 'StartDate', 'AccommodationType',
            'TravelerNationality', 'Cost', 'TransportType', 'TransportCost'
        ]].dropna(subset=['Cost', 'TransportCost'])
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load data
data = load_data()

if data is not None:
    # Update dropdown options based on actual data
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())
    
    # Feature Engineering

    class FeatureEngineer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.holidays_us = holidays.US()
            self.holidays_uk = holidays.UK()
            self.holidays_jp = holidays.JP()
            self.holidays_de = holidays.DE()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Date features (only if StartDate exists)
        if 'StartDate' in X.columns:
            X['Year'] = X['StartDate'].dt.year
            X['Month'] = X['StartDate'].dt.month
            X['Day'] = X['StartDate'].dt.day
            X['DayOfWeek'] = X['StartDate'].dt.dayofweek
            X['IsWeekend'] = X['DayOfWeek'].isin([5,6]).astype(int)
            X['Quarter'] = X['StartDate'].dt.quarter
            X['DayOfYear'] = X['StartDate'].dt.dayofyear
            X['WeekOfYear'] = X['StartDate'].dt.isocalendar().week
            X = X.drop('StartDate', axis=1)
        
        # Holiday features
        if 'TravelerNationality' in X.columns:
            X['IsHoliday'] = X.apply(self._check_holiday, axis=1)
        
        # Seasonality
        if 'Month' in X.columns:
            X['IsPeakSeason'] = X['Month'].isin([6,7,8,12]).astype(int)
            X['IsShoulderSeason'] = X['Month'].isin([4,5,9,10]).astype(int)
            X['IsLowSeason'] = X['Month'].isin([1,2,3,11]).astype(int)
        
        # Duration features (only if Duration exists)
        if 'Duration' in X.columns:
            X['LogDuration'] = np.log1p(X['Duration'])
            X['SqrtDuration'] = np.sqrt(X['Duration'])
            X['DurationBins'] = pd.cut(X['Duration'], 
                                     bins=[0,3,7,14,30,90],
                                     labels=['0-3','4-7','8-14','15-30','30+'])
            if 'IsPeakSeason' in X.columns:
                X['PeakDuration'] = X['IsPeakSeason'] * X['Duration']
            if 'IsWeekend' in X.columns:
                X['WeekendDuration'] = X['IsWeekend'] * X['Duration']
        
        return X
    
    def _check_holiday(self, row):
        if 'StartDate' not in row or pd.isna(row['StartDate']):
            return 0
        if 'TravelerNationality' not in row:
            return 0
            
        date = row['StartDate']
        country = row['TravelerNationality']
        
        try:
            if country == 'United States':
                return date in self.holidays_us
            elif country == 'United Kingdom':
                return date in self.holidays_uk
            elif country == 'Japan':
                return date in self.holidays_jp
            elif country == 'Germany':
                return date in self.holidays_de
            return 0
        except:
            return 0
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

    # Train transportation model
    @st.cache_resource
    def train_transport_model():
        
        
        # Feature engineering
    transport_data = data[['Destination', 'TransportType', 'TravelerNationality', 'TransportCost', 'StartDate']].copy()
    
    # Add Duration with a default value if it doesn't exist
    if 'Duration' not in transport_data.columns:
        transport_data['Duration'] = 1  # Default duration for transportation
        
    transport_data = FeatureEngineer().fit_transform(transport_data)
    
    # Remove unnecessary columns
    transport_data = transport_data.drop(['Cost', 'AccommodationType'], axis=1, errors='ignore')
    
    X = transport_data.drop('TransportCost', axis=1)
    y = transport_data['TransportCost']
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Destination', 'TransportType', 'TravelerNationality'])
            ])
        
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', TransformedTargetRegressor(
                regressor=base,
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])

        model = Pipeline([
          ('preprocessor', preprocessor),
          ('regressor', LGBMRegressor(
              n_estimators=500,
              learning_rate=0.05,
              max_depth=10,
              random_state=42,
              n_jobs=-1
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
        sns.barplot(data=data, x='TransportType', y='TransportCost', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("**Nationality Preferences**")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='TravelerNationality', hue='TransportType', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- ACCOMMODATION COST PREDICTION ---
    st.header("Cost Prediction")

    # Prepare features and target
    features = ['Destination', 'Duration', 'AccommodationType', 'TravelerNationality', 
                'Month', 'IsWeekend', 'IsPeakSeason']
    target = 'Cost'

    X = engineered_data[features]
    y = engineered_data[target]

    # Preprocessing
    categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
    numeric_features = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
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

            param_distributions = {
                'regressor__n_estimators':       [100, 200, 300, 400, 500],
                'regressor__max_depth':          [None, 5, 10, 20, 30],
                'regressor__min_samples_split':  [2, 5, 10, 15],
                'regressor__min_samples_leaf':   [1, 2, 4, 6],
                'regressor__max_features':       ['sqrt', 'log2', None],
                'regressor__bootstrap':          [True, False]
            }
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=50,                     # sample 50 different combos
                cv=5,
                scoring='neg_mean_absolute_error',  # often better aligned with cost errors
                n_jobs=-1,
                random_state=42,
                verbose=2
            )
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            st.write("🔑 Best params:", search.best_params_)
           
            
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
        st.subheader("Calculate Accommodation Costs")
        
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
            'TravelerNationality': trans_nationality,
            'PeakSeason': int(is_peak)
        }])
        
        pred_cost = transport_model.predict(input_data)[0]
        
        st.success(f"### Estimated Transportation Cost: ${pred_cost:.2f}")
        st.session_state['trans_pred'] = pred_cost

        # Show cost factors
        st.write("**Cost Factors:**")
        if is_peak:
            st.write("- Peak season surcharge applied")
        if trans_nationality == 'Japanese' and trans_type == 'Train':
            st.write("- Japanese travelers typically prefer trains (higher quality expectation)")
        if trans_destination == 'Bali' and trans_type == 'Train':
            st.warning("Limited train options in Bali - consider flights or car rental")

    # --- INTEGRATION ---
    st.header("💵 Combined Cost Prediction")

    if 'accom_pred' in st.session_state and 'trans_pred' in st.session_state:
        total_cost = st.session_state['accom_pred'] + st.session_state['trans_pred']
        st.success(f"## Total Estimated Trip Cost: ${total_cost:.2f}")
        st.write(f"- Accommodation: ${st.session_state['accom_pred']:.2f}")
        st.write(f"- Transportation: ${st.session_state['trans_pred']:.2f}")
else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists in the same directory.")
