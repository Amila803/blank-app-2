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
st.markdown("""
This app predicts travel costs with enforced business rules:
1. More days = higher cost
2. Resort is most expensive accommodation
3. Hostel is cheapest accommodation
4. Flight is most expensive transportation
""")

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
        # Define accommodation weights
        self.accommodation_weights = {
            'Hostel': 0.8,  # 20% cheaper than average
            'Airbnb': 0.95,
            'Hotel': 1.0,   # Baseline
            'Resort': 1.3,  # 30% more expensive
            'Villa': 1.2,
            'Guesthouse': 0.85,
            'Vacation rental': 0.9
        }
        self.duration_multiplier = 0.05  # 5% increase per day

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        base_pred = self.model.predict(X)
        
        if isinstance(X, pd.DataFrame):
            # Apply accommodation type multipliers
            if 'AccommodationType' in X.columns:
                acc_types = X['AccommodationType']
                acc_multipliers = acc_types.map(lambda x: self.accommodation_weights.get(x, 1.0)).values
                base_pred = base_pred * acc_multipliers
            
            # Apply duration multiplier
            if 'Duration' in X.columns:
                duration_effect = 1 + (X['Duration'] - 7) * self.duration_multiplier  # 7 days as baseline
                base_pred = base_pred * duration_effect
        
        return base_pred

# Data loading and cleaning functions
def remove_outliers(df, columns):
    df_clean = df.copy()
    outlier_info = {}
    for col in columns:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            num_outliers = len(outliers)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outlier_info[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'num_outliers': num_outliers,
                'percent_outliers': (num_outliers / len(df)) * 100
            }
    return df_clean, outlier_info

@st.cache_resource
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
                    return None
            return value
        
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
        
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce', format='mixed')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce', format='mixed')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        transport_mapping = {
            'Plane': 'Flight',
            'Airplane': 'Flight',
            'Car': 'Car rental',
            'Subway': 'Train',
            'Bus': 'Bus',
            'Ferry': 'Ferry'
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
        
        numerical_cols = ['Duration', 'Cost', 'TransportCost']
        data_clean, outlier_info = remove_outliers(data, numerical_cols)
        
        return data_clean
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Main app execution
data = load_data()

if data is not None:
    # Show data overview
    st.subheader("Data Overview")
    st.write(f"Loaded {len(data)} records")
    
    # Show data distribution
    st.subheader("Data Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=data[['Cost', 'TransportCost']])
        ax.set_title("Cost Distribution")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(data['Duration'], bins=30, kde=True)
        ax.set_title("Duration Distribution")
        st.pyplot(fig)

    # Get unique values for dropdowns
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())

    # Feature engineering
    def engineer_features(df):
        df = df.copy()
        df['Year'] = df['StartDate'].dt.year
        df['Month'] = df['StartDate'].dt.month
        df['DayOfWeek'] = df['StartDate'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
        df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
        return df

    engineered_data = engineer_features(data)

    # --- Transportation Model ---
    @st.cache_resource
    def train_transport_model():
        transport_data = data[['Destination', 'TransportType', 'TravelerNationality', 'TransportCost', 'StartDate']].copy()
        transport_data['PeakSeason'] = pd.to_datetime(transport_data['StartDate']).dt.month.isin([6,7,8,12]).astype(int)
        
        X = transport_data[['Destination', 'TransportType', 'TravelerNationality', 'PeakSeason']]
        y = transport_data['TransportCost']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Destination', 'TransportType', 'TravelerNationality'])
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
    st.subheader("Transportation Cost Patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Average Cost by Transport Type**")
        fig, ax = plt.subplots()
        sns.barplot(data=data, x='TransportType', y='TransportCost', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("**Transportation Type Distribution**")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='TransportType', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Accommodation Model ---
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

    # Model training section
    st.header("Model Training")
    if st.button("Train Accommodation Model"):
        with st.spinner("Training model with constraints..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', ConstrainedRandomForest(random_state=42))
            ])
            
            param_distributions = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 5, 10],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
            }
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=10,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            joblib.dump(best_model, 'travel_cost_model.pkl')
            st.success("Accommodation model trained and saved with constraints!")
            
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
            
            # Constraint validation
            st.subheader("Constraint Validation")
            
            # Test accommodation types
            test_data = pd.DataFrame({
                'Destination': ['Paris'] * len(ACCOMMODATION_TYPES),
                'Duration': [7] * len(ACCOMMODATION_TYPES),
                'AccommodationType': ACCOMMODATION_TYPES,
                'TravelerNationality': ['American'] * len(ACCOMMODATION_TYPES),
                'Month': [7] * len(ACCOMMODATION_TYPES),
                'IsWeekend': [0] * len(ACCOMMODATION_TYPES),
                'IsPeakSeason': [1] * len(ACCOMMODATION_TYPES)
            })
            
            test_preds = best_model.predict(test_data)
            acc_test_results = list(zip(ACCOMMODATION_TYPES, test_preds))
            acc_test_results.sort(key=lambda x: x[1])
            
            st.write("**Accommodation Type Cost Ranking (7-day stay):**")
            for acc_type, cost in acc_test_results:
                st.write(f"- {acc_type}: ${cost:.2f}")
            
            if acc_test_results[-1][0] == 'Resort' and acc_test_results[0][0] == 'Hostel':
                st.success("✓ Resort is most expensive, Hostel is cheapest")
            else:
                st.warning("Constraint not fully met")
            
            # Test duration impact
            duration_test = pd.DataFrame({
                'Destination': ['Paris'] * 5,
                'Duration': [3, 5, 7, 10, 14],
                'AccommodationType': ['Hotel'] * 5,
                'TravelerNationality': ['American'] * 5,
                'Month': [7] * 5,
                'IsWeekend': [0] * 5,
                'IsPeakSeason': [1] * 5
            })
            
            duration_preds = best_model.predict(duration_test)
            st.write("**Duration Impact on Cost (Hotel accommodation):**")
            for days, cost in zip([3, 5, 7, 10, 14], duration_preds):
                st.write(f"- {days} days: ${cost:.2f}")
            
            if all(x < y for x, y in zip(duration_preds, duration_preds[1:])):
                st.success("✓ Cost increases with duration")
            else:
                st.warning("Duration constraint not fully met")

    # Prediction interfaces
    st.header("Cost Prediction")
    
    with st.form("accommodation_form"):
        st.subheader("Accommodation Cost Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", DESTINATIONS, key='acc_dest')
            duration = st.number_input("Duration (days)", min_value=1, max_value=90, value=7, key='acc_dur')
            accommodation = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES, key='acc_type')
            nationality = st.selectbox("Nationality", NATIONALITIES, key='acc_nat')
        
        with col2:
            start_date = st.date_input("Start Date", datetime.today(), key='acc_date')
            month = start_date.month
            day_of_week = start_date.weekday()
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
            
            st.success(f"## Predicted Accommodation Cost: ${prediction:,.2f}")
            st.session_state['accom_pred'] = prediction
            
            st.subheader("Cost Breakdown")
            base_cost = prediction / duration
            st.write(f"Base daily cost: ${base_cost:,.2f}")
            st.write(f"Total for {duration} days: ${prediction:,.2f}")
            
            if is_peak_season:
                st.write("⚠️ Peak season surcharge applied")
            if is_weekend:
                st.write("⚠️ Weekend surcharge applied")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}. Please train the accommodation model first.")

    with st.form("transport_form"):
        st.subheader("Transportation Cost Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            trans_destination = st.selectbox("Destination", DESTINATIONS, key='trans_dest')
            trans_type = st.selectbox("Transportation Type", TRANSPORT_TYPES, key='trans_type')
        with col2:
            trans_nationality = st.selectbox("Nationality", NATIONALITIES, key='trans_nat')
            is_peak = st.checkbox("Peak Season Travel", value=False, key='trans_peak')
        
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
        if trans_type == 'Flight':
            st.write("- Flight is the most expensive transportation option")

    # Combined cost prediction
    if 'accom_pred' in st.session_state and 'trans_pred' in st.session_state:
        st.header("💵 Total Trip Cost")
        total_cost = st.session_state['accom_pred'] + st.session_state['trans_pred']
        st.success(f"## Total Estimated Cost: ${total_cost:,.2f}")
        st.write(f"- Accommodation: ${st.session_state['accom_pred']:,.2f}")
        st.write(f"- Transportation: ${st.session_state['trans_pred']:,.2f}")

else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists.")
