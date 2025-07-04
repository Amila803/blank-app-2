
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
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor



# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs based on actual travel data.
""")

def remove_outliers(df, columns):
    """
    Remove outliers using IQR method for specified columns
    Returns cleaned dataframe and information about removed outliers
    """
    df_clean = df.copy()
    outlier_info = {}
    
    for col in columns:
        if col in df_clean.columns:
            # Calculate IQR
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            num_outliers = len(outliers)
            
            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            # Store info
            outlier_info[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'num_outliers': num_outliers,
                'percent_outliers': (num_outliers / len(df)) * 100
            }
    
    return df_clean, outlier_info

# Load data function
@st.cache_data
def load_data():
    try:
        # [Previous data loading code remains the same until the end of cleaning]
        
        # Filter only needed columns and drop rows with missing critical data
        data = data[[
            'Destination', 'Duration', 'StartDate', 'AccommodationType',
            'TravelerNationality', 'Cost', 'TransportType', 'TransportCost'
        ]].dropna(subset=['Cost', 'TransportCost'])
        
        # Remove outliers from numerical columns
        numerical_cols = ['Duration', 'Cost', 'TransportCost']
        data_clean, outlier_info = remove_outliers(data, numerical_cols)
        
        # Show outlier information
        if st.checkbox("Show Outlier Removal Information"):
            st.subheader("Outlier Removal Summary")
            for col, info in outlier_info.items():
                st.write(f"**{col}**:")
                st.write(f"- Removed {info['num_outliers']} outliers ({info['percent_outliers']:.2f}%)")
                st.write(f"- Lower bound: {info['lower_bound']:.2f}, Upper bound: {info['upper_bound']:.2f}")
                st.write("---")
        
        return data_clean
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
        
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
        transport_data = data[['Destination', 'TransportType', 'TravelerNationality', 'TransportCost']].copy()
        transport_data['PeakSeason'] = pd.to_datetime(data['StartDate']).dt.month.isin([6,7,8,12]).astype(int)
        
        X = transport_data[['Destination', 'TransportType', 'TravelerNationality', 'PeakSeason']]
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
        with st.spinner("Training model…"):
            # 1) split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # 2) redefine pipeline to use log-transform + LGBM
        model = Pipeline([
            ("preprocessor", preprocessor),  # your existing ColumnTransformer
            ("regressor", TransformedTargetRegressor(
                regressor=LGBMRegressor(
                    random_state=42,
                    n_jobs=-1
                ),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])

        # 3) broader hyperparameter distributions
        param_dist = {
            "regressor__regressor__n_estimators":       [200, 500, 800, 1200, 2000],
            "regressor__regressor__learning_rate":      [0.005, 0.01, 0.03, 0.05, 0.1],
            "regressor__regressor__num_leaves":         [31, 50, 70, 100, 150],
            "regressor__regressor__max_depth":          [-1, 5, 10, 20, 30],
            "regressor__regressor__min_child_samples":  [5, 10, 20, 30, 50],
            "regressor__regressor__subsample":          [0.6, 0.8, 1.0],
            "regressor__regressor__colsample_bytree":   [0.6, 0.8, 1.0],
            "regressor__regressor__reg_alpha":          [0, 0.1, 0.5, 1.0],
            "regressor__regressor__reg_lambda":         [0, 0.1, 0.5, 1.0],
        }

        # 4) use repeated CV for stability
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=80,               # try 80 random combos
            cv=cv,
            scoring="r2",            # directly optimize R²
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(X_train, y_train)

        # 5) report best params & metrics
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        st.write("🔑 Best hyperparameters:", search.best_params_)
        st.metric("R² (test)", f"{r2_score(y_test, y_pred):.3f}")
        st.metric("MAE (test)", f"${mean_absolute_error(y_test, y_pred):.2f}")

        # 6) persist your model
        joblib.dump(best_model, "travel_cost_model.pkl")
        st.success("Model trained, evaluated, and saved!")

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
