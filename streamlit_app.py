import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs with improved accuracy using advanced machine learning techniques.
""")

# Enhanced data loading with better error handling
@st.cache_data
def load_data():
    try:
        # Load with optimized parameters
        data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig', engine='python')
        
        # Remove empty rows and duplicates
        data = data.dropna(how='all').drop_duplicates()
        
        # Enhanced currency cleaning
        def clean_currency(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, str):
                # Remove all non-numeric characters except decimal point
                value = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(value) if value else np.nan
            return float(value) if not pd.isna(value) else np.nan
        
        # Clean cost columns with outlier handling
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
            # Remove extreme outliers (top and bottom 1%)
            q_low = data[cost_col].quantile(0.01)
            q_hi = data[cost_col].quantile(0.99)
            data = data[(data[cost_col] >= q_low) & (data[cost_col] <= q_hi)]
        
        # Enhanced destination cleaning
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip().str.title()
        
        # More robust date conversion
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        data = data[data['Duration'] > 0]  # Remove negative durations
        
        # Enhanced transport type standardization
        transport_mapping = {
            'Plane': 'Flight', 'Airplane': 'Flight', 'Aeroplane': 'Flight',
            'Car': 'Car rental', 'Rental Car': 'Car rental', 'Taxi': 'Car rental',
            'Subway': 'Train', 'Rail': 'Train', 'Railway': 'Train'
        }
        data['Transportation type'] = data['Transportation type'].replace(transport_mapping)
        
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
    
    # Enhanced Feature Engineering
    @st.cache_data
    def engineer_features(df):
        df = df.copy()
        # Extract date features
        df['Year'] = df['StartDate'].dt.year
        df['Month'] = df['StartDate'].dt.month
        df['DayOfWeek'] = df['StartDate'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
        df['IsHolidaySeason'] = df['Month'].isin([11,12]).astype(int)
        
        # Create interaction features
        df['Duration_Peak'] = df['Duration'] * df['IsPeakSeason']
        df['Duration_Weekend'] = df['Duration'] * df['IsWeekend']
        
        # Destination popularity feature
        dest_counts = df['Destination'].value_counts(normalize=True)
        df['DestinationPopularity'] = df['Destination'].map(dest_counts)
        
        return df

    engineered_data = engineer_features(data)

    # Show enhanced data relationships
    st.header("Enhanced Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost Distribution")
        fig, ax = plt.subplots()
        sns.histplot(engineered_data['Cost'], kde=True, bins=30, ax=ax)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Cost by Destination (Top 10)")
        top_dests = engineered_data.groupby('Destination')['Cost'].mean().nlargest(10)
        fig, ax = plt.subplots()
        top_dests.sort_values().plot(kind='barh', ax=ax)
        st.pyplot(fig, use_container_width=True)

    # --- TRANSPORTATION COST PREDICTION ---
    st.header("🚆 Enhanced Transportation Cost Prediction")

    # Train enhanced transport model
    @st.cache_resource
    def train_transport_model():
        transport_data = data[['Destination', 'TransportType', 'TravelerNationality', 'TransportCost']].copy()
        transport_data['PeakSeason'] = pd.to_datetime(data['StartDate']).dt.month.isin([6,7,8,12]).astype(int)
        transport_data['HolidaySeason'] = pd.to_datetime(data['StartDate']).dt.month.isin([11,12]).astype(int)
        
        X = transport_data[['Destination', 'TransportType', 'TravelerNationality', 'PeakSeason', 'HolidaySeason']]
        y = transport_data['TransportCost']
        
        # Enhanced preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                ['Destination', 'TransportType', 'TravelerNationality']),
                ('scaler', RobustScaler(), ['PeakSeason', 'HolidaySeason'])
            ])
        
        # Optimized LGBM model with transformed target
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', TransformedTargetRegressor(
                regressor=LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])
        
        model.fit(X, y)
        return model

    transport_model = train_transport_model()

    # --- ACCOMMODATION COST PREDICTION ---
    st.header("🏨 Enhanced Accommodation Cost Prediction")

    # Prepare features and target
    features = ['Destination', 'Duration', 'AccommodationType', 'TravelerNationality', 
                'Month', 'IsWeekend', 'IsPeakSeason', 'IsHolidaySeason',
                'Duration_Peak', 'Duration_Weekend', 'DestinationPopularity']
    target = 'Cost'

    X = engineered_data[features]
    y = engineered_data[target]

    # Enhanced Preprocessing
    categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
    numeric_features = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason', 'IsHolidaySeason',
                       'Duration_Peak', 'Duration_Weekend', 'DestinationPopularity']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
        ], numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # Enhanced model with feature selection
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', RFECV(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42), 
            cv=5, 
            scoring='neg_mean_absolute_error',
            min_features_to_select=10
        )),
        ('regressor', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster training method
        ))
    ])

    # Enhanced Hyperparameter Tuning
    param_distributions = {
        'feature_selection__estimator__max_depth': [3, 5, 7, 9],
        'feature_selection__estimator__min_samples_split': [2, 5, 10],
        'regressor__n_estimators': [300, 500, 700],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [5, 7, 9],
        'regressor__subsample': [0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Train model with enhanced options
    if st.button("Train Enhanced Model"):
        with st.spinner("Training enhanced model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Apply log transformation to target for better performance
            y_train_transformed = np.log1p(y_train)
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=30,  # Reduced for faster training
                cv=3,      # Reduced for faster training
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            search.fit(X_train, y_train_transformed)
            
            best_model = search.best_estimator_
            st.write("🔑 Best params:", search.best_params_)
            
            # Evaluate on test set (reverse log transform)
            y_pred = np.expm1(best_model.predict(X_test))
            
            # Enhanced evaluation metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error")
                st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error")
                st.metric("R² Score", f"{r2:.4f}", help="Variance explained by model")
                st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
            
            with col2:
                # Enhanced actual vs predicted plot
                fig, ax = plt.subplots(figsize=(8,6))
                sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                ax.set_xlabel('Actual Cost')
                ax.set_ylabel('Predicted Cost')
                ax.set_title('Actual vs Predicted Costs')
                st.pyplot(fig, use_container_width=True)

            # Feature importance analysis
            try:
                st.subheader("Feature Importance")
                result = permutation_importance(
                    best_model, X_test, np.log1p(y_test), 
                    n_repeats=5, random_state=42, n_jobs=-1
                )
                
                sorted_idx = result.importances_mean.argsort()
                fig, ax = plt.subplots(figsize=(10,6))
                ax.boxplot(
                    result.importances[sorted_idx].T,
                    vert=False, 
                    labels=X_test.columns[sorted_idx]
                )
                ax.set_title("Permutation Importances (test set)")
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot feature importance: {str(e)}")

            # Save model
            joblib.dump(best_model, 'enhanced_travel_cost_model.pkl')
            st.success("Enhanced model trained and saved successfully!")

    # Enhanced Prediction Interface
    st.header("💰 Enhanced Cost Prediction")

    with st.form("enhanced_prediction_form"):
        st.subheader("Calculate Accommodation Costs")
        
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", options['DESTINATIONS'], key='enh_dest')
            duration = st.number_input("Duration (days)", min_value=1, max_value=90, value=7, key='enh_dur')
            accommodation = st.selectbox("Accommodation Type", options['ACCOMMODATION_TYPES'], key='enh_acc')
            nationality = st.selectbox("Nationality", options['NATIONALITIES'], key='enh_nat')
        
        with col2:
            start_date = st.date_input("Start Date", datetime.today(), key='enh_date')
            month = start_date.month
            is_weekend = 1 if start_date.weekday() >= 5 else 0
            is_peak_season = 1 if month in [6,7,8,12] else 0
            is_holiday_season = 1 if month in [11,12] else 0
            
            # Calculate destination popularity (mock)
            dest_popularity = 0.5  # Default
            if destination in options['DESTINATIONS'][:5]:
                dest_popularity = 0.8
            elif destination in options['DESTINATIONS'][-5:]:
                dest_popularity = 0.3
        
        submitted = st.form_submit_button("Calculate Enhanced Cost")

    if submitted:
        try:
            model = joblib.load('enhanced_travel_cost_model.pkl')
            
            input_data = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'AccommodationType': accommodation,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': is_weekend,
                'IsPeakSeason': is_peak_season,
                'IsHolidaySeason': is_holiday_season,
                'Duration_Peak': duration * is_peak_season,
                'Duration_Weekend': duration * is_weekend,
                'DestinationPopularity': dest_popularity
            }])
            
            prediction = model.predict(input_data)[0]
            
            st.success(f"## Predicted Cost: ${prediction:,.2f}")
            st.session_state['enh_accom_pred'] = prediction

            # Enhanced cost breakdown
            st.subheader("Cost Breakdown")
            base_cost = prediction / duration
            st.write(f"Base daily cost: ${base_cost:,.2f}")
            st.write(f"Total for {duration} days: ${base_cost * duration:,.2f}")
            
            if is_peak_season:
                st.write("⚠️ Peak season surcharge applied")
            if is_weekend:
                st.write("⚠️ Weekend surcharge applied")
            if is_holiday_season:
                st.write("⚠️ Holiday season premium applied")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # --- INTEGRATION ---
    st.header("💵 Enhanced Combined Cost Prediction")

    if 'enh_accom_pred' in st.session_state and 'trans_pred' in st.session_state:
        total_cost = st.session_state['enh_accom_pred'] + st.session_state['trans_pred']
        st.success(f"## Total Estimated Trip Cost: ${total_cost:,.2f}")
        st.write(f"- Accommodation: ${st.session_state['enh_accom_pred']:,.2f}")
        st.write(f"- Transportation: ${st.session_state['trans_pred']:,.2f}")
else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists.")
