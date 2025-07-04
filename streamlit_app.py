import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
from sklearn.feature_selection import SelectFromModel

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
This app predicts travel costs with enhanced accuracy using feature engineering and model optimization.
""")

# Improved data loading with more robust cleaning
@st.cache_data
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
                    return np.nan
            return value
        
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
        
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce', format='mixed')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce', format='mixed')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        transport_mapping = {
            'Plane': 'Flight', 'Airplane': 'Flight', 'Car': 'Car rental',
            'Subway': 'Train', 'Bus': 'Bus', 'Train': 'Train', 'Ferry': 'Ferry'
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
        
        # Advanced outlier handling using IQR
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.05)
            Q3 = df[column].quantile(0.95)
            IQR = Q3 - Q1
            return df[(df[column] >= Q1 - 1.5*IQR) & (df[column] <= Q3 + 1.5*IQR)]
        
        data = remove_outliers(data, 'Cost')
        data = remove_outliers(data, 'TransportCost')
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Enhanced feature engineering
def engineer_features(df):
    df = df.copy()
    # Date features
    df['Year'] = df['StartDate'].dt.year
    df['Month'] = df['StartDate'].dt.month
    df['DayOfWeek'] = df['StartDate'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
    
    # Advanced features
    df['DurationSquared'] = df['Duration'] ** 2
    df['LogDuration'] = np.log1p(df['Duration'])
    df['WeekendDuration'] = df['IsWeekend'] * df['Duration']
    df['PeakDuration'] = df['IsPeakSeason'] * df['Duration']
    
    # Destination popularity
    dest_counts = df['Destination'].value_counts(normalize=True)
    df['DestinationPopularity'] = df['Destination'].map(dest_counts)
    
    # Nationality preferences
    nationality_avg_cost = df.groupby('TravelerNationality')['Cost'].mean()
    df['NationalityAvgCost'] = df['TravelerNationality'].map(nationality_avg_cost)
    
    return df

# Load data
data = load_data()

if data is not None:
    engineered_data = engineer_features(data)
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())

    # Show data insights
    st.header("Data Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost Distribution")
        fig, ax = plt.subplots()
        sns.histplot(engineered_data['Cost'], kde=True, ax=ax, bins=30)
        st.pyplot(fig)

    with col2:
        st.subheader("Top Destinations by Cost")
        top_dests = engineered_data.groupby('Destination')['Cost'].mean().nlargest(10)
        fig, ax = plt.subplots()
        top_dests.sort_values().plot(kind='barh', ax=ax)
        st.pyplot(fig)

    # Create enhanced preprocessor with feature selection
    def create_preprocessor():
        numeric_features = ['Duration', 'Month', 'IsWeekend', 'IsPeakSeason',
                          'DurationSquared', 'LogDuration', 'WeekendDuration',
                          'PeakDuration', 'DestinationPopularity', 'NationalityAvgCost']
        categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ])
        
        return preprocessor

    # Enhanced model training with feature selection
    def train_model(X, y, model_name):
        preprocessor = create_preprocessor()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        param_grid = {
            'regressor__n_estimators': [200, 300, 400],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None],
            'feature_selection__threshold': ['median', 'mean', 0.1]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        
        # Save model and feature importance
        joblib.dump(best_model, f'travel_cost_model.pkl')
        
        # Get feature importance
        feature_importances = best_model.named_steps['regressor'].feature_importances_
        feature_names = (best_model.named_steps['preprocessor']
                        .get_feature_names_out())
        
        # Save feature importance
        pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).to_csv(f'{model_name}_feature_importance.csv', index=False)
        
        return best_model, grid_search.best_params_

    # Train models button with progress tracking
    if st.button("Train Enhanced Models"):
        with st.spinner("Training models with advanced feature engineering and tuning..."):
            # Train accommodation model
            X_accom = engineered_data.drop(columns=['Cost', 'TransportCost', 'StartDate', 'TransportType'])
            y_accom = engineered_data['Cost']
            accom_model, accom_params = train_model(X_accom, y_accom, 'accom')
            
            # Train transport model
            transport_data = data[['Destination', 'Duration', 'TransportType', 
                                 'TravelerNationality', 'StartDate']].copy()
            transport_data['PeakSeason'] = transport_data['StartDate'].dt.month.isin([6,7,8,12]).astype(int)
            transport_data = engineer_features(transport_data)
            X_trans = transport_data.drop(columns=['StartDate', 'TransportType'])
            y_trans = transport_data['TransportCost']
            trans_model, trans_params = train_model(X_trans, y_trans, 'trans')
            
            st.success("Models trained successfully with enhanced accuracy!")
            
            # Show best parameters
            st.subheader("Optimal Parameters Found")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Accommodation Model:**")
                st.json(accom_params)
            
            with col2:
                st.write("**Transport Model:**")
                st.json(trans_params)
            
            # Evaluate models
            st.subheader("Model Evaluation")
            
            # Accommodation evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_accom, y_accom, test_size=0.2, random_state=42)
            y_pred = accom_model.predict(X_test)
            
            st.write("**Accommodation Model Performance:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                
                baseline_mae = mean_absolute_error(y_test, [y_train.mean()]*len(y_test))
                improvement = 100*(baseline_mae - mean_absolute_error(y_test, y_pred))/baseline_mae
                st.metric("Improvement Over Baseline", f"{improvement:.1f}%")
            
            with col2:
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                ax.set_xlabel('Actual Cost')
                ax.set_ylabel('Predicted Cost')
                st.pyplot(fig)

    # Enhanced prediction interface
    st.header("Enhanced Cost Prediction")

    with st.form("enhanced_prediction_form"):
        st.subheader("Enter Trip Details")
        
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", DESTINATIONS)
            duration = st.number_input("Duration (days)", min_value=1, max_value=90, value=7)
            accommodation = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES)
        
        with col2:
            nationality = st.selectbox("Nationality", NATIONALITIES)
            start_date = st.date_input("Start Date", datetime.today())
            transport_type = st.selectbox("Transportation Type", TRANSPORT_TYPES)
        
        submitted = st.form_submit_button("Calculate Enhanced Prediction")

    if submitted:
        try:
            # Load models
            accom_model = joblib.load('travel_cost_model.pkl')
            trans_model = joblib.load('travel_cost_model.pkl')
            
            # Prepare input data with all engineered features
            month = start_date.month
            day_of_week = start_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_season = 1 if month in [6,7,8,12] else 0
            
            # Calculate additional features
            dest_popularity = engineered_data['Destination'].value_counts(normalize=True).get(destination, 0.5)
            nationality_avg = engineered_data.groupby('TravelerNationality')['Cost'].mean().get(nationality, 0)
            
            # Accommodation prediction
            accom_input = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'AccommodationType': accommodation,
                'TravelerNationality': nationality,
                'Month': month,
                'IsWeekend': is_weekend,
                'IsPeakSeason': is_peak_season,
                'DurationSquared': duration**2,
                'LogDuration': np.log1p(duration),
                'WeekendDuration': is_weekend * duration,
                'PeakDuration': is_peak_season * duration,
                'DestinationPopularity': dest_popularity,
                'NationalityAvgCost': nationality_avg
            }])
            
            accom_pred = accom_model.predict(accom_input)[0]
            
            # Transport prediction
            trans_input = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'TravelerNationality': nationality,
                'PeakSeason': is_peak_season,
                'Month': month,
                'IsWeekend': is_weekend,
                'DurationSquared': duration**2,
                'LogDuration': np.log1p(duration),
                'WeekendDuration': is_weekend * duration,
                'PeakDuration': is_peak_season * duration,
                'DestinationPopularity': dest_popularity,
                'NationalityAvgCost': nationality_avg
            }])
            
            trans_pred = trans_model.predict(trans_input)[0]
            
            total_cost = accom_pred + trans_pred
            
            # Enhanced results display
            st.success(f"## Enhanced Total Estimate: ${total_cost:,.2f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accommodation", f"${accom_pred:,.2f}")
                st.write(f"Type: {accommodation}")
                st.write(f"Duration: {duration} days")
            
            with col2:
                st.metric("Transportation", f"${trans_pred:,.2f}")
                st.write(f"Type: {transport_type}")
                st.write(f"Season: {'Peak' if is_peak_season else 'Off-peak'}")
            
            with col3:
                st.metric("Savings Tip", 
                         f"Try {'off-peak' if is_peak_season else 'weekday'} travel",
                         help="Based on current season analysis")
            
            # Advanced visualization
            st.subheader("Cost Composition")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accommodation breakdown
            accom_features = accom_model.named_steps['preprocessor'].get_feature_names_out()
            accom_importances = accom_model.named_steps['regressor'].feature_importances_
            top_accom = pd.DataFrame({'feature': accom_features, 'importance': accom_importances}) \
                        .nlargest(5, 'importance')
            sns.barplot(data=top_accom, y='feature', x='importance', ax=ax1)
            ax1.set_title('Top Accommodation Cost Factors')
            
            # Transport breakdown
            trans_features = trans_model.named_steps['preprocessor'].get_feature_names_out()
            trans_importances = trans_model.named_steps['regressor'].feature_importances_
            top_trans = pd.DataFrame({'feature': trans_features, 'importance': trans_importances}) \
                        .nlargest(5, 'importance')
            sns.barplot(data=top_trans, y='feature', x='importance', ax=ax2)
            ax2.set_title('Top Transportation Cost Factors')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please train the enhanced models first by clicking the 'Train Enhanced Models' button")

else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists in the same directory.")
