
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
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures



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
    def engineer_features(df):
        df = df.copy()
        # Extract date features
        df['Year'] = df['StartDate'].dt.year
        df['Month'] = df['StartDate'].dt.month
        df['DayOfMonth'] = df['StartDate'].dt.day
        df['DayOfWeek'] = df['StartDate'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
        
        # More sophisticated seasonality
        seasons = [(12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']
        for i, months in enumerate(seasons):
            df[f'Is{season_names[i]}'] = df['Month'].isin(months).astype(int)
        
        # Holiday periods
        df['IsHoliday'] = ((df['Month'] == 12) & (df['DayOfMonth'].between(15, 31)) | \
                          ((df['Month'] == 1) & (df['DayOfMonth'].between(1, 7))
        
        # Duration features
        df['LogDuration'] = np.log1p(df['Duration'])
        df['DurationSquared'] = df['Duration'] ** 2
        
        # Destination popularity
        dest_counts = df['Destination'].value_counts(normalize=True)
        df['DestinationPopularity'] = df['Destination'].map(dest_counts)
        
        # Traveler nationality frequency
        nationality_counts = df['TravelerNationality'].value_counts(normalize=True)
        df['NationalityFrequency'] = df['TravelerNationality'].map(nationality_counts)
       
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
               'Month', 'IsWeekend', 'IsPeakSeason', 'IsSummer', 'IsWinter',
               'IsHoliday', 'DurationSquared', 'LogDuration',
               'DestinationPopularity', 'NationalityFrequency']
    target = 'Cost'
    

    X = engineered_data[features]
    y = engineered_data[target]

    # Preprocessing
    categorical_features = ['Destination', 'AccommodationType', 'TravelerNationality']
    numeric_features = [col for col in features if col not in categorical_features + [target]]
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # Stacking Ensemble Model
    base_models = [
        ('xgb', XGBRegressor(random_state=42, n_jobs=-1)),
        ('lgbm', LGBMRegressor(random_state=42, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ]
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', RFECV(estimator=RandomForestRegressor(n_estimators=50), 
                                   cv=5, scoring='neg_mean_absolute_error')),
        ('regressor', StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(),
            cv=5,
            n_jobs=-1
        ))
    ])

            
    # Enhanced Hyperparameter Tuning
    param_distributions = {
        'feature_selection__estimator__max_depth': [3, 5, 7],
        'regressor__xgb__n_estimators': [100, 200, 300],
        'regressor__xgb__learning_rate': [0.01, 0.05, 0.1],
        'regressor__xgb__max_depth': [3, 5, 7],
        'regressor__lgbm__n_estimators': [100, 200, 300],
        'regressor__lgbm__learning_rate': [0.01, 0.05, 0.1],
        'regressor__lgbm__num_leaves': [31, 63, 127],
        'regressor__gbr__n_estimators': [100, 200],
        'regressor__gbr__learning_rate': [0.05, 0.1],
        'regressor__gbr__max_depth': [3, 5]
    }

    # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Apply log transformation to target for better performance
            y_train_transformed = np.log1p(y_train)
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=50,
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42,
                verbose=2
            )
            
            search.fit(X_train, y_train_transformed)
            
            best_model = search.best_estimator_
            st.write("🔑 Best params:", search.best_params_)
            
            # Evaluate on test set (reverse log transform)
            y_pred = np.expm1(best_model.predict(X_test))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
                st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            
            with col2:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                ax.set_xlabel('Actual Cost')
                ax.set_ylabel('Predicted Cost')
                ax.set_title('Actual vs Predicted Costs')
                st.pyplot(fig)
                
                # Feature importance plot
                try:
                    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
                        importances = best_model.named_steps['regressor'].feature_importances_
                    else:
                        importances = best_model.named_steps['regressor'].final_estimator_.coef_
                    
                    feature_names = (numeric_features + 
                                    list(best_model.named_steps['preprocessor']
                                        .named_transformers_['cat']
                                        .get_feature_names_out(categorical_features)))
                    feat_imp = pd.Series(importances, index=feature_names)
                    plt.figure(figsize=(10,6))
                    feat_imp.nlargest(20).plot(kind='barh')
                    plt.title('Top 20 Important Features')
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.warning(f"Could not plot feature importance: {str(e)}")

            # Save model
            joblib.dump(best_model, 'travel_cost_model.pkl')
            st.success("Model trained and saved!")        
          
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
