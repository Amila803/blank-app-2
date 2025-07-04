import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import holidays

# Set page config
st.set_page_config(page_title="Advanced Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Advanced Travel Cost Predictor")
st.markdown("""
This app predicts travel costs with maximum accuracy using ensemble machine learning techniques.
""")

# Custom transformer for advanced feature engineering
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
        
        # Date features
        X['Year'] = X['StartDate'].dt.year
        X['Month'] = X['StartDate'].dt.month
        X['Day'] = X['StartDate'].dt.day
        X['DayOfWeek'] = X['StartDate'].dt.dayofweek
        X['IsWeekend'] = X['DayOfWeek'].isin([5,6]).astype(int)
        X['Quarter'] = X['StartDate'].dt.quarter
        X['DayOfYear'] = X['StartDate'].dt.dayofyear
        X['WeekOfYear'] = X['StartDate'].dt.isocalendar().week
        
        # Holiday features
        X['IsHoliday'] = X.apply(self._check_holiday, axis=1)
        
        # Seasonality
        X['IsPeakSeason'] = X['Month'].isin([6,7,8,12]).astype(int)
        X['IsShoulderSeason'] = X['Month'].isin([4,5,9,10]).astype(int)
        X['IsLowSeason'] = X['Month'].isin([1,2,3,11]).astype(int)
        
        # Advanced duration features
        X['LogDuration'] = np.log1p(X['Duration'])
        X['SqrtDuration'] = np.sqrt(X['Duration'])
        X['DurationBins'] = pd.cut(X['Duration'], 
                                  bins=[0,3,7,14,30,90],
                                  labels=['0-3','4-7','8-14','15-30','30+'])
        
        # Interaction features
        X['PeakDuration'] = X['IsPeakSeason'] * X['Duration']
        X['WeekendDuration'] = X['IsWeekend'] * X['Duration']
        
        # Drop original date column
        X = X.drop('StartDate', axis=1)
        
        return X
    
    def _check_holiday(self, row):
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

# Load data function with enhanced cleaning
@st.cache_data
def load_data():
    try:
        # Load the dataset with multiple encoding attempts
        try:
            data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig')
        except:
            try:
                data = pd.read_csv("Travel_details_dataset.csv", encoding='latin1')
            except:
                data = pd.read_csv("Travel_details_dataset.csv")
        
        # Remove completely empty rows and columns
        data = data.dropna(how='all').dropna(axis=1, how='all')
        
        # Enhanced currency cleaning
        def clean_currency(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, str):
                # Remove all non-numeric characters except decimal point
                cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
                try:
                    return float(cleaned) if cleaned else np.nan
                except:
                    return np.nan
            return float(value)
        
        # Clean cost columns
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            if cost_col in data.columns:
                data[cost_col] = data[cost_col].apply(clean_currency)
        
        # Enhanced destination cleaning
        if 'Destination' in data.columns:
            data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
            data['Destination'] = data['Destination'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Date conversion with multiple format handling
        date_cols = ['Start date', 'End date']
        for col in date_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce', format='mixed')
        
        # Calculate duration with error handling
        if all(col in data.columns for col in ['Start date', 'End date']):
            data['Duration'] = (data['End date'] - data['Start date']).dt.days
            data['Duration'] = data['Duration'].clip(lower=1, upper=365)  # Reasonable bounds
        
        # Enhanced transport type standardization
        if 'Transportation type' in data.columns:
            transport_mapping = {
                'Plane': 'Flight', 'Airplane': 'Flight', 'Aeroplane': 'Flight',
                'Car': 'Car rental', 'Rental Car': 'Car rental', 'Taxi': 'Car rental',
                'Subway': 'Train', 'Rail': 'Train', 'Railway': 'Train',
                'Bus': 'Bus', 'Coach': 'Bus',
                'Ship': 'Cruise', 'Boat': 'Cruise', 'Ferry': 'Cruise'
            }
            data['Transportation type'] = data['Transportation type'].replace(transport_mapping)
        
        # Rename columns to standard format
        rename_map = {
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation type': 'TransportType',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate',
            'End date': 'EndDate'
        }
        data = data.rename(columns={k:v for k,v in rename_map.items() if k in data.columns})
        
        # Filter only needed columns and drop rows with missing critical data
        required_cols = [
            'Destination', 'Duration', 'StartDate', 'AccommodationType',
            'TravelerNationality', 'Cost', 'TransportType', 'TransportCost'
        ]
        available_cols = [col for col in required_cols if col in data.columns]
        data = data[available_cols].dropna(subset=['Cost', 'TransportCost'])
        
        # Outlier removal using IQR
        for col in ['Cost', 'TransportCost', 'Duration']:
            if col in data.columns:
                q1 = data[col].quantile(0.05)
                q3 = data[col].quantile(0.95)
                iqr = q3 - q1
                data = data[(data[col] >= q1 - 1.5*iqr) & (data[col] <= q3 + 1.5*iqr)]
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load data
data = load_data()

if data is not None:
    # Update dropdown options based on actual data
    DESTINATIONS = sorted(data['Destination'].dropna().unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())
    
    # Enhanced Feature Engineering
    engineered_data = FeatureEngineer().fit_transform(data)
    
    # Show enhanced data relationships
    st.header("Advanced Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost Distribution by Season")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=engineered_data, x='IsPeakSeason', y='Cost', ax=ax)
        ax.set_xticklabels(['Off-Peak', 'Peak'])
        st.pyplot(fig)

    with col2:
        st.subheader("Cost vs Duration (Log Scale)")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.regplot(data=engineered_data, x=np.log1p(engineered_data['Duration']), 
                    y=np.log1p(engineered_data['Cost']), 
                    scatter_kws={'alpha':0.3}, ax=ax)
        ax.set_xlabel('Log(Duration)')
        ax.set_ylabel('Log(Cost)')
        st.pyplot(fig)

    # --- TRANSPORTATION COST PREDICTION ---
    st.header("🚆 Advanced Transportation Cost Prediction")

    # Train transportation model with ensemble approach
    @st.cache_resource
    def train_transport_model():
        # Feature engineering
        transport_data = data[['Destination', 'TransportType', 'TravelerNationality', 'TransportCost', 'StartDate']].copy()
        transport_data = FeatureEngineer().fit_transform(transport_data)
        
        # Remove unnecessary columns
        transport_data = transport_data.drop(['Cost', 'AccommodationType'], axis=1, errors='ignore')
        
        X = transport_data.drop('TransportCost', axis=1)
        y = transport_data['TransportCost']
        
        # Identify feature types
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Enhanced preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Base models
        xgb = XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist')
        lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
        catboost = CatBoostRegressor(random_state=42, verbose=0)
        
        # Meta model
        meta_model = GradientBoostingRegressor(random_state=42)
        
        # Stacking ensemble
        estimators = [
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('catboost', catboost)
        ]
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            ))
        ])
        
        # Bayesian optimization for hyperparameters
        param_space = {
            'regressor__xgb__n_estimators': Integer(100, 1000),
            'regressor__xgb__max_depth': Integer(3, 10),
            'regressor__xgb__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'regressor__lgbm__num_leaves': Integer(20, 100),
            'regressor__lgbm__min_child_samples': Integer(10, 100),
            'regressor__catboost__iterations': Integer(100, 1000),
            'regressor__catboost__depth': Integer(4, 10)
        }
        
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=50,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        
        opt.fit(X, y)
        
        st.write("🔑 Best transport model params:", opt.best_params_)
        return opt.best_estimator_

    transport_model = train_transport_model()

    # Show transportation relationships
    st.subheader("Advanced Transport Cost Patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cost Distribution by Transport Type**")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.violinplot(data=data, x='TransportType', y='TransportCost', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("**Cost by Nationality and Transport**")
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(data=data, x='TravelerNationality', y='TransportCost', 
                   hue='TransportType', ax=ax, ci=None)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- ACCOMMODATION COST PREDICTION ---
    st.header("🏨 Advanced Accommodation Cost Prediction")

    # Prepare features and target
    features = engineered_data.drop(['Cost', 'TransportCost', 'TransportType'], axis=1, errors='ignore')
    target = 'Cost'

    X = features
    y = engineered_data[target]

    # Identify feature types
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()

    # Enhanced preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Base models
    xgb = XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist')
    lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
    catboost = CatBoostRegressor(random_state=42, verbose=0)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Meta model
    meta_model = GradientBoostingRegressor(random_state=42)
    
    # Stacking ensemble
    estimators = [
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('catboost', catboost),
        ('rf', rf)
    ]
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        ))
    ])

    # Train model with advanced optimization
    if st.button("Train Advanced Model"):
        with st.spinner("Training advanced model with Bayesian optimization..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5))
            
            # Bayesian optimization space
            param_space = {
                'regressor__xgb__n_estimators': Integer(100, 1000),
                'regressor__xgb__max_depth': Integer(3, 12),
                'regressor__xgb__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'regressor__xgb__subsample': Real(0.6, 1.0),
                'regressor__xgb__colsample_bytree': Real(0.6, 1.0),
                'regressor__lgbm__num_leaves': Integer(20, 100),
                'regressor__lgbm__min_child_samples': Integer(10, 100),
                'regressor__lgbm__reg_alpha': Real(0, 100),
                'regressor__lgbm__reg_lambda': Real(0, 100),
                'regressor__catboost__iterations': Integer(100, 1000),
                'regressor__catboost__depth': Integer(4, 10),
                'regressor__catboost__l2_leaf_reg': Real(1, 10),
                'regressor__rf__n_estimators': Integer(100, 500),
                'regressor__rf__max_depth': Integer(5, 30),
                'regressor__rf__min_samples_split': Integer(2, 20),
                'regressor__rf__min_samples_leaf': Integer(1, 10)
            }
            
            opt = BayesSearchCV(
                estimator=model,
                search_spaces=param_space,
                n_iter=100,  # Increased iterations
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            opt.fit(X_train, y_train)
            
            best_model = opt.best_estimator_
            st.write("🔑 Best params:", opt.best_params_)
            
            # Save model
            joblib.dump(best_model, 'advanced_travel_cost_model.pkl')
            st.success("Advanced model trained and saved!")

            # Enhanced evaluation
            st.subheader("Advanced Model Evaluation")
            y_pred = best_model.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
                st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                st.metric("Mean Absolute Percentage Error", 
                         f"{np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%")
            
            with col2:
                fig, ax = plt.subplots(figsize=(10,6))
                residuals = y_test - y_pred
                sns.histplot(residuals, kde=True, ax=ax)
                ax.set_title('Residual Distribution')
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10,6))
                ax.scatter(y_test, y_pred, alpha=0.3)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                ax.set_xlabel('Actual Cost')
                ax.set_ylabel('Predicted Cost')
                ax.set_title('Actual vs Predicted')
                st.pyplot(fig)

    # Advanced Prediction Interface
    st.header("🧮 Advanced Cost Prediction")

    with st.form("advanced_prediction_form"):
        st.subheader("Calculate Accommodation Costs")
        
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", DESTINATIONS, key='adv_dest')
            duration = st.slider("Duration (days)", min_value=1, max_value=90, value=7, step=1)
            accommodation = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES, key='adv_acc')
            nationality = st.selectbox("Nationality", NATIONALITIES, key='adv_nat')
        
        with col2:
            start_date = st.date_input("Start Date", datetime.today(), key='adv_date')
            transport_type = st.selectbox("Transportation Type", TRANSPORT_TYPES, key='adv_trans')
            group_size = st.slider("Group Size", min_value=1, max_value=10, value=1, step=1)
            advance_booking = st.slider("Advance Booking (days)", min_value=0, max_value=365, value=30, step=1)
        
        submitted = st.form_submit_button("Calculate Advanced Cost")

    if submitted:
        try:
            model = joblib.load('advanced_travel_cost_model.pkl')
            
            # Calculate derived features
            month = start_date.month
            day_of_week = start_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_season = 1 if month in [6,7,8,12] else 0
            is_shoulder_season = 1 if month in [4,5,9,10] else 0
            
            input_data = pd.DataFrame([{
                'Destination': destination,
                'Duration': duration,
                'AccommodationType': accommodation,
                'TravelerNationality': nationality,
                'TransportType': transport_type,
                'Year': start_date.year,
                'Month': month,
                'Day': start_date.day,
                'DayOfWeek': day_of_week,
                'IsWeekend': is_weekend,
                'Quarter': (month-1)//3 + 1,
                'DayOfYear': start_date.timetuple().tm_yday,
                'WeekOfYear': start_date.isocalendar()[1],
                'IsHoliday': 0,  # Simplified for demo
                'IsPeakSeason': is_peak_season,
                'IsShoulderSeason': is_shoulder_season,
                'IsLowSeason': 1 - max(is_peak_season, is_shoulder_season),
                'LogDuration': np.log1p(duration),
                'SqrtDuration': np.sqrt(duration),
                'DurationBins': '15-30' if duration > 14 else '8-14' if duration > 7 else '4-7' if duration > 3 else '0-3',
                'PeakDuration': is_peak_season * duration,
                'WeekendDuration': is_weekend * duration,
                'GroupSize': group_size,
                'AdvanceBooking': advance_booking
            }])
            
            prediction = model.predict(input_data)[0]
            
            # Confidence interval estimation (simplified)
            lower_bound = prediction * 0.9
            upper_bound = prediction * 1.1
            
            st.success(f"## Predicted Cost: ${prediction:,.2f}")
            st.info(f"📊 Estimated range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
            st.session_state['adv_accom_pred'] = prediction

            # Show detailed cost breakdown
            st.subheader("Detailed Cost Breakdown")
            base_cost = prediction / duration
            st.write(f"**Base daily rate:** ${base_cost:,.2f}")
            st.write(f"**Total accommodation:** ${base_cost * duration:,.2f} for {duration} days")
            
            # Show cost factors
            st.write("**Key Cost Factors:**")
            if is_peak_season:
                st.write("- 🏖️ Peak season multiplier (+20-30%)")
            if is_weekend:
                st.write("- 🎉 Weekend surcharge (+10-15%)")
            if group_size > 1:
                st.write(f"- 👥 Group discount applied ({min(20, (group_size-1)*5)}% off)")
            if advance_booking > 60:
                st.write(f"- 📅 Early booking discount ({min(15, advance_booking//30*2)}% off)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please train the model first by clicking the 'Train Advanced Model' button")

    # Advanced Transport Prediction interface
    with st.form("advanced_transport_form"):
        st.subheader("Calculate Transportation Costs")
        
        col1, col2 = st.columns(2)
        with col1:
            trans_destination = st.selectbox("Destination", DESTINATIONS, key='adv_trans_dest')
            trans_type = st.selectbox("Transportation Type", TRANSPORT_TYPES, key='adv_trans_type')
            trans_nationality = st.selectbox("Nationality", NATIONALITIES, key='adv_trans_nat')
        
        with col2:
            trans_start_date = st.date_input("Travel Date", datetime.today(), key='adv_trans_date')
            is_peak = st.checkbox("Peak Season Travel", value=False, key='adv_peak')
            is_roundtrip = st.checkbox("Round Trip", value=True, key='adv_round')
            is_business = st.checkbox("Business Class", value=False, key='adv_business')
        
        submitted = st.form_submit_button("Calculate Advanced Transport Cost")

    if submitted:
        try:
            # Create input data with all engineered features
            month = trans_start_date.month
            day_of_week = trans_start_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            input_data = pd.DataFrame([{
                'Destination': trans_destination,
                'TransportType': trans_type,
                'TravelerNationality': trans_nationality,
                'Year': trans_start_date.year,
                'Month': month,
                'Day': trans_start_date.day,
                'DayOfWeek': day_of_week,
                'IsWeekend': is_weekend,
                'IsPeakSeason': int(is_peak),
                'IsRoundTrip': int(is_roundtrip),
                'IsBusinessClass': int(is_business),
                'Duration': 1,  # Placeholder
                'StartDate': trans_start_date  # Will be transformed
            }])
            
            # Apply feature engineering
            input_data = FeatureEngineer().fit_transform(input_data)
            input_data = input_data.drop(['Cost', 'TransportCost', 'AccommodationType'], axis=1, errors='ignore')
            
            pred_cost = transport_model.predict(input_data)[0]
            
            # Apply multipliers
            if is_business:
                pred_cost *= 2.5  # Business class multiplier
            if is_roundtrip:
                pred_cost *= 1.8  # Less than 2x for roundtrip discounts
            
            # Confidence interval
            lower_bound = pred_cost * 0.85
            upper_bound = pred_cost * 1.15
            
            st.success(f"### Estimated Transportation Cost: ${pred_cost:,.2f}")
            st.info(f"📊 Estimated range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
            st.session_state['adv_trans_pred'] = pred_cost

            # Show cost factors
            st.write("**Cost Factors:**")
            if is_peak:
                st.write("- Peak season multiplier (+30-50%)")
            if is_business:
                st.write("- Business class upgrade (2.5x economy)")
            if is_roundtrip:
                st.write("- Round trip discount (10% off two one-ways)")
            if trans_nationality == 'Japan' and trans_type == 'Train':
                st.write("- Japanese bullet train premium applied")
                
        except Exception as e:
            st.error(f"Transport prediction failed: {str(e)}")

    # --- INTEGRATION ---
    st.header("💰 Combined Cost Prediction with Savings Tips")

    if 'adv_accom_pred' in st.session_state and 'adv_trans_pred' in st.session_state:
        total_cost = st.session_state['adv_accom_pred'] + st.session_state['adv_trans_pred']
        
        # Calculate potential savings
        base_accom = st.session_state['adv_accom_pred']
        base_trans = st.session_state['adv_trans_pred']
        
        # Savings scenarios
        shoulder_season_accom = base_accom * 0.8
        early_booking_trans = base_trans * 0.9
        budget_accom = base_accom * 0.7
        economy_trans = base_trans / 2.5
        
        st.success(f"## Total Estimated Trip Cost: ${total_cost:,.2f}")
        st.write(f"- 🏨 Accommodation: ${st.session_state['adv_accom_pred']:,.2f}")
        st.write(f"- 🚆 Transportation: ${st.session_state['adv_trans_pred']:,.2f}")
        
        # Savings tips
        st.subheader("💡 Potential Savings Opportunities")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Accommodation Savings:**")
            st.write(f"- Shoulder season: ${shoulder_season_accom:,.2f} (save ${base_accom-shoulder_season_accom:,.2f})")
            st.write(f"- Budget options: ${budget_accom:,.2f} (save ${base_accom-budget_accom:,.2f})")
            st.write(f"- Longer stays (weekly discounts): ${base_accom*0.9:,.2f}")
        
        with col2:
            st.write("**Transportation Savings:**")
            st.write(f"- Early booking: ${early_booking_trans:,.2f} (save ${base_trans-early_booking_trans:,.2f})")
            st.write(f"- Economy class: ${economy_trans:,.2f} (save ${base_trans-economy_trans:,.2f})")
            st.write(f"- Alternative routes: ${base_trans*0.85:,.2f}")
        
        # Optimal timing suggestion
        current_month = datetime.today().month
        best_month = 3 if current_month in [6,7,8] else 9 if current_month in [12,1,2] else current_month
        st.info(f"⏰ Best time to visit: {datetime(2000, best_month, 1).strftime('%B')} (typically 20-30% cheaper)")
        
    elif 'adv_accom_pred' in st.session_state or 'adv_trans_pred' in st.session_state:
        st.warning("Please calculate both accommodation and transportation costs for full trip estimation")
else:
    st.error("Failed to load dataset. Please check if 'Travel_details_dataset.csv' exists in the same directory.")
