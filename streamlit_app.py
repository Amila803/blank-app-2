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
from pathlib import Path
import os

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ End-to-End Travel Cost Predictor")
st.markdown("""
This app demonstrates a complete machine learning pipeline:
1. Data loading and exploration
2. Data visualization
3. Data preprocessing
4. Model training with hyperparameter tuning
5. Model evaluation
6. Cost prediction
""")

# Section 1: Data Loading
st.header("1. Data Loading")
uploaded_file = st.file_uploader("Upload your travel cost dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")
    
    # Section 2: Data Exploration
    st.header("2. Data Exploration")
    
    st.subheader("Data Preview")
    st.write(data.head())
    
    st.subheader("Data Summary")
    st.write(data.describe())
    
    st.subheader("Missing Values")
    st.write(data.isna().sum())
    
    # Section 3: Data Visualization
    st.header("3. Data Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numeric Features Distribution")
        num_cols = data.select_dtypes(include=np.number).columns
        selected_num_col = st.selectbox("Select numeric column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(data[selected_num_col], kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Categorical Features Distribution")
        cat_cols = data.select_dtypes(exclude=np.number).columns
        selected_cat_col = st.selectbox("Select categorical column", cat_cols)
        fig, ax = plt.subplots()
        sns.countplot(data=data, x=selected_cat_col, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Section 4: Data Preprocessing
    st.header("4. Data Preprocessing")
    
    st.subheader("Select Target Variable")
    target = st.selectbox("Choose your target variable", data.columns)
    
    st.subheader("Select Features")
    features = st.multiselect("Choose your features", 
                             [col for col in data.columns if col != target],
                             default=[col for col in data.columns if col != target])
    
    # Preprocessing pipeline
    numeric_features = data[features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = data[features].select_dtypes(exclude=np.number).columns.tolist()
    
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
    
    st.success("Preprocessing pipeline configured!")
    
    # Section 5: Model Training
    st.header("5. Model Training")
    
    # Split data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection and hyperparameter tuning
    st.subheader("Model Configuration")
    model_choice = st.selectbox("Select model", ["Random Forest"])
    
    if model_choice == "Random Forest":
        st.subheader("Random Forest Hyperparameters")
        
        n_estimators = st.slider("Number of trees", 50, 500, 100, 50)
        max_depth = st.slider("Max depth", 3, 20, 10)
        min_samples_split = st.slider("Min samples split", 2, 10, 2)
        
        param_grid = {
            'regressor__n_estimators': [n_estimators],
            'regressor__max_depth': [max_depth],
            'regressor__min_samples_split': [min_samples_split]
        }
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        if st.button("Train Model"):
            with st.spinner("Training model (this may take a while)..."):
                grid_search = GridSearchCV(model, param_grid, cv=5, 
                                         scoring='neg_mean_squared_error',
                                         return_train_score=True)
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                st.success("Model training complete!")
                
                # Save model
                model_path = "travel_cost_model.pkl"
                joblib.dump(best_model, model_path)
                
                # Section 6: Model Evaluation
                st.header("6. Model Evaluation")
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("MSE", f"{mse:.2f}")
                col3.metric("RMSE", f"{rmse:.2f}")
                col4.metric("R² Score", f"{r2:.4f}")
                
                # Plot actual vs predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted Costs')
                st.pyplot(fig)
                
                # Feature importance
                if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
                    try:
                        # Get feature names after preprocessing
                        feature_names = numeric_features.copy()
                        if len(categorical_features) > 0:
                            ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                            cat_feature_names = ohe.get_feature_names_out(categorical_features)
                            feature_names.extend(cat_feature_names)
                        
                        # Get importances
                        importances = best_model.named_steps['regressor'].feature_importances_
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                        
                        # Plot
                        fig, ax = plt.subplots()
                        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                        ax.set_title('Top 10 Important Features')
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {str(e)}")
                
                # Section 7: Cost Prediction
                st.header("7. Cost Prediction")
                
                # Create input form
                with st.form("prediction_form"):
                    st.subheader("Enter Trip Details")
                    
                    input_data = {}
                    for feature in features:
                        if data[feature].dtype == 'object':
                            input_data[feature] = st.selectbox(feature, data[feature].unique())
                        else:
                            input_data[feature] = st.number_input(feature, 
                                                                min_value=float(data[feature].min()),
                                                                max_value=float(data[feature].max()),
                                                                value=float(data[feature].median()))
                    
                    submitted = st.form_submit_button("Predict Cost")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    prediction = best_model.predict(input_df)[0]
                    
                    st.success(f"### Predicted Travel Cost: ${prediction:,.2f}")
else:
    st.info("Please upload a CSV file to get started")
