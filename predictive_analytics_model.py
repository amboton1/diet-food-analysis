"""
Food-Attention Nexus Project - Predictive Analytics Model

This script implements a predictive analytics model to forecast changes in cognitive performance
based on dietary modifications. The model builds on the integrative analysis and subgroup findings
to create personalized predictions for how dietary changes might impact attention metrics.

Key components:
1. Data loading and preprocessing
2. Time-series feature engineering
3. Predictive model development
4. Scenario analysis for dietary modifications
5. Personalized recommendation engine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for models and results
os.makedirs('models/predictive', exist_ok=True)
os.makedirs('results/predictive', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess all datasets for predictive modeling"""
    
    print("Loading datasets...")
    
    # Load all datasets
    demographics = pd.read_csv('data/demographics.csv')
    health_data = pd.read_csv('data/health_metrics.csv')
    dietary_data = pd.read_csv('data/dietary_data.csv')
    cognitive_data = pd.read_csv('data/cognitive_performance.csv')
    
    # Convert date columns to datetime
    for df in [health_data, dietary_data, cognitive_data]:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
    
    print("Preprocessing and merging datasets...")
    
    # Merge cognitive and dietary data
    merged_data = pd.merge(
        cognitive_data,
        dietary_data,
        on=['participant_id', 'date'],
        how='inner',
        suffixes=('_cog', '_diet')
    )
    
    # Add demographic information
    merged_data = pd.merge(
        merged_data,
        demographics,
        on='participant_id',
        how='left'
    )
    
    # Add health metrics (use the most recent health record before each date)
    # First, sort health data by date
    health_data = health_data.sort_values(['participant_id', 'record_date'])
    
    # Create a function to get the most recent health record
    def get_recent_health_record(row):
        participant_records = health_data[health_data['participant_id'] == row['participant_id']]
        recent_record = participant_records[participant_records['record_date'] <= row['date']]
        
        if recent_record.empty:
            return None
        else:
            return recent_record.iloc[-1].name
    
    # Apply the function to get indices of recent health records
    print("Finding most recent health records for each data point...")
    health_indices = merged_data.apply(get_recent_health_record, axis=1)
    health_indices = health_indices.dropna()
    
    # Get health records and merge
    health_subset = health_data.loc[health_indices]
    health_subset = health_subset.reset_index(drop=True)
    
    # Create a temporary key for merging
    merged_data['temp_key'] = range(len(merged_data))
    health_subset['temp_key'] = health_indices.index
    
    # Merge with health data
    merged_data = pd.merge(
        merged_data,
        health_subset.drop(['participant_id', 'record_date'], axis=1),
        on='temp_key',
        how='left'
    )
    
    # Drop temporary key
    merged_data = merged_data.drop('temp_key', axis=1)
    
    # Create time-series features
    print("Creating time-series features...")
    
    # Sort by participant and date
    merged_data = merged_data.sort_values(['participant_id', 'date'])
    
    # Create lag features for dietary patterns (1-7 days)
    dietary_patterns = [
        'mediterranean_diet_score', 
        'high_protein_diet_score', 
        'high_fiber_diet_score', 
        'western_diet_score'
    ]
    
    for pattern in dietary_patterns:
        for lag in range(1, 8):
            merged_data[f'{pattern}_lag{lag}'] = merged_data.groupby('participant_id')[pattern].shift(lag)
    
    # Create lag features for macronutrients (1-3 days)
    macronutrients = ['protein_g', 'carbs_g', 'fats_g', 'fiber_g']
    
    for nutrient in macronutrients:
        for lag in range(1, 4):
            merged_data[f'{nutrient}_lag{lag}'] = merged_data.groupby('participant_id')[nutrient].shift(lag)
    
    # Create rolling averages for dietary patterns (3, 7, and 14 days)
    for pattern in dietary_patterns:
        for window in [3, 7, 14]:
            merged_data[f'{pattern}_roll{window}'] = merged_data.groupby('participant_id')[pattern].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Create diet quality index
    merged_data['diet_quality_index'] = (
        merged_data['mediterranean_diet_score'] * 0.4 +
        merged_data['high_fiber_diet_score'] * 0.3 +
        merged_data['high_protein_diet_score'] * 0.2 -
        merged_data['western_diet_score'] * 0.1
    ) / 100
    
    # Create lag and rolling features for diet quality index
    for lag in range(1, 8):
        merged_data[f'diet_quality_index_lag{lag}'] = merged_data.groupby('participant_id')['diet_quality_index'].shift(lag)
    
    for window in [3, 7, 14]:
        merged_data[f'diet_quality_index_roll{window}'] = merged_data.groupby('participant_id')['diet_quality_index'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Create lag features for cognitive metrics (to predict future values)
    cognitive_metrics = ['reaction_time_ms', 'sustained_attention_score', 'task_switching_cost_ms']
    
    for metric in cognitive_metrics:
        for lag in range(1, 4):
            merged_data[f'{metric}_lag{lag}'] = merged_data.groupby('participant_id')[metric].shift(lag)
    
    # Create day of week features
    merged_data['day_of_week'] = merged_data['date'].dt.dayofweek
    merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)
    
    # Create day of experiment feature (days since start)
    merged_data['days_since_start'] = merged_data.groupby('participant_id')['date'].transform(
        lambda x: (x - x.min()).dt.days
    )
    
    # Define age groups
    merged_data['age_group'] = pd.cut(
        merged_data['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['18-30', '31-45', '46-60', '60+']
    )
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    print(f"Final dataset shape: {merged_data.shape}")
    
    return merged_data

def build_predictive_model(data):
    """Build a predictive model to forecast cognitive performance based on dietary patterns"""
    
    print("Building predictive model...")
    
    # Define target variables (cognitive performance metrics)
    target_vars = [
        'reaction_time_ms',
        'sustained_attention_score',
        'task_switching_cost_ms'
    ]
    
    # Define feature groups
    demographic_features = [
        'age', 'gender', 'education', 'socioeconomic_status'
    ]
    
    current_diet_features = [
        'mediterranean_diet_score', 'high_protein_diet_score', 
        'high_fiber_diet_score', 'western_diet_score',
        'protein_g', 'carbs_g', 'fats_g', 'fiber_g',
        'processed_food_servings', 'vegetables_fruits_servings',
        'whole_grains_servings', 'sugar_beverages_servings'
    ]
    
    diet_history_features = [
        'mediterranean_diet_score_lag1', 'high_protein_diet_score_lag1',
        'high_fiber_diet_score_lag1', 'western_diet_score_lag1',
        'mediterranean_diet_score_roll7', 'high_protein_diet_score_roll7',
        'high_fiber_diet_score_roll7', 'western_diet_score_roll7',
        'diet_quality_index', 'diet_quality_index_roll7'
    ]
    
    cognitive_history_features = [
        'reaction_time_ms_lag1', 'sustained_attention_score_lag1', 
        'task_switching_cost_ms_lag1'
    ]
    
    health_features = [
        'bmi', 'sleep_quality', 'sleep_duration_hours', 
        'physical_activity_steps', 'stress_level'
    ]
    
    temporal_features = [
        'day_of_week', 'is_weekend', 'days_since_start'
    ]
    
    # Combine all features
    all_features = (
        demographic_features + 
        current_diet_features + 
        diet_history_features + 
        cognitive_history_features + 
        health_features + 
        temporal_features
    )
    
    # Identify categorical features
    categorical_features = ['gender', 'education']
    
    # Identify numerical features
    numerical_features = [f for f in all_features if f not in categorical_features]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Build individual models for each target variable
    models = {}
    feature_importances = {}
    
    for target in target_vars:
        print(f"\nBuilding model for {target}...")
        
        # Prepare data
        X = data[all_features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline with XGBoost
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', xgb.XGBRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=6,
                random_state=42
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Model performance: RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.3f}")
        
        # Save model
        joblib.dump(pipeline, f'models/predictive/{target}_xgboost.pkl')
        
        # Store model
        models[target] = pipeline
        
        # Get feature importances
        feature_names = (
            numerical_features + 
            pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features).tolist()
        )
        
        importances = pipeline.named_steps['model'].feature_importances_
        
        # Create DataFrame for feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save feature importances
        importance_df.to_csv(f'results/predictive/{target}_feature_importance.csv', index=False)
        
        # Store feature importances
        feature_importances[target] = importance_df
        
        # Plot top 20 feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title(f'Top 20 Feature Importances for {target}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'results/predictive/{target}_feature_importance.png', dpi=300)
        plt.close()
    
    return models, feature_importances

def create_dietary_modification_scenarios():
    """Create scenarios for dietary modifications to test with the predictive model"""
    
    print("Creating dietary modification scenarios...")
    
    # Define baseline diet (average Western diet)
    baseline_diet = {
        'mediterranean_diet_score': 40,
        'high_protein_diet_score': 50,
        'high_fiber_diet_score': 40,
        'western_diet_score': 70,
        'protein_g': 90,
        'carbs_g': 300,
        'fats_g': 100,
        'fiber_g': 15,
        'processed_food_servings': 6,
        'vegetables_fruits_servings': 3,
        'whole_grains_servings': 2,
        'sugar_beverages_servings': 3
    }
    
    # Define dietary modification scenarios
    scenarios = {
        'Baseline': baseline_diet.copy(),
        
        'Mediterranean Diet': {
            'mediterranean_diet_score': 80,
            'high_protein_diet_score': 60,
            'high_fiber_diet_score': 70,
            'western_diet_score': 30,
            'protein_g': 80,
            'carbs_g': 250,
            'fats_g': 70,
            'fiber_g': 30,
            'processed_food_servings': 1,
            'vegetables_fruits_servings': 8,
            'whole_grains_servings': 6,
            'sugar_beverages_servings': 0.5
        },
        
        'High Protein Diet': {
            'mediterranean_diet_score': 50,
            'high_protein_diet_score': 85,
            'high_fiber_diet_score': 50,
            'western_diet_score': 40,
            'protein_g': 120,
            'carbs_g': 150,
            'fats_g': 80,
            'fiber_g': 20,
            'processed_food_servings': 2,
            'vegetables_fruits_servings': 5,
            'whole_grains_servings': 3,
            'sugar_beverages_servings': 1
        },
        
        'High Fiber Diet': {
            'mediterranean_diet_score': 60,
            'high_protein_diet_score': 50,
            'high_fiber_diet_score': 85,
            'western_diet_score': 30,
            'protein_g': 70,
            'carbs_g': 220,
            'fats_g': 60,
            'fiber_g': 40,
            'processed_food_servings': 1,
            'vegetables_fruits_servings': 10,
            'whole_grains_servings': 8,
            'sugar_beverages_servings': 0.5
        },
        
        'Reduced Processed Food': {
            'mediterranean_diet_score': 60,
            'high_protein_diet_score': 60,
            'high_fiber_diet_score': 60,
            'western_diet_score': 40,
            'protein_g': 90,
            'carbs_g': 250,
            'fats_g': 80,
            'fiber_g': 25,
            'processed_food_servings': 2,
            'vegetables_fruits_servings': 6,
            'whole_grains_servings': 4,
            'sugar_beverages_servings': 1
        },
        
        'Increased Vegetables & Fruits': {
            'mediterranean_diet_score': 65,
            'high_protein_diet_score': 55,
            'high_fiber_diet_score': 70,
            'western_diet_score': 45,
            'protein_g': 85,
            'carbs_g': 270,
            'fats_g': 75,
            'fiber_g': 30,
            'processed_food_servings': 4,
            'vegetables_fruits_servings': 8,
            'whole_grains_servings': 3,
            'sugar_beverages_servings': 2
        },
        
        'No Sugar Beverages': {
            'mediterranean_diet_score': 55,
            'high_protein_diet_score': 55,
            'high_fiber_diet_score': 55,
            'western_diet_score': 50,
            'protein_g': 90,
            'carbs_g': 270,
            'fats_g': 90,
            'fiber_g': 20,
            'processed_food_servings': 5,
            'vegetables_fruits_servings': 4,
            'whole_grains_servings': 3,
            'sugar_beverages_servings': 0
        }
    }
    
    # Calculate diet quality index for each scenario
    for 
(Content truncated due to size limit. Use line ranges to read in chunks)