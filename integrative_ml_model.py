"""
Food-Attention Nexus Project - Integrative Machine Learning Model

This script implements an integrative machine learning model to analyze the relationship
between dietary patterns and cognitive performance. The model integrates data from
multiple sources to identify correlations and potential causal links.

Key components:
1. Data loading and preprocessing
2. Feature engineering
3. Correlation analysis
4. Machine learning model implementation
5. Model evaluation and interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import os
import joblib
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess all datasets"""
    
    print("Loading datasets...")
    
    # Load all datasets
    demographics = pd.read_csv('data/demographics.csv')
    health_data = pd.read_csv('data/health_metrics.csv')
    dietary_data = pd.read_csv('data/dietary_data.csv')
    grocery_data = pd.read_csv('data/grocery_purchases.csv')
    cognitive_data = pd.read_csv('data/cognitive_performance.csv')
    
    # Convert date columns to datetime
    for df in [health_data, dietary_data, grocery_data, cognitive_data]:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
    
    print("Preprocessing and merging datasets...")
    
    # Merge cognitive and dietary data (main analysis dataset)
    # This creates a dataset where each row represents a participant's cognitive performance
    # and dietary intake for a specific day
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
    
    # Create lag features for dietary data (1-3 days)
    print("Creating lag features for dietary patterns...")
    
    # Sort by participant and date
    merged_data = merged_data.sort_values(['participant_id', 'date'])
    
    # Dietary pattern scores to create lags for
    diet_lag_cols = [
        'mediterranean_diet_score', 
        'high_protein_diet_score', 
        'high_fiber_diet_score', 
        'western_diet_score'
    ]
    
    # Create lag features
    for col in diet_lag_cols:
        for lag in range(1, 4):  # 1, 2, and 3 day lags
            merged_data[f'{col}_lag{lag}'] = merged_data.groupby('participant_id')[col].shift(lag)
    
    # Create rolling averages for dietary patterns (3-day and 7-day)
    print("Creating rolling averages for dietary patterns...")
    
    for col in diet_lag_cols:
        # 3-day rolling average
        merged_data[f'{col}_roll3'] = merged_data.groupby('participant_id')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 7-day rolling average
        merged_data[f'{col}_roll7'] = merged_data.groupby('participant_id')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
    
    # Create diet quality index
    merged_data['diet_quality_index'] = (
        merged_data['mediterranean_diet_score'] * 0.4 +
        merged_data['high_fiber_diet_score'] * 0.3 +
        merged_data['high_protein_diet_score'] * 0.2 -
        merged_data['western_diet_score'] * 0.1
    ) / 100
    
    # Create lag features for diet quality index
    for lag in range(1, 4):
        merged_data[f'diet_quality_index_lag{lag}'] = merged_data.groupby('participant_id')['diet_quality_index'].shift(lag)
    
    # Create rolling averages for diet quality index
    merged_data['diet_quality_index_roll3'] = merged_data.groupby('participant_id')['diet_quality_index'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    merged_data['diet_quality_index_roll7'] = merged_data.groupby('participant_id')['diet_quality_index'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Create day of week features
    merged_data['day_of_week_num'] = merged_data['date'].dt.dayofweek
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    print(f"Final dataset shape: {merged_data.shape}")
    
    return merged_data, demographics, health_data, dietary_data, grocery_data, cognitive_data

def analyze_correlations(merged_data):
    """Analyze correlations between dietary patterns and cognitive performance"""
    
    print("Analyzing correlations between dietary patterns and cognitive performance...")
    
    # Select key variables for correlation analysis
    key_vars = [
        # Cognitive performance metrics
        'reaction_time_ms', 'sustained_attention_score', 'task_switching_cost_ms',
        
        # Current dietary metrics
        'mediterranean_diet_score', 'high_protein_diet_score', 
        'high_fiber_diet_score', 'western_diet_score',
        
        # Lagged dietary metrics
        'mediterranean_diet_score_lag1', 'high_protein_diet_score_lag1',
        'high_fiber_diet_score_lag1', 'western_diet_score_lag1',
        
        # Rolling average dietary metrics
        'mediterranean_diet_score_roll7', 'high_protein_diet_score_roll7',
        'high_fiber_diet_score_roll7', 'western_diet_score_roll7',
        
        # Diet quality index
        'diet_quality_index', 'diet_quality_index_lag1', 'diet_quality_index_roll7',
        
        # Macronutrients
        'protein_g', 'carbs_g', 'fats_g', 'fiber_g',
        
        # Food categories
        'processed_food_servings', 'vegetables_fruits_servings',
        'whole_grains_servings', 'sugar_beverages_servings',
        
        # Other factors
        'sleep_quality', 'sleep_duration_hours', 'physical_activity_steps'
    ]
    
    # Calculate correlation matrix
    corr_matrix = merged_data[key_vars].corr()
    
    # Save correlation matrix
    corr_matrix.to_csv('results/correlation_matrix_full.csv')
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                center=0, linewidths=0.5, vmin=-0.5, vmax=0.5)
    plt.title('Correlation Matrix: Dietary Patterns vs. Cognitive Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap_full.png', dpi=300)
    plt.close()
    
    # Focus on correlations with cognitive performance metrics
    cognitive_metrics = ['reaction_time_ms', 'sustained_attention_score', 'task_switching_cost_ms']
    
    # Extract correlations with cognitive metrics
    cognitive_corr = corr_matrix[cognitive_metrics].drop(cognitive_metrics)
    
    # Sort by absolute correlation values
    cognitive_corr_sorted = {}
    for metric in cognitive_metrics:
        cognitive_corr_sorted[metric] = cognitive_corr[metric].abs().sort_values(ascending=False)
    
    # Create bar plots for top correlations
    for metric in cognitive_metrics:
        plt.figure(figsize=(12, 8))
        top_corr = cognitive_corr_sorted[metric].head(15)
        colors = ['red' if corr_matrix.loc[var, metric] < 0 else 'blue' for var in top_corr.index]
        sns.barplot(x=corr_matrix.loc[top_corr.index, metric].values, y=top_corr.index, palette=colors)
        plt.title(f'Top 15 Correlations with {metric}', fontsize=14)
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/top_correlations_{metric}.png', dpi=300)
        plt.close()
    
    # Return the correlation matrices for further analysis
    return corr_matrix, cognitive_corr_sorted

def build_integrative_model(merged_data):
    """Build an integrative machine learning model to predict cognitive performance"""
    
    print("Building integrative machine learning models...")
    
    # Define target variables (cognitive performance metrics)
    target_vars = [
        'reaction_time_ms',
        'sustained_attention_score',
        'task_switching_cost_ms'
    ]
    
    # Define feature groups
    demographic_features = [
        'age', 'gender', 'education', 'occupation', 
        'socioeconomic_status', 'location'
    ]
    
    dietary_features = [
        # Current dietary metrics
        'mediterranean_diet_score', 'high_protein_diet_score', 
        'high_fiber_diet_score', 'western_diet_score',
        
        # Lagged dietary metrics
        'mediterranean_diet_score_lag1', 'high_protein_diet_score_lag1',
        'high_fiber_diet_score_lag1', 'western_diet_score_lag1',
        
        # Rolling average dietary metrics
        'mediterranean_diet_score_roll3', 'high_protein_diet_score_roll3',
        'high_fiber_diet_score_roll3', 'western_diet_score_roll3',
        'mediterranean_diet_score_roll7', 'high_protein_diet_score_roll7',
        'high_fiber_diet_score_roll7', 'western_diet_score_roll7',
        
        # Diet quality index
        'diet_quality_index', 'diet_quality_index_lag1', 
        'diet_quality_index_roll3', 'diet_quality_index_roll7',
        
        # Macronutrients
        'protein_g', 'carbs_g', 'fats_g', 'fiber_g',
        'omega3_mg', 'b_vitamins_pct', 'antioxidants_au', 'vitamin_d_iu',
        
        # Food categories
        'processed_food_servings', 'vegetables_fruits_servings',
        'whole_grains_servings', 'sugar_beverages_servings',
        
        # Meal timing
        'breakfast_time', 'lunch_time', 'dinner_time', 'snacking_frequency'
    ]
    
    health_features = [
        'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'blood_glucose',
        'has_hypertension', 'has_diabetes', 'has_high_cholesterol',
        'medication_count', 'is_smoker', 'alcohol_consumption',
        'exercise_frequency', 'stress_level', 'work_schedule'
    ]
    
    lifestyle_features = [
        'sleep_quality', 'sleep_duration_hours', 'physical_activity_steps',
        'heart_rate_variability', 'screen_time_minutes',
        'social_media_minutes', 'productivity_minutes', 'entertainment_minutes'
    ]
    
    temporal_features = [
        'day_of_week_num', 'is_weekend_cog',
        'morning_performance', 'afternoon_performance', 'evening_performance'
    ]
    
    # Combine all features
    all_features = (
        demographic_features + 
        dietary_features + 
        health_features + 
        lifestyle_features + 
        temporal_features
    )
    
    # Identify categorical features
    categorical_features = [
        'gender', 'education', 'occupation', 'location',
        'alcohol_consumption', 'exercise_frequency', 'work_schedule'
    ]
    
    # Identify numerical features
    numerical_features = [f for f in all_features if f not in categorical_features]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Store results
    model_results = {}
    
    # Build models for each target variable
    for target in target_vars:
        print(f"\nBuilding models for {target}...")
        
        # Prepare data
        X = merged_data[all_features]
        y = merged_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Store model performance
        model_performance = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
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
            
            # Store performance metrics
            model_performance[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            # Save model
            joblib.dump(pipeline, f'models/{target}_{name.replace(" ", "_").lower()}.pkl')
            
            # Feature importance (for tree-based models)
            if name in ['Random Forest', 'Gradient Boosting']:
                # Get feature names after preprocessing
                feature_names = (
                    numerical_features + 
                    pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features).tolist()
                )
                
                # Get feature importances
                importances = pipeline.named_steps['model'].feature_importances_
                
                # Create DataFrame for feature importances
                feature_importance = pd
(Content truncated due to size limit. Use line ranges to read in chunks)