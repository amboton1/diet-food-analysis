"""
Food-Attention Nexus Project - Subgroup Analysis

This script performs subgroup analysis to examine whether certain populations
are more sensitive to dietary impacts on cognition. It analyzes the relationship
between dietary patterns and cognitive performance across different demographic
and health-related subgroups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for results
os.makedirs('results/subgroup_analysis', exist_ok=True)

def load_data():
    """Load and preprocess the integrated dataset"""
    
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
    
    print("Merging datasets for subgroup analysis...")
    
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
    
    # Create diet quality index
    merged_data['diet_quality_index'] = (
        merged_data['mediterranean_diet_score'] * 0.4 +
        merged_data['high_fiber_diet_score'] * 0.3 +
        merged_data['high_protein_diet_score'] * 0.2 -
        merged_data['western_diet_score'] * 0.1
    ) / 100
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    print(f"Final dataset shape: {merged_data.shape}")
    
    return merged_data

def define_subgroups(data):
    """Define subgroups for analysis"""
    
    print("Defining subgroups for analysis...")
    
    # Age groups
    data['age_group'] = pd.cut(
        data['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['18-30', '31-45', '46-60', '60+']
    )
    
    # Gender groups (already defined)
    
    # Education level groups (simplify)
    data['education_group'] = data['education'].map({
        'High School': 'Basic',
        'Some College': 'Basic',
        'Bachelor': 'Higher',
        'Master': 'Higher',
        'PhD': 'Higher'
    })
    
    # Socioeconomic status groups
    data['ses_group'] = pd.cut(
        data['socioeconomic_status'],
        bins=[0, 3, 7, 10],
        labels=['Low', 'Medium', 'High']
    )
    
    # BMI groups
    data['bmi_group'] = pd.cut(
        data['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # Exercise frequency groups (simplify if needed)
    data['exercise_group'] = data['exercise_frequency'].map({
        'None': 'Low',
        'Low': 'Low',
        'Moderate': 'Moderate',
        'High': 'High'
    })
    
    # Sleep quality groups
    data['sleep_quality_group'] = pd.cut(
        data['sleep_quality'],
        bins=[0, 50, 75, 100],
        labels=['Poor', 'Moderate', 'Good']
    )
    
    # Health condition groups
    data['health_condition'] = (
        (data['has_hypertension'] | 
         data['has_diabetes'] | 
         data['has_high_cholesterol'])
    ).map({True: 'Has Condition', False: 'Healthy'})
    
    # Screen time groups
    data['screen_time_group'] = pd.cut(
        data['screen_time_minutes'],
        bins=[0, 120, 240, 1000],
        labels=['Low', 'Medium', 'High']
    )
    
    # Define subgroup categories
    subgroup_categories = {
        'Demographic': ['age_group', 'gender', 'education_group', 'ses_group'],
        'Health': ['bmi_group', 'exercise_group', 'health_condition'],
        'Lifestyle': ['sleep_quality_group', 'screen_time_group']
    }
    
    return data, subgroup_categories

def analyze_diet_cognition_by_subgroup(data, subgroup_categories):
    """Analyze the relationship between diet and cognition across subgroups"""
    
    print("Analyzing diet-cognition relationship across subgroups...")
    
    # Cognitive metrics to analyze
    cognitive_metrics = [
        'reaction_time_ms', 
        'sustained_attention_score', 
        'task_switching_cost_ms'
    ]
    
    # Dietary metrics to analyze
    dietary_metrics = [
        'mediterranean_diet_score',
        'high_protein_diet_score',
        'high_fiber_diet_score',
        'western_diet_score',
        'diet_quality_index'
    ]
    
    # Store results
    subgroup_results = {}
    
    # Analyze each subgroup category
    for category, subgroups in subgroup_categories.items():
        print(f"Analyzing {category} subgroups...")
        
        category_results = {}
        
        # Analyze each subgroup variable
        for subgroup_var in subgroups:
            print(f"  Analyzing {subgroup_var}...")
            
            subgroup_result = {}
            
            # Get unique subgroup values
            subgroup_values = data[subgroup_var].unique()
            
            # For each cognitive metric
            for cognitive_metric in cognitive_metrics:
                metric_result = {}
                
                # For each dietary metric
                for dietary_metric in dietary_metrics:
                    # Calculate correlation for each subgroup value
                    correlations = {}
                    
                    for value in subgroup_values:
                        # Filter data for this subgroup
                        subgroup_data = data[data[subgroup_var] == value]
                        
                        # Calculate correlation
                        corr = subgroup_data[dietary_metric].corr(subgroup_data[cognitive_metric])
                        correlations[value] = corr
                    
                    metric_result[dietary_metric] = correlations
                
                subgroup_result[cognitive_metric] = metric_result
            
            category_results[subgroup_var] = subgroup_result
        
        subgroup_results[category] = category_results
    
    return subgroup_results

def visualize_subgroup_results(data, subgroup_results):
    """Visualize the results of the subgroup analysis"""
    
    print("Generating visualizations for subgroup analysis...")
    
    # For each category
    for category, category_results in subgroup_results.items():
        print(f"Creating visualizations for {category} subgroups...")
        
        # Create directory for this category
        os.makedirs(f'results/subgroup_analysis/{category}', exist_ok=True)
        
        # For each subgroup variable
        for subgroup_var, subgroup_result in category_results.items():
            # For each cognitive metric
            for cognitive_metric, metric_result in subgroup_result.items():
                # Create a DataFrame for plotting
                plot_data = []
                
                for dietary_metric, correlations in metric_result.items():
                    for subgroup_value, corr in correlations.items():
                        plot_data.append({
                            'Dietary Pattern': dietary_metric,
                            'Subgroup': subgroup_value,
                            'Correlation': corr
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                pivot_table = plot_df.pivot(
                    index='Subgroup', 
                    columns='Dietary Pattern', 
                    values='Correlation'
                )
                
                # Rename columns for better readability
                pivot_table.columns = [col.replace('_', ' ').title() for col in pivot_table.columns]
                
                # Create heatmap
                sns.heatmap(
                    pivot_table, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0, 
                    fmt='.2f',
                    linewidths=0.5
                )
                
                # Set title and labels
                metric_name = cognitive_metric.replace('_', ' ').title()
                plt.title(f'Diet-{metric_name} Correlation by {subgroup_var.replace("_", " ").title()}', fontsize=14)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(
                    f'results/subgroup_analysis/{category}/{subgroup_var}_{cognitive_metric}_heatmap.png',
                    dpi=300
                )
                plt.close()
                
                # Create bar plot for diet quality index only
                plt.figure(figsize=(10, 6))
                dqi_data = plot_df[plot_df['Dietary Pattern'] == 'diet_quality_index']
                
                # Sort by correlation value
                dqi_data = dqi_data.sort_values('Correlation')
                
                # Create bar plot
                bars = sns.barplot(
                    x='Correlation', 
                    y='Subgroup', 
                    data=dqi_data,
                    palette=['red' if x < 0 else 'blue' for x in dqi_data['Correlation']]
                )
                
                # Add vertical line at 0
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Set title and labels
                plt.title(f'Diet Quality Index - {metric_name} Correlation by {subgroup_var.replace("_", " ").title()}', fontsize=14)
                plt.xlabel('Correlation Coefficient', fontsize=12)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(
                    f'results/subgroup_analysis/{category}/{subgroup_var}_{cognitive_metric}_dqi_barplot.png',
                    dpi=300
                )
                plt.close()
    
    # Create box plots for diet quality index effect on cognitive metrics by subgroups
    for category, subgroups in {
        'Demographic': ['age_group', 'gender', 'education_group', 'ses_group'],
        'Health': ['bmi_group', 'health_condition'],
        'Lifestyle': ['sleep_quality_group']
    }.items():
        for subgroup_var in subgroups:
            for cognitive_metric in ['reaction_time_ms', 'sustained_attention_score', 'task_switching_cost_ms']:
                plt.figure(figsize=(12, 6))
                
                # Create box plot
                sns.boxplot(
                    x=subgroup_var,
                    y=cognitive_metric,
                    hue='diet_quality_index',
                    data=data,
                    palette='viridis'
                )
                
                # Set title and labels
                metric_name = cognitive_metric.replace('_', ' ').title()
                plt.title(f'{metric_name} by {subgroup_var.replace("_", " ").title()} and Diet Quality', fontsize=14)
                plt.ylabel(metric_name, fontsize=12)
                plt.xlabel(subgroup_var.replace('_', ' ').title(), fontsize=12)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(
                    f'results/subgroup_analysis/{category}/{subgroup_var}_{cognitive_metric}_boxplot.png',
                    dpi=300
                )
                plt.close()

def build_subgroup_models(data, subgroup_categories):
    """Build separate models for each subgroup to compare effect sizes"""
    
    print("Building machine learning models for subgroups...")
    
    # Target variables
    target_vars = [
        'reaction_time_ms',
        'sustained_attention_score',
        'task_switching_cost_ms'
    ]
    
    # Features to include
    features = [
        'mediterranean_diet_score', 'high_protein_diet_score', 
        'high_fiber_diet_score', 'western_diet_score',
        'protein_g', 'carbs_g', 'fats_g', 'fiber_g',
        'processed_food_servings', 'vegetables_fruits_servings',
        'whole_grains_servings', 'sugar_beverages_servings',
        'sleep_quality', 'sleep_duration_hours', 'physical_activity_steps'
    ]
    
    # Store results
    model_results = {}
    
    # For each subgroup category
    for category, subgroups in subgroup_categories.items():
        print(f"Building models for {category} subgroups...")
        
        category_results = {}
        
        # For each subgroup variable
        for subgroup_var in subgroups:
            print(f"  Building models for {subgroup_var}...")
            
            subgroup_result = {}
            
            # Get unique subgroup values
            subgroup_values = data[subgroup_var].unique()
            
            # For each target variable
            for target in target_vars:
                target_result = {}
                
                # For each subgroup value
                for value in subgroup_values:
                    # Filter data for this subgroup
                    subgroup_data = data[data[subgroup_var] == value]
                    
                    # Skip if not enough data
                    if len(subgroup_data) < 100:
                        print(f"    Skipping {subgroup_var}={value} (insufficient data)")
                        continue
                    
                    # Prepare data
                    X = subgroup_data[features]
                    y = subgroup_data[target]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
     
(Content truncated due to size limit. Use line ranges to read in chunks)