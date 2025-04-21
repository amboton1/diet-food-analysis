"""
Food-Attention Nexus Project - Simulated Data Generation

This script creates simulated datasets for the Food-Attention Nexus project:
1. Wearable & Mobile Data (Cognitive Performance Metrics)
2. Food Purchase & Dietary Data
3. Health Records & Demographics

The datasets are designed to have realistic relationships between dietary patterns
and cognitive performance, with appropriate confounding factors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create output directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Constants
NUM_PARTICIPANTS = 500
NUM_DAYS = 90
START_DATE = datetime(2024, 1, 1)  # Start date for the simulation

# Generate participant IDs and demographic data
def generate_demographic_data():
    """Generate demographic data for participants"""
    
    print("Generating demographic data...")
    
    # Create participant IDs
    participant_ids = [f"P{i:03d}" for i in range(1, NUM_PARTICIPANTS + 1)]
    
    # Age distribution (18-75)
    ages = np.random.normal(42, 15, NUM_PARTICIPANTS)
    ages = np.clip(ages, 18, 75).astype(int)
    
    # Gender (binary for simplicity, could be expanded)
    genders = np.random.choice(['Male', 'Female'], size=NUM_PARTICIPANTS)
    
    # Education level
    education_levels = np.random.choice(
        ['High School', 'Some College', 'Bachelor', 'Master', 'PhD'],
        p=[0.2, 0.3, 0.3, 0.15, 0.05],
        size=NUM_PARTICIPANTS
    )
    
    # Occupation categories
    occupation_categories = np.random.choice(
        ['Office Worker', 'Manual Labor', 'Healthcare', 'Education', 'Technology', 'Service', 'Retired', 'Student'],
        size=NUM_PARTICIPANTS
    )
    
    # Socioeconomic status (1-10 scale)
    ses = np.random.normal(5, 2, NUM_PARTICIPANTS)
    ses = np.clip(ses, 1, 10).round(1)
    
    # Geographic location (urban/suburban/rural)
    locations = np.random.choice(
        ['Urban', 'Suburban', 'Rural'],
        p=[0.6, 0.3, 0.1],
        size=NUM_PARTICIPANTS
    )
    
    # Create the demographics dataframe
    demographics = pd.DataFrame({
        'participant_id': participant_ids,
        'age': ages,
        'gender': genders,
        'education': education_levels,
        'occupation': occupation_categories,
        'socioeconomic_status': ses,
        'location': locations
    })
    
    return demographics

# Generate health metrics data
def generate_health_data(demographics):
    """Generate health metrics data based on demographics"""
    
    print("Generating health metrics data...")
    
    # Create empty lists to store data
    health_records = []
    
    # Generate monthly health records for each participant
    for _, row in demographics.iterrows():
        participant_id = row['participant_id']
        age = row['age']
        
        # Base BMI influenced by age
        base_bmi = np.random.normal(25 + (age - 40) * 0.05, 3)
        base_bmi = max(18.5, base_bmi)
        
        # Base blood pressure influenced by age
        base_systolic = 110 + age * 0.5 + np.random.normal(0, 5)
        base_diastolic = 70 + age * 0.2 + np.random.normal(0, 3)
        
        # Base cholesterol influenced by age
        base_cholesterol = 180 + age * 0.5 + np.random.normal(0, 10)
        
        # Base glucose influenced by age
        base_glucose = 90 + age * 0.1 + np.random.normal(0, 5)
        
        # Health conditions (more likely with age)
        has_hypertension = np.random.random() < (0.1 + age * 0.01)
        has_diabetes = np.random.random() < (0.05 + age * 0.005)
        has_high_cholesterol = np.random.random() < (0.1 + age * 0.008)
        
        # Medication use based on conditions
        medication_count = sum([has_hypertension, has_diabetes, has_high_cholesterol])
        medication_count += np.random.binomial(2, 0.1)  # Additional random medications
        
        # Lifestyle factors
        is_smoker = np.random.random() < 0.15
        alcohol_consumption = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], 
                                              p=[0.3, 0.4, 0.2, 0.1])
        exercise_frequency = np.random.choice(['None', 'Low', 'Moderate', 'High'], 
                                             p=[0.2, 0.3, 0.3, 0.2])
        
        # Self-reported stress (1-10)
        base_stress = np.random.normal(5, 1.5)
        base_stress = np.clip(base_stress, 1, 10)
        
        # Work schedule
        work_schedule = np.random.choice(['Regular', 'Shift Work'], p=[0.8, 0.2])
        
        # Generate monthly records
        for month in range(3):  # 3 months (90 days)
            # Add small variations for each month
            bmi = base_bmi + np.random.normal(0, 0.3)
            systolic = base_systolic + np.random.normal(0, 3)
            diastolic = base_diastolic + np.random.normal(0, 2)
            cholesterol = base_cholesterol + np.random.normal(0, 5)
            glucose = base_glucose + np.random.normal(0, 3)
            stress = base_stress + np.random.normal(0, 0.5)
            stress = np.clip(stress, 1, 10).round(1)
            
            # Record date (beginning of each month)
            record_date = START_DATE + timedelta(days=month*30)
            
            # Add to health records
            health_records.append({
                'participant_id': participant_id,
                'record_date': record_date,
                'bmi': round(bmi, 1),
                'systolic_bp': round(systolic),
                'diastolic_bp': round(diastolic),
                'cholesterol': round(cholesterol),
                'blood_glucose': round(glucose),
                'has_hypertension': has_hypertension,
                'has_diabetes': has_diabetes,
                'has_high_cholesterol': has_high_cholesterol,
                'medication_count': medication_count,
                'is_smoker': is_smoker,
                'alcohol_consumption': alcohol_consumption,
                'exercise_frequency': exercise_frequency,
                'stress_level': stress,
                'work_schedule': work_schedule
            })
    
    # Create the health dataframe
    health_df = pd.DataFrame(health_records)
    
    return health_df

# Generate daily dietary data
def generate_dietary_data(demographics, health_data):
    """Generate daily dietary data for each participant"""
    
    print("Generating dietary data...")
    
    # Create empty lists to store data
    dietary_records = []
    
    # Define dietary patterns
    dietary_patterns = {
        'Mediterranean': {
            'protein': (80, 10),  # (mean, std)
            'carbs': (250, 30),
            'fats': (70, 10),
            'fiber': (30, 5),
            'omega3': (2000, 500),  # mg
            'b_vitamins': (90, 10),  # % daily value
            'antioxidants': (80, 15),  # arbitrary units
            'vitamin_d': (600, 200),  # IU
            'processed_food': (1, 1),  # servings
            'vegetables_fruits': (8, 2),
            'whole_grains': (6, 2),
            'sugar_beverages': (0.5, 0.5)
        },
        'High-Protein': {
            'protein': (120, 20),
            'carbs': (150, 30),
            'fats': (80, 15),
            'fiber': (20, 5),
            'omega3': (1500, 500),
            'b_vitamins': (85, 10),
            'antioxidants': (60, 15),
            'vitamin_d': (500, 200),
            'processed_food': (2, 1),
            'vegetables_fruits': (5, 2),
            'whole_grains': (3, 1),
            'sugar_beverages': (1, 1)
        },
        'High-Fiber': {
            'protein': (70, 10),
            'carbs': (220, 30),
            'fats': (60, 10),
            'fiber': (40, 8),
            'omega3': (1200, 400),
            'b_vitamins': (95, 10),
            'antioxidants': (90, 10),
            'vitamin_d': (400, 150),
            'processed_food': (1, 1),
            'vegetables_fruits': (10, 2),
            'whole_grains': (8, 2),
            'sugar_beverages': (0.5, 0.5)
        },
        'Western': {
            'protein': (90, 15),
            'carbs': (300, 50),
            'fats': (100, 20),
            'fiber': (15, 5),
            'omega3': (800, 300),
            'b_vitamins': (60, 15),
            'antioxidants': (40, 15),
            'vitamin_d': (300, 150),
            'processed_food': (6, 2),
            'vegetables_fruits': (3, 2),
            'whole_grains': (2, 1),
            'sugar_beverages': (3, 2)
        }
    }
    
    # Assign primary dietary pattern to each participant
    # This will be their "base" pattern, but we'll add day-to-day variation
    participant_patterns = {}
    
    for participant_id in demographics['participant_id']:
        # Randomly assign a primary dietary pattern
        pattern = np.random.choice(list(dietary_patterns.keys()))
        participant_patterns[participant_id] = pattern
    
    # Generate daily dietary data for each participant
    for participant_id, primary_pattern in participant_patterns.items():
        # Get participant demographics
        participant_demo = demographics[demographics['participant_id'] == participant_id].iloc[0]
        
        # Get participant health data (use the first month's record)
        participant_health = health_data[health_data['participant_id'] == participant_id].iloc[0]
        
        # Base adherence to primary pattern (0-100)
        base_adherence = np.random.normal(80, 15)
        base_adherence = np.clip(base_adherence, 40, 100)
        
        # Calculate adherence to other patterns
        # Lower adherence to primary pattern means higher adherence to others
        other_patterns = [p for p in dietary_patterns.keys() if p != primary_pattern]
        other_adherence = {}
        for pattern in other_patterns:
            # Inverse relationship with primary pattern adherence
            adh = np.random.normal(100 - base_adherence, 15)
            adh = np.clip(adh, 0, 100 - (base_adherence * 0.5))  # Can't fully adhere to multiple patterns
            other_adherence[pattern] = adh
        
        # Generate daily records
        for day in range(NUM_DAYS):
            # Current date
            current_date = START_DATE + timedelta(days=day)
            
            # Day of week effect (weekends vs weekdays)
            is_weekend = current_date.weekday() >= 5
            
            # Adherence varies day to day
            if is_weekend:
                # People tend to deviate more from their patterns on weekends
                daily_adherence = base_adherence * np.random.normal(0.9, 0.1)
            else:
                daily_adherence = base_adherence * np.random.normal(1.0, 0.05)
            
            daily_adherence = np.clip(daily_adherence, 0, 100)
            
            # Calculate daily dietary values as weighted average of patterns
            # Start with primary pattern
            primary_weight = daily_adherence / 100
            
            # Initialize dietary values with contribution from primary pattern
            dietary_values = {}
            pattern_data = dietary_patterns[primary_pattern]
            
            for key, (mean, std) in pattern_data.items():
                # Add some random variation
                value = np.random.normal(mean, std) * primary_weight
                dietary_values[key] = value
            
            # Add contribution from other patterns
            remaining_weight = 1 - primary_weight
            if remaining_weight > 0:
                # Distribute remaining weight among other patterns
                total_adherence = sum(other_adherence.values())
                if total_adherence > 0:  # Prevent division by zero
                    for pattern, adherence in other_adherence.items():
                        pattern_weight = (adherence / total_adherence) * remaining_weight
                        pattern_data = dietary_patterns[pattern]
                        
                        for key, (mean, std) in pattern_data.items():
                            # Add contribution from this pattern
                            value = np.random.normal(mean, std) * pattern_weight
                            dietary_values[key] += value
            
            # Meal timing
            if is_weekend:
                breakfast_time = np.random.normal(9.5, 1)  # Later breakfast on weekends
            else:
                breakfast_time = np.random.normal(7.5, 0.5)
            
            lunch_time = breakfast_time + np.random.normal(5, 0.5)
            dinner_time = lunch_time + np.random.normal(6, 1)
            
            # Clip to reasonable hours
            breakfast_time = np.clip(breakfast_time, 5, 12)
            lunch_time = np.clip(lunch_time, 11, 15)
            dinner_time = np.clip(dinner_time, 17, 23)
            
            # Snacking frequency
            snacking_frequency = np.random.poisson(3)  # Average 3 snacks per day
            
            # Calculate adherence scores for each diet type
            mediterranean_score = np.random.normal(
                dietary_values['vegetables_fruits'] * 5 + 
                dietary_values['whole_grains'] * 5 - 
                dietary_values['processed_food'] * 10, 10)
            
            high_protein_score = np.random.normal(
                dietary_values['protein'] * 0.5 - 
                dietary_values['carbs'] * 0.2, 10)
            
            high_fiber_score = np.random.normal(
                dietary_values['fiber'] * 2 + 
                dietary_values['vegetables_fruits'] * 3, 10)
            
            western_score = np.random.normal(
                dietary_values['processed_food'] * 10 + 
                dietary_values['sugar_beverages'] * 10 - 
                dietary_values['vegetables_fruits'] * 5, 10)
            
            # Clip scores to 0-100 range
            mediterranean_score = np.clip(mediterranean_score, 0, 100)
            high_protein_score = np.clip(high_protein_score, 0, 100)
            high_fiber_score = np.clip(high_fiber_score, 0, 100)
            western_score = np.clip(western_score, 0, 100)
            
            # Add to dietary records
            dietary_records.append({
                'participant_id': participant_id,
                'date': current_date,
                'day_of_week': current_date.strftime('%A'),
                'is_weekend': is_weekend,
                'protein_g': round(dietary_values['protein'], 1),
                'carbs_g': round(dietary_values['carbs'], 1),
                'fats_g': round(dietary_values['fats'], 1),
                'fiber_g': round(dietary_values['fiber'], 1),
                'omega3_mg': round(float(dietary_values['omega3'])),
                'b_vitamins_pct': round(float(dietary_values['b_vitamins'])),
                'antioxidants_au': round(float(dietary_values['antioxidants'])),
                'vitamin_d_iu': round(float(dietary_values['vitamin_d'])),
                'processed_food_servings': round(dietary_values['processed_food'], 1),
                'vegetables_fruits_servings': round(dietary_values['vegetables_fruits'], 1),
                'whole_grains_servings': round(dietary_values['whole_grains'], 1),
                'sugar_beverages_servings': round(dietary_values['sugar_beverages'], 1),
                'breakfast_time': round(breakfast_time, 1),
                'lunch_time': round(lunch_time, 1),
                'dinner_time': round(dinner_time, 1),
                'snacking_frequency': snacking_frequency,
                'mediterranean_diet_score': round(mediterranean_score),
                'high_protein_diet_score': round(high_protein_score),
                'high_fiber_diet_score': round(high_fiber_score),
                'western_diet_score': round(western_score),
                'primary_dietary_
(Content truncated due to size limit. Use line ranges to read in chunks)