import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import os  # Operating system interactions
# Neural Networks with Autoencoder Model

import matplotlib.pyplot as plt # Plotting library


import seaborn as sns  # Statistical data visualization

from sklearn.preprocessing import MinMaxScaler  # Scaling features to a range
from sklearn.model_selection import train_test_split  # Splitting data into training and test sets
from sklearn.metrics.pairwise import cosine_similarity  # Calculating cosine similarity between vectors

import tensorflow as tf  # TensorFlow library for machine learning
from tensorflow.keras.models import Sequential  # Sequential model from Keras
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout  # Layers for the neural network
from tensorflow.keras.optimizers import Adam  # Optimizer for training the model

# Load the dataset from the specified CSV file
data = pd.read_csv('/Users/nithinabraham/Downloads/main project/Dietary System copy 2/recipes.csv')

# Display the first few rows of the dataset to understand its structure
data.head()

# Calculate Basal Metabolic Rate (BMR)
def compute_bmr(gender, body_weight, body_height, age):
    """
    Calculate Basal Metabolic Rate (BMR) based on gender, body weight, body height, and age.

    Args:
        gender (str): Gender of the individual ('male' or 'female').
        body_weight (float): Body weight of the individual in kilograms.
        body_height (float): Body height of the individual in centimeters.
        age (int): Age of the individual in years.

    Return:
        float: Basal Metabolic Rate (BMR) value.
    """
    if gender == 'male':
        # For Men: BMR = (10 x weight in kg) + (6.25 x height in cm) - (5 x age in years) + 5
        bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age + 5
    elif gender == 'female':
        # For Women: BMR = (10 x weight in kg) + (6.25 x height in cm) - (5 x age in years) - 161
        bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please choose 'male' or 'female'.")
    return bmr_value

def compute_daily_caloric_intake(bmr, activity_intensity, objective):
    """
    Calculate total daily caloric intake based on Basal Metabolic Rate (BMR), activity level, and personal goal.

    Args:
        bmr (float): Basal Metabolic Rate (BMR) value.
        activity_intensity (str): Activity level of the individual ('sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extra_active').
        objective (str): Personal goal of the individual ('weight_loss', 'muscle_gain', 'health_maintenance').

    Return:
        int: Total daily caloric intake.
    """
    # Define activity multipliers based on intensity
    intensity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }

    # Define goal adjustments based on objective
    objective_adjustments = {
        'weight_loss': 0.8,
        'muscle_gain': 1.2,
        'health_maintenance': 1
    }

    # Calculate maintenance calories based on activity intensity
    maintenance_calories = bmr * intensity_multipliers[activity_intensity]

    # Adjust maintenance calories based on personal objective
    total_caloric_intake = maintenance_calories * objective_adjustments[objective]

    return round(total_caloric_intake)



def find_recipes_near_target(caloric_goal, recipes_df, tolerance=50):
    """
    Find recipes close to the caloric goal.

    Args:
        caloric_goal (int): Target calories for the meal.
        recipes_df (pd.DataFrame): DataFrame containing recipe names and calories.
        tolerance (int): Allowable difference from the target calories.

    Returns:
        pd.DataFrame: Recommended recipes close to the caloric goal.
    """
    # Filter recipes within the tolerance range of the caloric goal
    matching_recipes = recipes_df[(recipes_df['Calories'] >= caloric_goal - tolerance) &
                                  (recipes_df['Calories'] <= caloric_goal + tolerance)]

    # Return an empty DataFrame if no recipes match the caloric goal
    if matching_recipes.empty:
        return pd.DataFrame()

    return matching_recipes



    
# Define user profile and dietary goal
category = 'male'  # Gender of the user
body_weight = 80   # Weight of the user in kilograms
body_height = 170  # Height of the user in centimeters
age = 46           # Age of the user in years
activity_intensity = 'moderately_active'  # Physical activity level of the user
objective = 'weight_loss'  # Dietary goal of the user

# Define tolerance for caloric matching
tolerance = 50  # Allowable difference from the target calories

# Calculate BMR and daily caloric intake
bmr_value = compute_bmr(category, body_weight, body_height, age)
daily_caloric_intake = compute_daily_caloric_intake(bmr_value, activity_intensity, objective)

# Display total daily calories required
print(f"\nTotal Daily Calories Required: {daily_caloric_intake} kcal\n")  # Ensure this is outside any conditional





def generate_meal_plan(category, body_weight, body_height, age, activity_intensity, objective, recipes_df, tolerance=50):
    """
    Generate meal recommendations for each meal based on user's profile and dietary goals.

    Args:
        category (str): Gender of the user ('male' or 'female').
        body_weight (float): Weight of the user in kilograms.
        body_height (float): Height of the user in centimeters.
        age (int): Age of the user in years.
        activity_intensity (str): Physical activity level of the user.
        objective (str): Dietary goal of the user ('weight_loss', 'muscle_gain', 'health_maintenance').
        recipes_df (pd.DataFrame): DataFrame containing recipe names and calories.
        tolerance (int): Allowable difference from the target calories.

    Returns:
        dict: Dictionary containing one recommendation per meal.
    """
    # Calculate the Basal Metabolic Rate (BMR)
    bmr_value = compute_bmr(category, body_weight, body_height, age)

    # Calculate the total daily caloric intake
    daily_caloric_intake = compute_daily_caloric_intake(bmr_value, activity_intensity, objective)

    # Define proportions for each meal
    meal_proportions = {'breakfast': 0.50, 'lunch': 0.40, 'dinner': 0.10}
    caloric_targets = {meal: int(daily_caloric_intake * prop) for meal, prop in meal_proportions.items()}

    # Prepare recommendations dictionary
    recommendations = {}
    for meal, target_calories in caloric_targets.items():
        options = find_recipes_near_target(target_calories, recipes_df, tolerance)
        if not options.empty:
            recommendations[meal] = options.iloc[0]  # Take the first recipe only
        else:
            recommendations[meal] = None  # No matching recipes found

    return recommendations


meal_plan = generate_meal_plan(
    category=category,
    body_weight=body_weight,
    body_height=body_height,
    age=age,
    activity_intensity=activity_intensity,
    objective=objective,
    recipes_df=data,
    tolerance=tolerance
)


print("Meal Plan Recommendations:")
for meal, recommendation in meal_plan.items():
    print(f"{meal.capitalize()} Recommendation:")
    if recommendation is not None:
        print(f"  - {recommendation['Name']} ({recommendation['Calories']} calories)")
    else:
        print("  No suitable recipe found.")
    print()  # Blank line for better readability


# Extract food names from the meal_plan
breakfast_name = meal_plan['breakfast']['Name'] if meal_plan['breakfast'] is not None else "No recommendation"
lunch_name = meal_plan['lunch']['Name'] if meal_plan['lunch'] is not None else "No recommendation"
dinner_name = meal_plan['dinner']['Name'] if meal_plan['dinner'] is not None else "No recommendation"

# Print the variables to verify
print(f"Breakfast: {breakfast_name}")
print(f"Lunch: {lunch_name}")
print(f"Dinner: {dinner_name}")