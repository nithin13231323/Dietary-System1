
from .Diet_plans import dinner_name, daily_caloric_intake, lunch_name, breakfast_name

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# StandardScaler, NearestNeighbors, Pipeline, FunctionTransformer

data=pd.read_csv('/Users/nithinabraham/Downloads/Diet-Recommendation-System-main-3/recipes.csv')


dataset=data.copy()
columns=['RecipeId','Name','CookTime','PrepTime','TotalTime','RecipeIngredientParts','Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent','RecipeInstructions']
dataset=dataset[columns]

max_Calories=200000
max_daily_fat=100000
max_daily_Saturatedfat=13000
max_daily_Cholesterol=300000
max_daily_Sodium=2300000
max_daily_Carbohydrate=325000
max_daily_Fiber=40000
max_daily_Sugar=40000
max_daily_Protein=200000
max_list=[max_Calories,max_daily_fat,max_daily_Saturatedfat,max_daily_Cholesterol,max_daily_Sodium,max_daily_Carbohydrate,max_daily_Fiber,max_daily_Sugar,max_daily_Protein]

extracted_data=dataset.copy()
for column,maximum in zip(extracted_data.columns[6:15],max_list):
    extracted_data=extracted_data[extracted_data[column]<maximum]

    from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
prep_data=scaler.fit_transform(extracted_data.iloc[:,6:15].to_numpy())

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(metric='cosine',algorithm='brute')
neigh.fit(prep_data)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(neigh.kneighbors,kw_args={'return_distance':False})
pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])

params={'n_neighbors':10,'return_distance':False}
pipeline.get_params()
pipeline.set_params(NN__kw_args=params)

pipeline.transform(extracted_data.iloc[0:1,6:15].to_numpy())[0]

extracted_data.iloc[pipeline.transform(extracted_data.iloc[0:1,6:15].to_numpy())[0]]

extracted_data[extracted_data['RecipeIngredientParts'].str.contains("egg",regex=False)]

def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

def extract_data(dataframe,ingredient_filter,max_nutritional_values):
    extracted_data=dataframe.copy()
    for column,maximum in zip(extracted_data.columns[6:15],max_nutritional_values):
        extracted_data=extracted_data[extracted_data[column]<maximum]
    if ingredient_filter!=None:
        for ingredient in ingredient_filter:
            extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient,regex=False)] 
    return extracted_data

def apply_pipeline(pipeline,_input,extracted_data):
    return extracted_data.iloc[pipeline.transform(_input)[0]]

def recommand(dataframe,_input,max_nutritional_values,ingredient_filter=None,params={'return_distance':False}):
    extracted_data=extract_data(dataframe,ingredient_filter,max_nutritional_values)
    prep_data,scaler=scaling(extracted_data)
    neigh=nn_predictor(prep_data)
    pipeline=build_pipeline(neigh,scaler,params)
    return apply_pipeline(pipeline,_input,extracted_data)




# Function to get recommendations based on food name
def get_recommendations_for_meal(meal_name):
    # Filter the row by name instead of using iloc
    selected_rows = extracted_data[extracted_data['Name'] == meal_name]

    # Ensure at least one row exists before proceeding
    if not selected_rows.empty:
        # Select only the first row for recommendations
        selected_row = selected_rows.iloc[0:1]

        # Extract the required columns (Name and columns 6 to 15) and convert to a DataFrame
        test_input_df = selected_row.loc[:, ['Name'] + list(extracted_data.columns[6:15])]

        # Convert the required numerical columns (6 to 15) to a NumPy array for processing
        test_input = selected_row.iloc[:, 6:15].to_numpy()

        # Call the recommendation function and capture the result
        recommendations = recommand(dataset, test_input, max_list)

        # Remove the first item if it matches the input food name
        if isinstance(recommendations, pd.DataFrame):
            # Filter out the recommendation matching the input food name
            recommendations = recommendations[recommendations['Name'] != meal_name]

        return recommendations
    else:
        print(f"Recipe '{meal_name}' not found in the dataset.")
        return None

# Get recommendations for breakfast, lunch, and dinner
breakfast_recommendations = get_recommendations_for_meal(breakfast_name)
lunch_recommendations = get_recommendations_for_meal(lunch_name)
dinner_recommendations = get_recommendations_for_meal(dinner_name)

# Print recommendations for each meal
print("Breakfast Recommendations:")
if breakfast_recommendations is not None:
    print(breakfast_recommendations)

print("\nLunch Recommendations:")
if lunch_recommendations is not None:
    print(lunch_recommendations)

print("\nDinner Recommendations:")
if dinner_recommendations is not None:
    print(dinner_recommendations)