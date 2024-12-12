import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab 
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

data=pd.read_csv('/Users/nithinabraham/Downloads/Diet-Recommendation-System-main-3/recipes.csv')


stats.probplot(data.Calories.to_numpy(), dist="norm", plot=pylab)

dataset=data.copy()
columns=['RecipeId','Name','CookTime','PrepTime','TotalTime','RecipeIngredientParts','Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent','RecipeInstructions']
dataset=dataset[columns]

max_Calories=2000
max_daily_fat=100
max_daily_Saturatedfat=13
max_daily_Cholesterol=300
max_daily_Sodium=2300
max_daily_Carbohydrate=325
max_daily_Fiber=40
max_daily_Sugar=40
max_daily_Protein=200
max_list=[max_Calories,max_daily_fat,max_daily_Saturatedfat,max_daily_Cholesterol,max_daily_Sodium,max_daily_Carbohydrate,max_daily_Fiber,max_daily_Sugar,max_daily_Protein]

extracted_data=dataset.copy()
for column,maximum in zip(extracted_data.columns[6:15],max_list):
    extracted_data=extracted_data[extracted_data[column]<maximum]


scaler=StandardScaler()
prep_data=scaler.fit_transform(extracted_data.iloc[:,6:15].to_numpy())

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(metric='cosine',algorithm='brute')
neigh.fit(prep_data)

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


test_input=extracted_data.iloc[0:1,6:15].to_numpy()
recommand(dataset,test_input,max_list)

def search_nutrient_values_with_100g(item_name, extracted_data):
    """
    Search for nutrient values of a specific recipe and calculate values per 100g.

    Parameters:
    - item_name: str, the name of the recipe to search for.
    - extracted_data: pandas DataFrame, the dataset containing recipe and nutrient information.

    Returns:
    - dict: Nutrient values of the recipe (per serving and per 100g), or a message if not found.
    """
    # Filter rows where 'Name' matches the item_name
    result = extracted_data[extracted_data['Name'].str.contains(item_name, case=False, na=False)]

    if not result.empty:
        # Select relevant nutrient columns
        nutrient_columns = [
            'Calories', 'ProteinContent', 'FatContent', 
            'CarbohydrateContent', 'SugarContent', 'FiberContent', 
            'CholesterolContent'
        ]
        selected_row = result.iloc[0]  # Get the first matching result

        # Extract nutrient values and servings
        nutrient_values = selected_row[nutrient_columns]
        servings = selected_row.get("RecipeServings", 1)  # Default to 1 if not available

        # Approximate weight per serving based on nutrient values
        total_nutrient_weight = nutrient_values.sum()  # Approximation of weight in grams
        weight_per_serving = total_nutrient_weight / servings

       

        # Return results as dictionaries
        return {
            "Original Nutrients (per serving)": nutrient_values.to_dict(),
           
        }
    else:
        return f"Item '{item_name}' not found in the dataset."





