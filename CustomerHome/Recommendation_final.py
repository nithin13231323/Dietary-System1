from Recommendation import search_nutrient_values_with_100g, extracted_data

item_name = "Chicken Biriyani"  # Replace with your desired food item
result = search_nutrient_values_with_100g(item_name, extracted_data)

if isinstance(result, dict):
    print("Nutrient Values Per Serving:")
    print(result["Original Nutrients (per serving)"])
    
else:
    print(result)