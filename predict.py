
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


data_processing_metadata_path = Path("../Data/DataProcessing/metadata.pkl")
with open(data_processing_metadata_path, 'rb') as f:
    loaded_dataprocessing_metadata = pickle.load(f)
feature_engineering_metadata_path = Path("../Data/FeatureEngineering/feature_engineering_metadata.pkl")
with open(feature_engineering_metadata_path, 'rb') as f:
    loaded_featureengineering_metadata = pickle.load(f)
with open('../Models/logistic_regression_model.pkl', 'rb') as f:    
    ds_model = pickle.load(f)
print(ds_model)
def predict_food(model,data):
    input_data = {}
    for feature in loaded_featureengineering_metadata['selected_features']:
        if feature not in data:
            input_data[feature] = 0
        else:
            input_data[feature] = data[feature]
    
    input_df = pd.DataFrame([input_data])
    for i in loaded_dataprocessing_metadata['numerical_features']:
        lower_bound, upper_bound = loaded_dataprocessing_metadata['outlier_boundaries'][i]
        input_df[i] = np.clip(input_df[i], lower_bound, upper_bound)
    input_df[loaded_dataprocessing_metadata['numerical_features']] = loaded_featureengineering_metadata['trained_scaler'].transform(input_df[loaded_dataprocessing_metadata['numerical_features']])
    print(input_df)
    prediction_encoded = model.predict(input_df)
    prediction_label = loaded_featureengineering_metadata['label_encoder'].inverse_transform(prediction_encoded)
    return f"Predicted Food Item: {prediction_encoded[0]} - {prediction_label[0]}"
# predict_food(ds_model['model'],{'Calories': 200, 'Protein': 0, 'Fat': 0, 'Carbs': 30, 'Sugar': 500, 'Fiber': 0, 'Sodium': 0,'Cholesterol': 0,'Glycemic_Index': 0,'Water_Content': 60, 'Serving_Size': 50,'Is_Vegan': 0,'Is_Gluten_Free': 0,'Meal_breakfast': 0,'Meal_dinner': 0,'Meal_lunch': 0,'Meal_snack': 1,'Prep_baked': 0,'Prep_fried': 0,'Prep_grilled': 0,'Prep_raw': 1})

