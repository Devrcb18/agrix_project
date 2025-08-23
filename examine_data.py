import pandas as pd
import pickle
import os

print("=== Examining Data Structure ===")

# Check crop recommendation data
print("\n1. Crop Recommendation Data:")
try:
    df_crop = pd.read_csv('Crop_recommendation (2).csv')
    print(f"Shape: {df_crop.shape}")
    print(f"Columns: {list(df_crop.columns)}")
    print(f"Unique crops: {df_crop['label'].unique()}")
    print(f"Sample data:")
    print(df_crop.head())
    print(f"\nData types:")
    print(df_crop.dtypes)
except Exception as e:
    print(f"Error reading crop recommendation data: {e}")

# Check crop yield data
print("\n2. Crop Yield Data:")
try:
    df_yield = pd.read_csv('crop_yield.csv')
    print(f"Shape: {df_yield.shape}")
    print(f"Columns: {list(df_yield.columns)}")
    print(f"Unique crops: {df_yield['Crop'].unique()[:10]}")  # First 10
    print(f"Sample data:")
    print(df_yield.head())
except Exception as e:
    print(f"Error reading crop yield data: {e}")

# Check available models
print("\n3. Available Models:")
models_dir = 'models'
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    print(f"Model files: {model_files}")
    
    # Try to load crop model to check structure
    try:
        with open(os.path.join(models_dir, 'model_for_crops.pkl'), 'rb') as f:
            crop_model = pickle.load(f)
            print(f"Crop model type: {type(crop_model)}")
            if hasattr(crop_model, 'feature_names_in_'):
                print(f"Feature names: {crop_model.feature_names_in_}")
            if hasattr(crop_model, 'classes_'):
                print(f"Classes: {crop_model.classes_}")
    except Exception as e:
        print(f"Error loading crop model: {e}")
    
    # Try to load yield model
    try:
        with open(os.path.join(models_dir, 'model_for_yield.pkl'), 'rb') as f:
            yield_model = pickle.load(f)
            print(f"Yield model type: {type(yield_model)}")
            if hasattr(yield_model, 'feature_names_in_'):
                print(f"Yield model features: {yield_model.feature_names_in_}")
    except Exception as e:
        print(f"Error loading yield model: {e}")
        
    # Try to load production model
    try:
        with open(os.path.join(models_dir, 'model_for_production.pkl'), 'rb') as f:
            production_model = pickle.load(f)
            print(f"Production model type: {type(production_model)}")
            if hasattr(production_model, 'feature_names_in_'):
                print(f"Production model features: {production_model.feature_names_in_}")
    except Exception as e:
        print(f"Error loading production model: {e}")
        
    # Check scaler
    try:
        with open(os.path.join(models_dir, 'scalar_for_crop.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            print(f"Scaler type: {type(scaler)}")
            if hasattr(scaler, 'feature_names_in_'):
                print(f"Scaler features: {scaler.feature_names_in_}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
else:
    print("Models directory not found")
