import pandas as pd
import numpy as np
import joblib
import os

def feature_engineering(df):
    """Feature engineering function exactly as used in training"""
    # Ensure column order matches training
    columns = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE',
               'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX_ENCODED']
    
    # Create engineered features
    df['THROMBOCYTE_LEUCOCYTE_RATIO'] = df['THROMBOCYTE'] / (df['LEUCOCYTE'] + 1e-6)
    df['ERYTHROCYTE_LEUCOCYTE'] = df['ERYTHROCYTE'] * df['LEUCOCYTE']
    
    # Add engineered features to columns
    columns.extend(['THROMBOCYTE_LEUCOCYTE_RATIO', 'ERYTHROCYTE_LEUCOCYTE'])
    
    # Ensure all columns are present and in correct order
    return df[columns]

def prepare_input_data(data):
    """Prepare input data with correct feature names and order"""
    # Create DataFrame with proper column names
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df = feature_engineering(df)
    
    return df

def load_model(model_path):
    """Load the model from the specified path"""
    try:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        if model is None:
            raise ValueError("Model failed to load")
            
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded model does not have predict method")
            
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(os.path.dirname(model_path))}")
        raise 