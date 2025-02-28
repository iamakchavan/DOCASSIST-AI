import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ModelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model
        
    def feature_engineering(self, df):
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
    
    def predict(self, X):
        """Make predictions after applying feature engineering"""
        X_transformed = self.feature_engineering(X)
        return self.model.predict(X_transformed)
    
    def fit(self, X, y=None):
        """Fit method (required for sklearn compatibility)"""
        return self 