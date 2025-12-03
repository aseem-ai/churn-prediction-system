import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Logic: People with partners AND dependents are likely more stable
        X['FamilySize'] = (X['Partner'] == 'Yes').astype(int) + (X['Dependents'] == 'Yes').astype(int)
        
        # 2. Logic: High charges over short tenure = High Risk
        X['AvgChargesPerMonth'] = X['TotalCharges'] / (X['tenure'] + 1)
        
        # 3. Binning Tenure (grouping new users vs loyal users)
        X['TenureGroup'] = pd.cut(X['tenure'], bins=[0, 12, 24, 48, 60, 100], labels=[1, 2, 3, 4, 5])
        
        return X