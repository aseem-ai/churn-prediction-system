import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Loads CSV data."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Basic cleaning logic."""
    # Force TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop missing values
    df = df.dropna()
    return df

def split_data(df, target_column='Churn', test_size=0.2):
    """Splits data into Train and Test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)