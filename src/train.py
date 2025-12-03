import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.preprocessing import load_data, clean_data, split_data

def build_pipeline(numeric_features, categorical_features):
    """Creates the ML pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def train_and_save(data_path, model_path):
    print("Loading & Cleaning Data...")
    df = load_data(data_path)
    df = clean_data(df)
    
    features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                'gender', 'Partner', 'Dependents', 'InternetService', 
                'Contract', 'PaymentMethod']
    
    # Filter only relevant columns + target
    df = df[features + ['Churn']]
    
    X_train, X_test, y_train, y_test = split_data(df, target_column='Churn')
    
    print("Building Model...")
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['SeniorCitizen', 'gender', 'Partner', 'Dependents', 
                            'InternetService', 'Contract', 'PaymentMethod']
    
    model = build_pipeline(numeric_features, categorical_features)
    
    print("Training...")
    model.fit(X_train, y_train)
    
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save("data/data.csv", "models/model.pkl")