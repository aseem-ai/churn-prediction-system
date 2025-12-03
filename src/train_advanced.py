import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.preprocessing import load_data, clean_data, split_data
from src.features import FeatureEngineer

def train_advanced_model(data_path, model_path):
    print("🚀 Loading & Cleaning Data (Production Mode)...")
    df = load_data(data_path)
    df = clean_data(df)

    target = 'Churn'
    X_train, X_test, y_train, y_test = split_data(df, target_column=target)
    
    y_train = y_train.map({'Yes': 1, 'No': 0})
    y_test = y_test.map({'Yes': 1, 'No': 0})

    # --- 1. Define the Preprocessing Pipeline ---
    # Define which columns are which
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['SeniorCitizen', 'gender', 'Partner', 'Dependents', 
                            'InternetService', 'Contract', 'PaymentMethod']

    # Advanced: Using MinMaxScaler for some, StandardScaler for others
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- 2. The Full Pipeline ---
    # Steps: Feature Engineering -> Preprocessing -> XGBoost
    pipeline = Pipeline([
        ('feature_eng', FeatureEngineer()), 
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # --- 3. Hyperparameter Tuning (The "Grid Search") ---
    print("🔧 Tuning Hyperparameters (This will take a minute)...")
    
    param_grid = {
        'classifier__n_estimators': [100, 200],      # How many trees?
        'classifier__learning_rate': [0.01, 0.1],    # How fast to learn?
        'classifier__max_depth': [3, 5],             # How complex the trees?
        'classifier__subsample': [0.8, 1.0]          # Prevent overfitting
    }

    # StratifiedKFold ensures each test batch has enough "Churners"
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='accuracy', 
        verbose=1,
        n_jobs=-1 # Use all CPU cores
    )

    grid_search.fit(X_train, y_train)

    print(f"🏆 Best Params: {grid_search.best_params_}")
    
    # --- 4. Final Evaluation ---
    best_model = grid_search.best_estimator_
    print("🧪 Evaluating on Test Set...")
    
    predictions = best_model.predict(X_test)
    print("\nDetailed Report:\n")
    print(classification_report(y_test, predictions))

    # --- 5. Save ---
    joblib.dump(best_model, model_path)
    print(f"💾 Advanced XGBoost Model saved to {model_path}")

if __name__ == "__main__":
    train_advanced_model("data/data.csv", "models/model_xgb.pkl")