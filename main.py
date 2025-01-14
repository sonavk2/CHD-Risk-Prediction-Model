import os
import joblib
import pandas as pd
from app.evaluate_models import evaluate_model

from app.preprocess import load_data, clean_data, split_data, engineer_features, scale_data
from app.train_models import (
    train_svm,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_gradient_boosting,
    train_adaboost,
    apply_smote
)


MODEL_PATH = "model.pkl"



def main():
    
    df = load_data("data_cardiovascular_risk.csv")
    df = clean_data(df)
    
    X, y = engineer_features(df, target_col="TenYearCHD")
    
    X_train, X_test, y_train, y_test = split_data(df, target_col="TenYearCHD", test_size=0.3, random_state=42)
    
    
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    
    
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Found existing model file: {MODEL_PATH}. Loading...")
        model = joblib.load(MODEL_PATH)
    else:
        print(f"[WARNING] No pre-trained model found at {MODEL_PATH}. Training a new model...")
        
        
        model = train_random_forest(X_train_smote, y_train_smote, random_state=42, n_estimators=100)
        
        
        joblib.dump(model, MODEL_PATH)
        print(f"[INFO] New model trained and saved to {MODEL_PATH}.")
    
    
    
    result_df = evaluate_model(model, X_test, y_test, model_name="RandomForest")
    
    print("\n=== Final Results ===")
    print(result_df)

    
if __name__ == "__main__":
    main()
