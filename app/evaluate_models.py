# evaluate_models.py

import pandas as pd
from typing import Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model: Any, X_test, y_test, model_name: str = "Model") -> pd.DataFrame:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"Precision:{precision:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    results = {
        'model': model_name,
        'accuracy': acc,
        'f1_score': f1,
        'recall': recall,
        'precision': precision
    }
    return pd.DataFrame([results])
