import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("data_cardiovascular_risk.csv")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})
    df['is_smoking'] = df['is_smoking'].map({'YES': 1, 'NO': 0})
    df = df.fillna(0)
    
    return df

def engineer_features(df: pd.DataFrame, target_col: str = "TenYearCHD") -> pd.DataFrame:
    df['BMI'] = df['BMI'] * df['age']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]

    return pd.DataFrame(X_selected, columns=selected_features), y

def split_data(df: pd.DataFrame, target_col: str = "TenYearCHD", test_size: float = 0.3, random_state: int = 42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Scale data using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
