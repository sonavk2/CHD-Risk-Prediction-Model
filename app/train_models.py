from typing import Any
from app.preprocess import clean_data
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE


def train_svm(X_train, y_train, kernel='rbf', random_state=42) -> Any:
    svm = SVC(kernel=kernel, random_state=random_state)
    svm.fit(X_train, y_train)
    return svm


def train_logistic_regression(X_train, y_train, random_state=42) -> Any:
    logreg = LogisticRegression(multi_class='multinomial', random_state=random_state, max_iter=1000)
    logreg.fit(X_train, y_train)
    return logreg


def train_decision_tree(X_train, y_train, random_state=42) -> Any:
    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train, y_train)
    return dt


def train_random_forest(X_train, y_train, random_state=42, n_estimators=100) -> Any:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def train_gradient_boosting(X_train, y_train, random_state=42, n_estimators=100) -> Any:
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    gbc.fit(X_train, y_train)
    return gbc


def train_adaboost(X_train, y_train, random_state=42, n_estimators=200) -> Any:
    abc = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        learning_rate=0.1,
        random_state=random_state
    )
    abc.fit(X_train, y_train)
    return abc


def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE to balance the dataset.
    """
    sm = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote
