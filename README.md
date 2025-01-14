# Predicting 10 year risk of Coronary Heart Disease

## Objective
The main objective of this analysis is to develop classifier models focused on prediction and compare them to see which model is best. The analysis aims to provide accurate predictions of the 10-year risk of future coronary heart disease(CHD) for patients
based on their demographic, behavioral and medical attributes. The business or stakeholders of this data, such as healthcare providers or researchers, can benefit from the analysis by identifying individuals at high risk of CHD and implementing preventive 
measures.

## Features

- Data cleaning and preprocessing.
- Implementation of multiple machine learning models:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Decision Trees
  - Gradient Boosting
  - AdaBoost
- Handling imbalanced datasets using SMOTE.
- Hyperparameter tuning using GridSearchCV.
- Feature engineering and interaction terms.
- Modularized codebase for better maintainability and scalability.

## Directory Structure

```
CHD-Risk-Prediction-Model/
│
├── data_cardiovascular_risk.csv   # Dataset file
├── main.py                        # Entry point for the project
├── preprocess.py                  # Data loading and preprocessing functions
├── train_models.py                # Functions for training various models
├── evaluate_models.py             # Model evaluation functions
├── requirements.txt               # Python dependencies
├── model.pkl                      # Saved model file
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
```

## Data
The chosen dataset is from Kaggle(https://www.kaggle.com/datasets/mamta1999/cardiovascular-risk-data), which includes information on over 4,000 patients and 15 attributes representing potential risk factors for CHD.

**Features**

1. Demographic:
- Sex: male or female ("M" or "F")
- Age: Age of the patient (Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
- Education: The level of education of the patient (categorical values - 1,2,3,4)

2. Behavioral:
- is_smoking: whether or not the patient is a current smoker ("YES" or "NO")
- Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)

3. Medical (history):
- BP Meds: whether or not the patient was on blood pressure medication (Nominal)
- Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
- Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
- Diabetes: whether or not the patient had diabetes (Nominal)

4. Medical (current):
- Tot Chol: total cholesterol level (Continuous)
- Sys BP: systolic blood pressure (Continuous)
- Dia BP: diastolic blood pressure (Continuous)
- BMI: Body Mass Index (Continuous)
- Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
- Glucose: glucose level (Continuous)

**Target**

10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”)

## Results

- The best-performing model (Random Forest) achieved:
  - **Accuracy**: 81.02%
  - **F1-score**: 78.34%
  - **Precision**: 76.50%
  - **Recall**: 81.02%

Model deployed as a REST API using Flask.
