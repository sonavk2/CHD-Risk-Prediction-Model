# Predicting 10 year risk of Coronary Heart Disease

### Install

This project required **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

## Objective
The main objective of this analysis is to develop classifier models focused on prediction and compare them to see which model is best. The analysis aims to provide accurate predictions of the 10-year risk of future coronary heart disease(CHD) for patients
based on their demographic, behavioral and medical attributes. The business or stakeholders of this data, such as healthcare providers or researchers, can benefit from the analysis by identifying individuals at high risk of CHD and implementing preventive 
measures.

## Data
The chosen dataset is from Kaggle(https://www.kaggle.com/datasets/mamta1999/cardiovascular-risk-data), which includes information on over 4,000 patients and 15 attributes representing potential risk factors for CHD.

**Features**
1.Demographic:
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

## Contents
1. Data Cleaning and preprocessing steps
2. EDA
3. Models Trained and Evaluated:
- SVM
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boost
- AdaBoost
