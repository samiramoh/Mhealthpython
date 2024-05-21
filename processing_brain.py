import pandas as pd
import numpy as np
import joblib
def add_bmi_category( height, weight):
  BMI =weight / (height / 100) ** 2

    # Define BMI categories according to common thresholds
  conditions = [
        (BMI< 18.5),
        (BMI>= 18.5) & (BMI< 25),
        (BMI>= 25) & (BMI< 30),
        (BMI>= 30)
    ]
  categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese']

    # Use numpy.select to apply the conditions to categories
  bmi_cat = np.select(conditions, categories, default='Unknown')

  return bmi_cat.item()
def categorize_age(age):
  age_category= pd.cut([age], bins=[18, 30, 40, 50,60,70,80,101], labels=['18-29', '30-39', '40-49', '50-59', '60-69','70-79','80-100'],right=False, include_lowest=True)

  return age_category[0]
def categorize_blood_pressure(sys_value, dia_value):
    conditions = [
        (sys_value < 120) & (dia_value < 80),
        (sys_value >= 120) & (sys_value < 130) & (dia_value < 80),
        ((sys_value >= 130) & (sys_value < 140)) | ((dia_value >= 80) & (dia_value < 90)),
        (sys_value >= 140) | (dia_value >= 90),
        (sys_value > 180) | (dia_value > 120)
    ]

    categories = [
        'Normal',
        'Elevated',
        'High Blood Pressure (Hypertension) Stage 1',
        'High Blood Pressure (Hypertension) Stage 2',
        'Hypertensive Crisis'
    ]

    # Initialize hypertension flag
    hypertension_flag = 0

    # Set hypertension flag if blood pressure falls into stage 1 or 2 categories
    if conditions[2] or conditions[3]:
        hypertension_flag = 1

    # Categorize blood pressure
    bp_category = np.select(conditions, categories, default='Unknown')

    return hypertension_flag

def classify_glucose_level(glucose_level):
    if glucose_level=='Normal':
        return 1  # Normal
    elif glucose_level=='Above Normal':
        return 2  # Above Normal
    elif glucose_level=='Well Above Normal':
        return 3  # Well Above Normal
    else:
        return 0  # Value below normal range
def create_dataframe(weight, height,
                     systolic, diastolic, smoking_status,
                     heart_disease, Glucose, age, gender):

    # Create a dictionary with the data
    data = {
        'gender': ['Male' if gender else 'Female'],
        'BMI_Category': [add_bmi_category(height,weight)],
        'age_category': [categorize_age(age)],
        'hypertension':[categorize_blood_pressure(systolic,diastolic)],
        'heart_disease': [1 if heart_disease else 0],
        'glucose_level':[classify_glucose_level(Glucose)],
        'smoking_status':[smoking_status]
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    return df
def encoded_scaled(df):
    encoder = joblib.load('brainstrokeEncoder1.pkl')
    categorical_columns = ['BMI_Category','age_category','gender','smoking_status']
    one_hot_encoded = encoder.transform(df[categorical_columns].astype('category'))
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    data_encoded = pd.concat([df, one_hot_df], axis=1)
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    return data_encoded
def model(df):
  model = joblib.load('brainstrokemodel1.pkl')
  predictions = model.predict(df)
  return predictions
def message(predictions):
  if predictions.item()==1:
    message="You are experiencing symptoms of Brain Stroke, please visit the nearest hospital or call emergency services immediately"
  else:
    message="All Good!"
  return message