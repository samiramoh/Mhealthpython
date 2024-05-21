import numpy as np
import pandas as pd
import joblib
def process_sleep_duration(sleep_duration):
    # Split the input string by colon
    parts = sleep_duration.split(':')

    # Convert the second part (minutes) to a float and divide by 60
    minutes_as_hours = (int(parts[1]) / 60)*10
    hours=int(parts[0])
    # Return the modified parts
    return float(f'{hours}.{int(minutes_as_hours)}')
def add_bmi_category( height, weight):
  BMI =weight / (height / 100) ** 2

    # Define BMI categories according to common thresholds
  conditions = [
        (BMI< 18.5),
        (BMI>= 18.5) & (BMI< 25),
        (BMI>= 25) & (BMI< 30),
        (BMI>= 30)
    ]
  categories = ['Underweight', 'Normal', 'Overweight', 'Obese']

    # Use numpy.select to apply the conditions to categories
  bmi_cat = np.select(conditions, categories, default='Unknown')

  return bmi_cat.item()
def categorize_age(age):
  age_category= pd.cut([age], bins=[18, 30, 40, 50, 101], labels=['18-29', '30-39', '40-49', '50-100'],right=False, include_lowest=True)

  return age_category[0]
def categorize_blood_pressure(sys_column, dia_column):
  conditions = [
        (sys_column < 120) & (dia_column < 80),
        (sys_column >= 120) & (sys_column < 130) & (dia_column < 80),
        ((sys_column >= 130) & (sys_column < 140)) | ((dia_column >= 80) & (dia_column < 89)),
        (sys_column >= 140) | (dia_column >= 90),
        (sys_column > 180) | (dia_column > 120)
    ]

  categories = [
        'Normal',
        'Elevated',
        'High Blood Pressure (Hypertension) Stage 1',
        'High Blood Pressure (Hypertension) Stage 2',
        'Hypertensive Crisis'
    ]

  BP_Category = np.select(conditions, categories, default='Unknown')

  return BP_Category.item()
def create_dataframe(user_stress_level, weight, height, heart_rate,
                     systolic, diastolic, physical_activity_minutes,
                     steps, sleep_duration, age, gender):

    # Create a dictionary with the data
    data = {
        'Gender': ['Male' if gender else 'Female'],
        'Age': [age],
        'Sleep Duration': [process_sleep_duration(sleep_duration)],
        'Physical Activity': [physical_activity_minutes],
        'Stress Level': [user_stress_level],
        'BMI Category': [add_bmi_category(height,weight)],
        'age_category': [categorize_age(age)],
        'BP_Category':[categorize_blood_pressure(systolic,diastolic)],
        'Systolic': [systolic],
        'Diastolic':[diastolic],
        'Heart Rate': [heart_rate],
        'Daily Steps': [steps]
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    return df


def encoded_scaled(df):
    encoder = joblib.load('encoder.pkl')
    categorical_columns = ['Gender','BMI Category','BP_Category','age_category']
    one_hot_encoded = encoder.transform(df[categorical_columns].astype('category'))
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    data_encoded = pd.concat([df, one_hot_df], axis=1)
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    cols = ['Sleep Duration', 'Stress Level', 'Physical Activity', 'Heart Rate', 'Daily Steps']

    # DataFrame to hold the transformed new data
    transformed_new_data = pd.DataFrame()

    for c in cols:
        # Load the scaler for each column
        scaler = joblib.load(f'{c}_scaler.pkl')

        # Reshape the new data column to (-1, 1) because .transform() expects 2D inputs
        column_data = data_encoded[c].values.reshape(-1, 1)

        # Transform the new data using the loaded scaler
        data_encoded[c] = scaler.transform(column_data)
        # Add the transformed data back to the DataFrame
        #transformed_new_data[c] = transformed_data.flatten(
    data_encoded.drop(['Age','Systolic','Diastolic'],axis=1,inplace=True)
    return data_encoded
def model(df):
  model = joblib.load('sleepdisordermodel (2).pkl')
  predictions = model.predict(df)
  return predictions
def message(predictions):
  if predictions.item()==0:
    message="You are experiencing symptoms of Insomnia, please visit the nearest hospital or call emergency services immediately"
  elif predictions.item()==2:
    message="You are experiencing symptoms of Sleep Apnea, please visit the nearest hospital or call emergency services immediately"
  else:
    message="All Good!"
  return message