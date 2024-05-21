import numpy as np
import pandas as pd
import joblib
def categorize_age(age):
  age_category= pd.cut([age], bins=[18, 30, 40, 50, 60, 101], labels=['18-29', '30-39', '40-49', '50-59', '60-100'],right=False, include_lowest=True)
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
def classify_glucose_level(glucose_level):
    if glucose_level=='Normal' or glucose_level=='Below Normal' :
        return 1  # Normal
    elif glucose_level=='Above Normal':
        return 2  # Above Normal
    elif glucose_level=='Well Above Normal':
        return 3  # Well Above Normal
def classify_chol_level(cholesterol):
    if cholesterol=='Normal' :
        return 1  # Normal
    elif cholesterol=='Above Normal':
        return 2  # Above Normal
    elif cholesterol=='Well Above Normal':
        return 3  # Well Above Normal
def classify_smoke_level(smoke):
    if smoke=='formerly smoked' or 'smokes' :
        return 1
    else:
      return 0
def classify_active(active):
  if active>=30:
    return 1
  else:
    return 0
def create_dataframe(gender, weight, height,cholesterol,
                     systolic, diastolic, physical_activity_minutes,
                     gluc, smoking_status, age, alco):

    # Create a dictionary with the data
    data = {
        'gender': [2 if gender else 1],
        'age': [age],
        'height':[height],
        'weight':[weight],
        'ap_hi':[systolic],
        'ap_lo':[diastolic],
        'BMI_Category':[add_bmi_category(height,weight)],
        'BP_Category':[categorize_blood_pressure(systolic,diastolic)],
        'age_category':[categorize_age(age)],
        'cholesterol':[classify_chol_level(cholesterol)],
        'gluc':[classify_glucose_level(gluc)],
        'smoke':[classify_smoke_level(smoking_status)],
        'alco':[1 if alco else 0],
        'active':[classify_active(physical_activity_minutes)],
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    return df
def encoded_scaled(df):
    df['gender']=df['gender'].astype('object')
    df['cholesterol']=df['cholesterol'].astype('object')
    df['gluc']=df['gluc'].astype('object')
    df['smoke']=df['smoke'].astype('object')
    df['alco']=df['alco'].astype('object')
    df['active']=df['active'].astype('object')
    df['age_category']=df['age_category'].astype('category')
    df['BMI_Category']=df['BMI_Category'].astype('object')
    df['BP_Category']=df['BP_Category'].astype('object')
    encoder = joblib.load('cardioencoder1.pkl')
    categorical_columns = ['gender','alco','smoke','active']
    one_hot_encoded = encoder.transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    data_encoded = pd.concat([df, one_hot_df], axis=1)
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    cols = ['age', 'height', 'weight','ap_lo','ap_hi']

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
    #data_encoded.drop(['Age','Systolic','Diastolic'],axis=1,inplace=True)
    return data_encoded
def custom_label_encoder(df):
    mappings = {
        'BMI_Category': ['Normal', 'Obese', 'Overweight', 'Underweight'],
        'BP_Category': ['Unknown', 'Normal', 'Elevated', 'High Blood Pressure (Hypertension) Stage 1', 'High Blood Pressure (Hypertension) Stage 2'],
        'age_category': ['18-29', '30-39', '40-49', '50-59', '60-100']
    }

    for column, order in mappings.items():
        mapping = {category: idx for idx, category in enumerate(order)}
        df[column + ' Code'] = df[column].map(mapping)
    df.drop(['BMI_Category','BP_Category','age_category'],axis=1,inplace=True)
    return df
def converter(df):
   df['cholesterol'] = df['cholesterol'].astype('category')
   df['gluc'] = df['gluc'].astype('category')
   df['age_category Code'] = df['age_category Code'].astype('category')
   return df
def model(df):
  model = joblib.load('cardiomodel.pkl')
  predictions = model.predict(df)
  return predictions
def message(predictions):
  if predictions.item()==1:
    message="You are experiencing symptoms of Cardio Vascular Disease, please visit the nearest hospital or call emergency services immediately"
  else:
    message="All Good!"
  return message