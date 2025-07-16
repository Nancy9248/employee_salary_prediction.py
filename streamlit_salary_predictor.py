import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
data = pd.read_csv("adult 3.csv")
data = data.replace("?", pd.NA).dropna()
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include="object"):
    data[col] = data[col].str.strip()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include="object"):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop("income", axis=1)
y = data["income"]

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Streamlit UI
st.title(" Employee Salary Prediction")

st.write("Enter the details below to predict if income is <=50K or >50K.")

def user_input_features():
    age = st.slider("Age", 17, 90, 30)
    education_num = st.slider("Education Num (1-16)", 1, 16, 10)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)

    workclass = st.selectbox("Workclass", label_encoders["workclass"].classes_)
    marital_status = st.selectbox("Marital Status", label_encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
    relationship = st.selectbox("Relationship", label_encoders["relationship"].classes_)
    race = st.selectbox("Race", label_encoders["race"].classes_)
    sex = st.selectbox("Sex", label_encoders["sex"].classes_)
    native_country = st.selectbox("Native Country", label_encoders["native-country"].classes_)

    workclass_enc = label_encoders["workclass"].transform([workclass])[0]
    marital_enc = label_encoders["marital-status"].transform([marital_status])[0]
    occupation_enc = label_encoders["occupation"].transform([occupation])[0]
    relationship_enc = label_encoders["relationship"].transform([relationship])[0]
    race_enc = label_encoders["race"].transform([race])[0]
    sex_enc = label_encoders["sex"].transform([sex])[0]
    native_enc = label_encoders["native-country"].transform([native_country])[0]

    data_dict = {
        "age": age,
        "workclass": workclass_enc,
        "fnlwgt": 0,  # placeholder
        "education": 0,  # placeholder
        "education-num": education_num,
        "marital-status": marital_enc,
        "occupation": occupation_enc,
        "relationship": relationship_enc,
        "race": race_enc,
        "sex": sex_enc,
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": hours_per_week,
        "native-country": native_enc,
    }
    return pd.DataFrame([data_dict])

input_df = user_input_features()

# Make prediction
prediction = clf.predict(input_df)
prediction_label = label_encoders["income"].inverse_transform(prediction)[0]

st.subheader("Prediction:")
st.write(f"**Income:** {prediction_label}")
