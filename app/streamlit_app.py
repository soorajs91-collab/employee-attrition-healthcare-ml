import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/best_model.pkl')

st.title("Employee Attrition Prediction")

st.write(
    "Predict employee attrition risk using Machine Learning."
)

# User inputs
age = st.number_input("Age", 18, 65, 30)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    50000,
    5000
)

years_at_company = st.number_input(
    "Years At Company",
    0,
    40,
    5
)

distance_from_home = st.number_input(
    "Distance From Home",
    1,
    50,
    10
)

# Overtime
overtime = st.selectbox(
    "Overtime",
    ["Yes", "No"]
)

overtime_yes = 1 if overtime == "Yes" else 0

# Prediction button
if st.button("Predict Attrition"):

    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'YearsAtCompany': [years_at_company],
        'DistanceFromHome': [distance_from_home],
        'OverTime_Yes': [overtime_yes]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(
            f"Employee likely to leave. Risk Score: {probability:.2f}"
        )
    else:
        st.success(
            f"Employee likely to stay. Risk Score: {probability:.2f}"
        )