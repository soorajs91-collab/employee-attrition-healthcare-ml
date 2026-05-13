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

    input_dict = {
    'Age': age,
    'DailyRate': 0,
    'DistanceFromHome': distance_from_home,
    'Education': 0,
    'EmployeeCount': 1,
    'EnvironmentSatisfaction': 0,
    'HourlyRate': 0,
    'JobInvolvement': 0,
    'JobLevel': 0,
    'JobSatisfaction': 0,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': 0,
    'NumCompaniesWorked': 0,
    'PercentSalaryHike': 0,
    'PerformanceRating': 0,
    'RelationshipSatisfaction': 0,
    'StandardHours': 80,
    'Shift': 0,
    'TotalWorkingYears': 0,
    'TrainingTimesLastYear': 0,
    'WorkLifeBalance': 0,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': 0,
    'YearsSinceLastPromotion': 0,
    'YearsWithCurrManager': 0,

    # Encoded categorical columns
    'BusinessTravel_Travel_Frequently': 0,
    'BusinessTravel_Travel_Rarely': 0,
    'Department_Maternity': 0,
    'Department_Neurology': 0,
    'EducationField_Life Sciences': 0,
    'EducationField_Marketing': 0,
    'EducationField_Medical': 0,
    'EducationField_Other': 0,
    'EducationField_Technical Degree': 0,
    'Gender_Male': 0,
    'JobRole_Administrative': 0,
    'JobRole_Nurse': 0,
    'JobRole_Other': 0,
    'JobRole_Therapist': 0,
    'MaritalStatus_Married': 0,
    'MaritalStatus_Single': 0,
    'OverTime_Yes': overtime_yes
    }

    input_data = pd.DataFrame([input_dict])

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