import streamlit as st
import pickle
import numpy as np

with open('logistic_regression.pkl','rb') as file:
    model = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title("Vehicle Sales Prediction using Logistic Regression")
st.write("Predict whether a user will purchase vehicle: ")

age = st.number_input("Enter Age", min_value = 18, max_value = 100, step = 1)
estimated_salary = st.number_input("Enter Estimated Salary: ",min_value = 0, step = 1000)

if st.button("Predict"):
    input_data = np.array([[age, estimated_salary]])
    input_data_scaled = scaler.transform(input_data)

    predict = model.predict(input_data_scaled)

    if predict[0] == 1:
        st.success(f"The user is likely to purchase the vehicle.")
    else:
        st.error(f"The user is unlikely to purchase the vehicle.")