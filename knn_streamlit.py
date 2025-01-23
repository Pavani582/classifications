import streamlit as st
import pickle
import numpy as np

with open("knn_model.pkl",'rb') as file:
    model = pickle.load(file)
with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

st.title("Vehicle Sales Prediction using KNN")
st.write("Predict whether a user will purchase the vehicle: ")

age = st.number_input("Age: ",min_value=18, max_value=100 , step=1)
salary = st.number_input("Estimated Salary: ",min_value=0, step = 1000)

if st.button("predict"):
    input_data = np.array([[age, salary]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled) 

    if prediction[0] == 1:
        st.success("The user is likely to purchase the vehicle! :)")
    else:
        st.error("The user is unlikely to purchase the vehicle! :(")

        
