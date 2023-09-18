# <====================Importing the required libraries==============>
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# <====================Loading the Model========================>
loaded_model = load_model('ANN_Model.h5')

def predict_diabetes(data):
    prediction = loaded_model.predict(data)
    prediction = (prediction >= 0.7).astype(int)
    return prediction

def main():
    st.title("Diabetes Prediction Web app")
    st.sidebar.header("User Input")

    pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, value=80)
    bmi = st.number_input("BMI", min_value=0, value=25)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5, step=0.01)

    if st.button("Predict"):
        # Combine user inputs into a feature array
        user_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function]).reshape(1, -1)

        # Scale the user input
        scaler = MinMaxScaler()
        user_input_scaled = scaler.fit_transform(user_input)

        # Predict
        prediction = predict_diabetes(user_input_scaled)

        # Display the prediction
        if prediction > 0.7:
            st.write("Prediction: Person has diabetes")
        else:
            st.write("Prediction: Person does not have diabetes")




if __name__ == "__main__":
    main()