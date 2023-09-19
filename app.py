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
    prediction = prediction[0,0]
    return prediction

#<======================= Custom CSS & HTML style for the header==========================>
# Load CSS file
with open("styles.css") as f:
    css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load HTML file
with open("Header.html") as f:
    html = f.read()
    st.markdown(html, unsafe_allow_html=True)

# <======================================== Sidebar for all details ==============================>
st.sidebar.header("Diabetes prediction web application🩺")
with st.sidebar:
    st.markdown("This Web Application is built by Arya Chakraborty")
    st.markdown("I would like to acknowledge Exposys Data Labs for providing me with the opportunity to work on this project as part of the Data Science internship program. The experience gained during this internship has been invaluable in honing my skills and understanding in the field of data science. I am grateful for the mentorship and support provided by the Exposys Data Labs team throughout the duration of this project.")
    st.markdown("[My GitHub](https://github.com/Arya920)")
    st.markdown("[My LinkedIn](https://www.linkedin.com/in/arya-chakraborty-95a8411b2/)")
    st.markdown("[My Portfolio](https://arya920.github.io/My_Portfolio/)")

    pass

pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=0)
glucose = st.number_input("Glucose Level", min_value=0, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin Level", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0, value=25)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
Age_Brack = st.number_input("Skin Thickness", min_value=1, value=8)

if st.button("Predict"):
    # Combine user inputs into a feature array
    user_input = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, Age_Brack]]

    # Scale the user input
    scaler = MinMaxScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Predict
    prediction = predict_diabetes(user_input_scaled)

    # Display the prediction
    if prediction == 0:
        st.write("Prediction: Person has diabetes")
    else:
        st.write("Prediction: Person does not have diabetes")

