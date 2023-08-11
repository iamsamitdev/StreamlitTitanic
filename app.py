# import libraries
import streamlit as st
from joblib import load

# load the model from disk
model = load('titatic_survival.joblib')

# Create Streamlit Web App
st.title('Titanic Survival Prediction')

# Sidebar with menu options
st.sidebar.title('Menu')

# Menu options
menu = ['Home', 'Prediction']
st.sidebar.selectbox('', menu)

# Input components
age = st.slider('Age', 0.42, 80.0, 30.0)
sibsp = st.slider('SibSp', 0, 8, 0)
parch = st.slider('Parch', 0, 9, 0)
fare = st.slider('Fare', 0.0, 512.30, 32.20)

# Add Predict button
predict_button = st.button('Predict')

# Prediction logic
if predict_button:
    input_data = [[age, sibsp, parch, fare]]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display prediction
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.write('Survived')
    else:
        st.write('Did not survive')

    # Display prediction probability
    st.subheader('Prediction Probability:')
    st.write(f'Survived: {prediction_proba[0][1]:.2f}')
    st.write(f'Did not survive: {prediction_proba[0][0]:.2f}')
