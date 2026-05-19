import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="StarPower: Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

# Load the trained model and department list
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('popularity_model.pkl')
    departments = joblib.load('departments.pkl')
    return model, departments

try:
    model, departments = load_model_artifacts()
except FileNotFoundError:
    st.error("Model files not found! Make sure popularity_model.pkl and departments.pkl are in the same folder.")
    st.stop()

# Title and App description
st.title("🎬 StarPower: Popularity Predictor")
st.markdown("""
This application uses a Machine Learning model trained on industry data to estimate 
a person's **Popularity Score** based on their professional department and gender profile.
""")

st.write("---")

# Layout columns for inputs
col1, col2 = st.columns(2)

with col1:
    gender_input = st.selectbox(
        "Select Gender Role",
        options=["Female", "Male", "Unknown/Other"]
    )

with col2:
    dept_input = st.selectbox(
        "Select Primary Department",
        options=sorted(departments)
    )

st.write("---")

# Prediction Trigger
if st.button("🔮 Predict Popularity Score", use_container_width=True):
    input_data = pd.DataFrame([{
        'gender': gender_input,
        'known_for_department': dept_input
    }])
    
    prediction = model.predict(input_data)[0]
    
    st.success("### Prediction Complete!")
    st.metric(label="Estimated Popularity Rating", value=f"{prediction:.4f}")
    st.info(f"An individual identifying as **{gender_input}** working primarily in **{dept_input}** holds an expected base popularity score of **{prediction:.2f}** based on dataset trends.")
