import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(
    page_title="StarPower: Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

# Cache the dataset and model training so it only runs ONCE when the app starts
@st.cache_resource
def train_and_cache_model():
    # Load data directly from your repository
    try:
        df = pd.read_csv("popular_people.csv")
    except FileNotFoundError:
        st.error("Error: 'popular_people.csv' not found in your GitHub repository.")
        st.stop()
        
    df = df.dropna(subset=['gender', 'known_for_department', 'popularity'])

    # Extract unique departments for the dropdown menu
    departments = sorted(df['known_for_department'].dropna().unique().tolist())

    # Features and Target
    X = df[['gender', 'known_for_department']].copy()
    y = df['popularity']

    # Map numeric genders to human-readable text
    X['gender'] = X['gender'].map({0: 'Unknown/Other', 1: 'Female', 2: 'Male'}).fillna('Unknown/Other')

    # Preprocessing Pipeline
    categorical_features = ['gender', 'known_for_department']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)]
    )

    # Create and Train Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    model_pipeline.fit(X, y)
    return model_pipeline, departments

# Run the training sequence (cached dynamically)
with st.spinner("Initializing predictive engine..."):
    model, departments = train_and_cache_model()

# --- UI Interface Layout ---
st.title("🎬 StarPower: Popularity Predictor")
st.markdown("""
This application uses a Machine Learning model trained on industry data to estimate 
a person's **Popularity Score** based on their professional department and gender profile.
""")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    gender_input = st.selectbox(
        "Select Gender Role",
        options=["Female", "Male", "Unknown/Other"]
    )

with col2:
    dept_input = st.selectbox(
        "Select Primary Department",
        options=departments
    )

st.write("---")

if st.button("🔮 Predict Popularity Score", use_container_width=True):
    input_data = pd.DataFrame([{
        'gender': gender_input,
        'known_for_department': dept_input
    }])
    
    prediction = model.predict(input_data)[0]
    
    st.success("### Prediction Complete!")
    st.metric(label="Estimated Popularity Rating", value=f"{prediction:.4f}")
    st.info(f"An individual identifying as **{gender_input}** working primarily in **{dept_input}** holds an expected base popularity score of **{prediction:.2f}** based on dataset trends.")

