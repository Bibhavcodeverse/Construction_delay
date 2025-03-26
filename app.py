import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load label encoders
with open("label_encoders.pkl", "rb") as file:
    label_classes = pickle.load(file)  # Load only class mappings

# Streamlit UI
st.title("🚧 Construction Delay Prediction")
st.write("Enter project details to predict the expected delay.")

# User input fields
col1, col2 = st.columns(2)
with col1:
    project_size = st.selectbox("Project Size", label_classes["Project_Size"])
    weather_conditions = st.selectbox("Weather Conditions", label_classes["Weather_Conditions"])
    labor_availability = st.selectbox("Labor Availability", label_classes["Labor_Availability"])
    material_availability = st.selectbox("Material Availability", label_classes["Material_Availability"])
with col2:
    site_conditions = st.selectbox("Site Conditions", label_classes["Site_Conditions"])
    government_approvals = st.selectbox("Government Approvals", label_classes["Government_Approvals"])
    budget_issues = st.selectbox("Budget Issues", label_classes["Budget_Issues"])
    equipment_availability = st.selectbox("Equipment Availability", label_classes["Equipment_Availability"])
    planned_duration = st.number_input("Planned Duration (Days)", min_value=1)

# Encode categorical inputs
input_data = [
    np.where(label_classes["Project_Size"] == project_size)[0][0],
    np.where(label_classes["Weather_Conditions"] == weather_conditions)[0][0],
    np.where(label_classes["Labor_Availability"] == labor_availability)[0][0],
    np.where(label_classes["Material_Availability"] == material_availability)[0][0],
    np.where(label_classes["Site_Conditions"] == site_conditions)[0][0],
    np.where(label_classes["Government_Approvals"] == government_approvals)[0][0],
    np.where(label_classes["Budget_Issues"] == budget_issues)[0][0],
    np.where(label_classes["Equipment_Availability"] == equipment_availability)[0][0],

    planned_duration
]

# Predict delay
if st.button("🔍 Predict Delay"):
    prediction = model.predict([input_data])
    st.success(f"Predicted Completion Time: {prediction[0]:.2f} days")
