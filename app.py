import os
import pickle
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Liver Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Load the model
liver_model = pickle.load(open(r'Liver Cirrhosis Stage Detection.pkl', 'rb'))

# Define mappings for categorical fields to numeric values
status_map = {'C': 1, 'CL': 2, 'D': 3}
drug_map = {'Placebo': 1, 'D-penicillamine': 2}
sex_map = {'M': 1, 'F': 0}
ascites_map = {'Y': 1, 'N': 0}
hepatomegaly_map = {'Y': 1, 'N': 0}
spiders_map = {'Y': 1, 'N': 0}
edema_map = {'Y': 1, 'N': 0, 'S': 2}

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Liver Cirrhosis Detection System",
        options=["Home", "Data Upload", "Cirrhosis Stage Prediction", "Model Insights"],
        icons=['house', 'cloud-upload', 'activity', 'bar-chart-line'],
        menu_icon="hospital-fill",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "nav-link": {"margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4CAF50"}
        }
    )

# Home Section
if selected == "Home":
    st.title("Liver Cirrhosis Stage Detection App")
    st.write("This app predicts the stage of liver cirrhosis using patient data. It helps you understand the current state of liver health and offers predictions for medical guidance.")
    st.image("https://londonsono.com/wp-content/uploads/2023/08/stages-of-liver-disease-leading-to-cirrhosis.webp", use_column_width=True)

# Data Upload Section
elif selected == "Data Upload":
    st.title("Upload Your Patient Data")
    st.write("Upload a CSV file containing the patient details for liver cirrhosis prediction.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read and display the uploaded data
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview")
        st.write(data.head())

        st.write("### Data Summary")
        st.write(data.describe())

        st.write("### Columns Expected")
        st.write([
            "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", 
            "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", 
            "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
        ])
        st.write("Ensure that your data follows this format for accurate predictions.")

# Cirrhosis Stage Prediction Section
elif selected == "Cirrhosis Stage Prediction":
    st.title('Liver Cirrhosis Stage Prediction 🩺')
    st.write("Enter patient details below to predict the stage of liver cirrhosis:")

    # Form for user input
    with st.form("liver_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            N_Days = st.number_input('Number of Days in Study', min_value=0, step=1, value=0)
            Status = st.selectbox('Status', ['C', 'CL', 'D'])

        with col2:
            Drug = st.selectbox('Drug Type', ['Placebo', 'D-penicillamine'])
            Age = st.number_input('Age (in Days)', min_value=0, value=0, step=1)
            Sex = st.selectbox('Sex', ['M', 'F'])
            Ascites = st.selectbox('Ascites', ['Y', 'N'])

        with col3:
            Hepatomegaly = st.selectbox('Hepatomegaly', ['Y', 'N'])
            Spiders = st.selectbox('Spiders', ['Y', 'N'])
            Edema = st.selectbox('Edema', ['Y', 'N', 'S'])

        with col1:
            Bilirubin = st.number_input('Bilirubin Level (mg/dL)', min_value=0.0, value=0.0)
            Cholesterol = st.number_input('Cholesterol Level (mg/dL)', min_value=0.0, value=0.0)

        with col2:
            Albumin = st.number_input('Albumin Level (g/dL)', min_value=0.0, value=0.0)
            Copper = st.number_input('Copper Level (µg/dL)', min_value=0.0, value=0.0)

        with col3:
            Alk_Phos = st.number_input('Alkaline Phosphatase Level (IU/L)', min_value=0.0, value=0.0)
            SGOT = st.number_input('SGOT Level (IU/L)', min_value=0.0, value=0.0)

        with col1:
            Tryglicerides = st.number_input('Triglycerides Level (mg/dL)', min_value=0.0, value=0.0)
            Platelets = st.number_input('Platelet Count (×10^3/µL)', min_value=0.0, value=0.0)

        with col2:
            Prothrombin = st.number_input('Prothrombin Time (sec)', min_value=0.0, value=0.0)

        submit_button = st.form_submit_button(label="Predict", type='primary')

    # Prediction logic
    if submit_button:
        with st.spinner('Predicting...'):
            # Convert categorical inputs to numeric using predefined mappings
            user_input = [
                N_Days, 
                status_map[Status], 
                drug_map[Drug], 
                Age, 
                sex_map[Sex], 
                ascites_map[Ascites], 
                hepatomegaly_map[Hepatomegaly], 
                spiders_map[Spiders], 
                edema_map[Edema], 
                Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT,
                Tryglicerides, Platelets, Prothrombin
            ]
            try:
                # Predict the cirrhosis stage
                cirrhosis_prediction = liver_model.predict([user_input])
                st.success(f"**Predicted Cirrhosis Stage: Stage {cirrhosis_prediction[0]}**")

                # Display health tips
                st.markdown("**To manage liver cirrhosis effectively, consider these tips:**")
                st.write("- Follow a liver-friendly diet")
                st.write("- Avoid alcohol and smoking")
                st.write("- Get regular check-ups with a hepatologist")
                st.write("- Monitor symptoms and maintain a healthy lifestyle")

            except Exception as e:
                st.error(f"Error in prediction: {e}")

# Model Insights Section
elif selected == "Model Insights":
    st.title("Model Insights 📊")
    st.write("Understand which features contribute most to the prediction of liver cirrhosis stage.")

    feature_importances = liver_model.feature_importances_
    feature_names = [
        "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", 
        "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", 
        "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
    ]

    # Plotting feature importances
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)
