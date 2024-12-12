import pickle
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from fpdf import FPDF
from streamlit_chat import message 
import requests
from geopy.geocoders import Nominatim


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

# Function to clean the report text to avoid special characters that cause encoding issues
def clean_report_text(text):
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace('•', '-')
    text = text.replace('…', '...')
    text = text.replace('©', '(C)')
    return text

# Function to generate the report based on the selected stage
def generate_report(stage):
    report = ""
    if stage == 1:
        report = """
        ## Stage 1: Compensated Cirrhosis
        **Description**:
        - Compensated cirrhosis is the early stage of liver cirrhosis where the liver still functions well despite some damage. At this stage, the liver has the ability to compensate for the damage.
        
        **Symptoms**:
        - Fatigue
        - Mild abdominal pain or discomfort
        - Slight enlargement of the liver (hepatomegaly)
        
        **Diagnosis**:
        - Liver function tests show minor alterations.
        - Imaging may show early scarring.
        
        **Affects**:
        - Minimal liver damage.
        - The liver can still perform its essential functions, like detoxifying blood and producing bile.
        - The patient may not experience any noticeable symptoms.

        **Precautions**:
        - Regular liver function monitoring.
        - Avoid alcohol, drugs, and other substances that could harm the liver.
        - Maintain a balanced, nutritious diet.
        - Weight management is crucial to prevent additional stress on the liver.
        
        **Lifestyle Recommendations**:
        - Exercise regularly to help maintain a healthy weight and prevent fat buildup in the liver.
        - Vaccinations for hepatitis A and B are essential, as the liver is more vulnerable at this stage.
        - Limit alcohol intake to minimize further damage.
        
        **Medical Interventions**:
        - No immediate need for medical treatment in this stage.
        - Regular follow-ups with a hepatologist (liver specialist) to monitor liver health.

        **Risk of Progression**:
        - If lifestyle changes are not implemented, or if alcohol consumption continues, this stage can progress to decompensated cirrhosis (Stage 2).
        """
        
    elif stage == 2:
        report = """
        ## Stage 2: Decompensated Cirrhosis
        **Description**:
        - Decompensated cirrhosis occurs when the liver starts to lose its ability to function properly. Complications such as fluid retention (ascites) and jaundice appear, and the liver may no longer be able to compensate for the damage.
        
        **Symptoms**:
        - Jaundice (yellowing of the skin and eyes)
        - Ascites (fluid buildup in the abdomen)
        - Swelling in the legs (edema)
        - Fatigue and weakness
        - Loss of appetite
        
        **Diagnosis**:
        - Blood tests reveal abnormal liver function, elevated bilirubin, and low albumin.
        - Imaging studies show more severe liver damage with the presence of ascites.
        
        **Affects**:
        - Liver function begins to decline significantly, and complications such as fluid retention become more apparent.
        - The liver struggles to detoxify the body, resulting in a buildup of toxins in the blood.
        - A risk of bleeding due to low platelets and impaired clotting.

        **Precautions**:
        - Strict monitoring of liver function through blood tests and imaging.
        - Dietary modifications to manage ascites and other symptoms.
        - Diuretics may be prescribed to help manage fluid retention.
        
        **Lifestyle Recommendations**:
        - Avoid alcohol completely to reduce the risk of further liver damage.
        - Regular physical activity can help manage fluid buildup.
        - A low-salt diet to help control ascites and reduce fluid retention.
        
        **Medical Interventions**:
        - Medications such as diuretics to reduce ascites.
        - Endoscopic treatment for variceal bleeding (if bleeding occurs).
        - Possible consideration for liver transplantation evaluation if the condition worsens.
        
        **Risk of Progression**:
        - If the liver’s function continues to decline and complications worsen, the patient could progress to end-stage cirrhosis (Stage 3).
        """
        
    elif stage == 3:
        report = """
        ## Stage 3: End-Stage Cirrhosis
        **Description**:
        - End-stage cirrhosis is the final stage of liver cirrhosis, where the liver is severely damaged and no longer capable of performing its vital functions. At this stage, the liver's inability to function properly leads to life-threatening complications.
        
        **Symptoms**:
        - Severe jaundice and skin discoloration
        - Ascites and edema (swelling)
        - Severe fatigue and weakness
        - Confusion or memory problems due to hepatic encephalopathy
        - Bleeding due to impaired blood clotting
        
        **Diagnosis**:
        - Advanced liver failure as indicated by liver function tests.
        - Imaging shows extensive scarring, large amounts of ascites, and possible liver cancer (hepatocellular carcinoma).
        
        **Affects**:
        - Complete liver failure with a severe loss of liver function.
        - Hepatic encephalopathy (confusion and cognitive issues) may develop due to toxins building up in the blood.
        - Risk of spontaneous bleeding and infection due to clotting issues and immune suppression.
        
        **Precautions**:
        - Strict medical supervision is essential.
        - Consideration for liver transplant evaluation due to the severe loss of liver function.
        - Treatment of complications like infections, bleeding, and ascites.
        
        **Lifestyle Recommendations**:
        - At this stage, the primary focus is on managing complications and maintaining quality of life.
        - Nutritional support to improve overall health and strengthen the immune system.
        - Avoid alcohol and any substances that further stress the liver.
        
        **Medical Interventions**:
        - Liver transplant evaluation is often required as the liver’s ability to regenerate is no longer present.
        - Medications to manage hepatic encephalopathy and reduce complications.
        - Frequent hospital visits to manage ascites, variceal bleeding, and infections.
        
        **Risk of Progression**:
        - Without a liver transplant or effective management of complications, the patient may face life-threatening outcomes within a short period.
        """
    else:
        report = "Invalid stage selection."

    return report


# Function to save the report as PDF
def save_pdf(report, stage):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Liver Cirrhosis Stage {stage} Report", ln=True, align="C")

    # Clean the report text
    report = clean_report_text(report)

    # Add report content
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    try:
        # Encoding text to avoid unicode errors
        report = report.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, txt=report)
    except Exception as e:
        st.error(f"Error while encoding report text: {e}")
        return None

    # Save PDF to disk
    pdf_output_path = f"liver_cirrhosis_stage_{stage}_report.pdf"
    
    # Print the file path for debugging
    print(f"PDF saved at: {pdf_output_path}")
    
    pdf.output(pdf_output_path)
    return pdf_output_path

# Dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Toggle Button
st.sidebar.checkbox(
    "Dark Mode", 
    value=st.session_state.dark_mode, 
    on_change=lambda: st.session_state.update(dark_mode=not st.session_state.dark_mode)
)

# Styles based on mode
if st.session_state.dark_mode:
    bg_color = "#121212"  # Dark background
    text_color = "#FFFFFF"  # Light text
    hover_color = "#333333"
    selected_color = "#006400"  # Dark green for selected
else:
    bg_color = "#f0f2f6"  # Light background
    text_color = "#000000"  # Dark text
    hover_color = "#eee"
    selected_color = "#4CAF50"  # Light green for selected

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Liver Cirrhosis Detection System",
        options=["Home", "Data Upload", "Cirrhosis Stage Prediction", "Model Insights", "Report Generation", "Chatbot", "Hospital Finder"],
        icons=['house', 'cloud-upload', 'activity', 'bar-chart-line', 'file-earmark-text', 'chat-dots', 'hospital'],
        menu_icon="hospital-fill",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": bg_color},
            "nav-link": {"margin": "0px", "--hover-color": hover_color, "color": text_color},
            "nav-link-selected": {"background-color": selected_color, "color": text_color},
        }
    )


# Home Section
if selected == "Home":
    st.title("Liver Cirrhosis Stage Detection App")
    st.write("This app predicts the stage of liver cirrhosis using patient data. It helps you understand the current state of liver health and offers predictions for medical guidance.")
    
    # Add Image for better visualization
    st.image("https://londonsono.com/wp-content/uploads/2023/08/stages-of-liver-disease-leading-to-cirrhosis.webp", use_column_width=True)

    # Information Section
    st.markdown("""**Learn More About Liver Cirrhosis:** Liver cirrhosis is a serious liver condition that can lead to permanent liver damage. The stages of cirrhosis depend on how much liver damage has occurred. It is important to diagnose it early for better management.""")

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
            Alk_Phos = st.number_input('Alkaline Phosphatase Level (U/L)', min_value=0, value=0)
            SGOT = st.number_input('SGOT Level (U/L)', min_value=0, value=0)
            Tryglicerides = st.number_input('Triglycerides Level (mg/dL)', min_value=0, value=0)

        Platelets = st.number_input('Platelet Count (x10^3/µL)', min_value=0, value=0)
        Prothrombin = st.number_input('Prothrombin Time (seconds)', min_value=0.0, value=0.0)

        submit_button = st.form_submit_button("Predict Stage")

        if submit_button:
            # Prepare user input for prediction
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
                Bilirubin,
                Cholesterol,
                Albumin,
                Copper,
                Alk_Phos,
                SGOT,
                Tryglicerides,
                Platelets,
                Prothrombin
            ]

            # Make prediction
            stage_prediction = liver_model.predict([user_input])[0]
            st.success(f"The predicted stage of liver cirrhosis is: Stage {stage_prediction}")
# Model Insights Section
elif selected == "Model Insights":
    st.title("Model Insights")
    st.write("""
    In this section, we provide detailed insights about the model used for liver cirrhosis stage prediction, 
    including its performance, feature importance, and more.
    """)

    # Displaying Model Performance (Accuracy, Precision, Recall, etc.)
    st.subheader("Model Performance")
    
    st.write("For demonstration purposes, the model's performance metrics are shown below:")
    accuracy = 0.9322  # Example accuracy value (replace with actual evaluation metric)

    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Displaying Model Performance (Class-wise Precision, Recall, F1-Score)
    st.subheader("Stage-wise Model Performance")

    # Performance metrics for each stage
    performance_data = {
        "Stage": [1, 2, 3],
        "Precision (%)": [94.00, 91.00, 94.00],
        "Recall (%)": [91.00, 93.00, 95.00],
        "F1-Score (%)": [92.00, 92.00, 95.00],
    }

    # Creating a DataFrame
    performance_df = pd.DataFrame(performance_data)

    # Set 'Stage' as the index to avoid showing an extra index column
    performance_df.set_index("Stage", inplace=True)

    # Display the table in Streamlit without an index column
    st.table(performance_df)



    st.subheader("Feature Importance")
    
    # Displaying feature importance from the model
    if hasattr(liver_model, 'feature_importances_'):
        feature_importance = liver_model.feature_importances_
        features = [
            "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", 
            "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", 
            "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
        ]
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        st.write("The following table shows the importance of each feature in predicting liver cirrhosis stage:")
        st.write(importance_df)

        # Plotting feature importance graph
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)

    else:
        st.write("The model does not have feature importance information.")

    st.subheader("Model Overview")
    st.write("""
    The model used for this prediction is a Random Forest Classifier, which is an ensemble learning method 
    that combines multiple decision trees to improve prediction accuracy.
    
    Key characteristics of the model:
    - **Training Dataset**: The model was trained using historical liver cirrhosis patient data.
    - **Type of Model**: Random Forest Classifier
    - **Prediction Objective**: Predict the stage of liver cirrhosis (Stage 1, Stage 2, or Stage 3)
    - **Input Features**: Patient health data including age, ascites, bilirubin levels, etc.
    - **Output**: Predicted stage of cirrhosis based on input features
    """)

# Report Generation Section (In-depth detailed report and PDF saving)
elif selected == "Report Generation":
    st.title("Generate Liver Cirrhosis Report")
    stage_selected = st.selectbox('Select Cirrhosis Stage', [1, 2, 3])

    if st.button('Generate Report'):
        # Generate the report
        report = generate_report(stage_selected)
        st.markdown(report, unsafe_allow_html=True)

        # Save the report as a PDF
        pdf_path = save_pdf(report, stage_selected)

        # Provide download link for the PDF
        st.download_button("Download Report as PDF", pdf_path, file_name=f"liver_cirrhosis_stage_{stage_selected}_report.pdf")



# Chatbot Section
if selected == "Chatbot":
    st.title("Liver Health Chatbot 🩺")
    st.write("Ask any questions about liver health or cirrhosis.")

    # Initialize the message history if not already done
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Create a form for user input
    with st.form("chat_form"):
        user_input = st.text_input("Ask me something:")
        send = st.form_submit_button("Send")
        clear = st.form_submit_button("Clear Chat")  # Clear chat button

    # Function to provide answers based on user input
    def get_bot_response(user_input):
        # Convert input to lowercase for easier matching
        user_input = user_input.lower()
        
        # Basic responses based on questions related to liver cirrhosis
        if "symptoms" in user_input:
            return "Common symptoms of liver cirrhosis include fatigue, abdominal pain, swelling, jaundice, and weight loss."
        elif "treatment" in user_input or "cure" in user_input:
            return "Liver cirrhosis treatment options include medications, lifestyle changes, managing complications, and in severe cases, a liver transplant."
        elif "causes" in user_input:
            return "Common causes of liver cirrhosis include chronic alcohol abuse, viral hepatitis (hepatitis B and C), and non-alcoholic fatty liver disease (NAFLD)."
        elif "prevent" in user_input or "prevention" in user_input:
            return "Preventing liver cirrhosis involves avoiding excessive alcohol use, managing chronic liver conditions, avoiding hepatitis infections, and maintaining a healthy diet and weight."
        elif "jaundice" in user_input:
            return "Jaundice is a condition where the skin and eyes turn yellow due to the liver's inability to process bilirubin, and it is often associated with liver cirrhosis."
        elif "lifestyle" in user_input:
            return "Lifestyle changes for liver cirrhosis include avoiding alcohol, eating a healthy and low-sodium diet, staying active, and managing underlying liver conditions."
        elif "transplant" in user_input:
            return "A liver transplant may be necessary for patients with advanced cirrhosis who do not respond to other treatments."
        elif "early signs" in user_input:
            return "Early signs of liver cirrhosis include fatigue, loss of appetite, nausea, and easy bruising."
        elif "alcohol" in user_input:
            return "Alcohol consumption can damage liver cells and is one of the leading causes of liver cirrhosis. Reducing or avoiding alcohol is crucial for liver health."
        elif "liver cirrhosis" in user_input:
            return "Liver cirrhosis is a condition characterized by scarring of liver tissue, often due to long-term damage from conditions like hepatitis or alcohol abuse."
        elif "difference between hepatitis and cirrhosis" in user_input:
            return "Hepatitis is inflammation of the liver, often caused by viral infections, while cirrhosis is the scarring of the liver tissue due to long-term damage."
        elif "non-alcoholic fatty liver disease" in user_input or "nafld" in user_input:
            return "Non-alcoholic fatty liver disease (NAFLD) is a condition where fat builds up in the liver, often linked to obesity, diabetes, or high cholesterol."
        elif "impact liver function" in user_input:
            return "Cirrhosis impairs liver function by causing scarring, which hinders blood flow and the liver's ability to process toxins and produce essential proteins."
        elif "complications" in user_input:
            return "Complications of liver cirrhosis include ascites, variceal bleeding, hepatic encephalopathy, and an increased risk of liver cancer."
        elif "diagnosis" in user_input:
            return "Doctors diagnose liver cirrhosis through physical exams, blood tests, imaging studies like ultrasounds or MRIs, and sometimes a liver biopsy."
        else:
            return "I'm here to help with liver health queries. Can you please ask something specific?"

    # Process the user input and display messages
    if send:
        if user_input.strip():  # Check if input is not empty
            bot_response = get_bot_response(user_input)
            # Add user input and bot response to message history
            st.session_state['messages'].append({"user": user_input, "bot": bot_response})
        else:
            st.warning("Please ask a question.")

     # Clear the chat if the button is pressed
    if clear:
        st.session_state['messages'] = []  # Clear chat history

    # Display messages in the chat
    for i, msg in enumerate(st.session_state['messages']):
        # Use unique keys to prevent DuplicateWidgetID error
        message(msg['user'], is_user=True, key=f"user_{i}")  # Display user message
        message(msg['bot'], key=f"bot_{i}")  # Display bot response


# Function to get hospitals using Overpass API from OpenStreetMap
def find_hospitals_osm(location, radius=5000):
    geolocator = Nominatim(user_agent="liver_health_assistant")
    location = geolocator.geocode(location)

    if location:
        latitude = location.latitude
        longitude = location.longitude
        st.write(f"Searching for hospitals near {location.address}...")

        # Query Overpass API for hospitals within a radius of the given location
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:{radius},{latitude},{longitude});
          way["amenity"="hospital"](around:{radius},{latitude},{longitude});
          relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
        );
        out body;
        """
        
        response = requests.get(overpass_url, params={'data': overpass_query})
        
        if response.status_code == 200:
            data = response.json()
            hospitals = data['elements']
            
            if len(hospitals) > 0:
                st.write(f"Found {len(hospitals)} hospitals nearby:")
                for hospital in hospitals:
                    name = hospital.get('tags', {}).get('name', 'No Name Available')
                    address = hospital.get('tags', {}).get('addr:full', 'No Address Available')
                    st.write(f"- {name} - {address}")
            else:
                st.write("No hospitals found in the nearby area.")
        else:
            st.error("Error querying Overpass API.")
    else:
        st.error("Location not found. Please try a valid location.")

# Hospital Finder Section
if selected == "Hospital Finder":
    st.title("Find Hospitals for Liver Cirrhosis Treatment")
    st.write("""
    In this section, you can search for hospitals nearby that treat liver cirrhosis or liver-related issues.
    Simply enter a city or address to get a list of hospitals.
    """)

    # Input for the user's location (city or address)
    location = st.text_input("Enter your city or address", "")

    if st.button("Search for nearby Hospitals"):
        find_hospitals_osm(location)