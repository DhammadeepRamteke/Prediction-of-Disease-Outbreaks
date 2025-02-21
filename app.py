import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

# ---------------- Setup Streamlit Page ----------------
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

# Ensure directories exist
MODEL_DIR = "saved_models"
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)  # Create history folder if not exists

# ---------------- Load Models Safely ----------------
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file **{model_name}** not found! Train and save it first.")
        st.stop()
    return joblib.load(model_path)

# Load all three models
diabetes_model = load_model("diabetes_model.joblib")
heart_model = load_model("heart_disease_model.joblib")
parkinsons_model = load_model("parkinsons_disease_model.joblib")

# ---------------- Sidebar Menu ----------------
with st.sidebar:
    st.title("🩺 Prediction of Disease Outbreaks")  # 1) Sidebar Title
    selected = option_menu(
        "Main Menu",
        ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Disease Prediction", "History"],
        icons=["house", "activity", "heart", "person", "clock"],
        menu_icon="hospital",
        default_index=0  # 2) 'Home' is the default page
    )

# ---------------- Save Prediction History ----------------
def save_to_history(filename, data):
    """
    Appends a single record (list) to a CSV file.
    """
    file_path = os.path.join(HISTORY_DIR, filename)
    df = pd.DataFrame([data])  # Convert single record to DataFrame
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)  # Append data
    else:
        df.to_csv(file_path, index=False)  # Create new file

# ---------------- Helper Function for Risk Message ----------------
def get_risk_message(risk_percentage):
    """
    Returns a custom message based on the given risk percentage.
    """
    if risk_percentage < 30:
        return "✅ **Low Risk:** Keep maintaining a healthy lifestyle!"
    elif risk_percentage < 70:
        return "⚠️ **Moderate Risk:** Consider consulting a medical professional."
    else:
        return "🚨 **High Risk:** Immediate medical attention is advised!"

# ---------------- HOME PAGE ----------------
if selected == "Home":
    st.title("Welcome to the Disease Prediction Model")
    st.markdown("""
    This AI-powered system helps predict the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson’s**.
    
    ### 🔹 Key Features:
    - **Fast & Accurate** Predictions
    - **User-Friendly** Interface
    - **Data-Driven** Insights
    
    ### 🚀 How to Use:
    1. Select a disease from the left menu.
    2. Enter the required health parameters.
    3. Click **Predict** to see your result and risk percentage.
    
    **Note**: This is just an Machine Learning model and not a substitute for professional medical advice. If you have any health concerns, please consult your doctor to ensure proper diagnosis and treatment.
    """)

# ---------------- DIABETES PREDICTION ----------------
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure Value', min_value=0)
    with col1:
        SkinThickness = st.number_input('Skin Thickness Value', min_value=0)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0)
    with col3:
        BMI = st.number_input('Body Mass Index', min_value=0.0, format="%.5f")
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.5f")
    with col2:
        Age = st.number_input('Age', min_value=0, step=1)

    if st.button("Predict Diabetes"):
        features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = diabetes_model.predict(features)[0]
        
        # 3) Calculate risk percentage using predict_proba
        probability = diabetes_model.predict_proba(features)[0][1]
        risk_percentage = int(probability * 100)

        # Define final result string
        result = "Has Diabetes" if prediction == 1 else "No Diabetes"

        # Display the result and risk
        st.success(f"**Prediction:** {result}")
        st.subheader(f"📊 Risk Percentage: **{risk_percentage}%**")
        st.info(get_risk_message(risk_percentage))

        # Save history (with risk_percentage)
        save_to_history(
            "diabetes_history.csv",
            [
                datetime.now(),  # Timestamp
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age,
                result, risk_percentage
            ]
        )

# ---------------- HEART DISEASE PREDICTION ----------------
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    # Adjust your input fields to match the order your model expects
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input('Age', min_value=0, step=1)
    with col2:
        Sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    with col3:
        ChestPain = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, step=1)
    with col1:
        RestingBP = st.number_input('Resting Blood Pressure', min_value=0)
    with col2:
        Cholesterol = st.number_input('Cholesterol Level', min_value=0)
    with col3:
        FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)', [0, 1])
    with col1:
        RestECG = st.number_input('Resting ECG Results (0-2)', min_value=0, max_value=2, step=1)
    with col2:
        MaxHR = st.number_input('Maximum Heart Rate Achieved', min_value=0)
    with col3:
        ExerciseAngina = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
    with col1:
        Oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, format="%.5f")
    with col2:
        Slope = st.number_input('Slope of the ST Segment (0-2)', min_value=0, max_value=2, step=1)
    with col3:
        Ca = st.number_input('Number of Major Vessels (0-4)', min_value=0, max_value=4, step=1)
    with col1:
        Thal = st.number_input('Thalassemia (0-3)', min_value=0, max_value=3, step=1)

    if st.button("Predict Heart Disease"):
        features = [[Age, Sex, ChestPain, RestingBP, Cholesterol, FastingBS, RestECG, MaxHR,
             ExerciseAngina, Oldpeak, Slope, Ca, Thal]]
        prediction = heart_model.predict(features)[0]
        
        # 3) Calculate risk percentage
        probability = heart_model.predict_proba(features)[0][1]
        risk_percentage = int(probability * 100)

        # Define final result string
        result = "Has Heart Disease" if prediction == 1 else "No Heart Disease"

        # Display the result and risk
        st.success(f"**Prediction:** {result}")
        st.subheader(f"📊 Risk Percentage: **{risk_percentage}%**")
        st.info(get_risk_message(risk_percentage))

        # Save history (with risk_percentage)
        save_to_history(
            "heart_history.csv",
            [
                datetime.now(), Age, Sex, ChestPain, RestingBP, Cholesterol, FastingBS,
                MaxHR, ExerciseAngina, Oldpeak, Slope, Ca, Thal, result, risk_percentage
            ]
        )

# ---------------- PARKINSON’S DISEASE PREDICTION ----------------
if selected == "Parkinson's Disease Prediction":
    st.title("Parkinson’s Disease Prediction")

    # Adjust your input fields to match the order your model expects
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo = st.number_input('MDVP:Fo (Fundamental Frequency in Hz)', format="%.5f")
    with col2:
        MDVP_Fhi = st.number_input('MDVP:Fhi (Highest Frequency in Hz)', format="%.5f")
    with col3:
        MDVP_Flo = st.number_input('MDVP:Flo (Lowest Frequency in Hz)', format="%.5f")
    with col1:
        MDVP_Jitter = st.number_input('MDVP:Jitter (%)', format="%.5f")
    with col2:
        MDVP_Jitter_Abs = st.number_input('MDVP:Jitter (Abs)', format="%.5f")
    with col3:
        MDVP_RAP = st.number_input('MDVP:RAP', format="%.5f")
    with col1:
        MDVP_PPQ = st.number_input('MDVP:PPQ', format="%.5f")
    with col2:
        Jitter_DDP = st.number_input('Jitter:DDP', format="%.5f")
    with col3:
        MDVP_Shimmer = st.number_input('MDVP:Shimmer', format="%.5f")
    with col1:
        MDVP_Shimmer_dB = st.number_input('MDVP:Shimmer (dB)', format="%.5f")
    with col2:
        Shimmer_APQ3 = st.number_input('Shimmer:APQ3', format="%.5f")
    with col3:
        Shimmer_APQ5 = st.number_input('Shimmer:APQ5', format="%.5f")
    with col1:
        MDVP_APQ = st.number_input('MDVP:APQ', format="%.5f")
    with col2:
        Shimmer_DDA = st.number_input('Shimmer:DDA', format="%.5f")
    with col3:
        NHR = st.number_input('NHR (Noise-to-Harmonics Ratio)', format="%.5f")
    with col1:
        HNR = st.number_input('HNR (Harmonics-to-Noise Ratio)', format="%.5f")
    with col2:
        RPDE = st.number_input('RPDE (Recurrence Period Density Entropy)', format="%.5f")
    with col3:
        DFA = st.number_input('DFA (Detrended Fluctuation Analysis)', format="%.5f")
    with col1:
        spread1 = st.number_input('Spread1 (Nonlinear Measure of Fundamental Frequency)', format="%.5f")
    with col2:
        spread2 = st.number_input('Spread2', format="%.5f")
    with col3:
        D2 = st.number_input('D2 (Correlation Dimension)', format="%.5f")
    with col1:
        PPE = st.number_input('PPE (Pitch Period Entropy)', format="%.5f")

    if st.button("Predict Parkinson’s Disease"):
        # Adjust features to the exact input order your model expects
        features = [[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ,
                     Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,
                     Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
        prediction = parkinsons_model.predict(features)[0]

        # 3) Calculate risk percentage
        probability = parkinsons_model.predict_proba(features)[0][1]
        risk_percentage = int(probability * 100)

        # Define final result string
        result = "Has Parkinson’s" if prediction == 1 else "No Parkinson’s"

        # Display the result and risk
        st.success(f"**Prediction:** {result}")
        st.subheader(f"📊 Risk Percentage: **{risk_percentage}%**")
        st.info(get_risk_message(risk_percentage))

        # Save history (with risk_percentage)
        save_to_history(
            "parkinsons_history.csv",
            [
                datetime.now(), MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ,
                     Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,
                     Shimmer_DDA, NHR, HNR, RPDE, DFA, result, risk_percentage
            ]
        )

# ---------------- HISTORY SECTION ----------------
if selected == "History":
    st.title("Prediction History")
    history_choice = st.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Parkinson’s"])

    file_map = {
        "Diabetes": "diabetes_history.csv",
        "Heart Disease": "heart_history.csv",
        "Parkinson’s": "parkinsons_history.csv"
    }
    history_file = os.path.join(HISTORY_DIR, file_map[history_choice])

    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        st.dataframe(df)

        # --- Download CSV ---
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=file_map[history_choice],
            mime="text/csv"
        )

        # --- Clear History ---
        if st.button("Clear History"):
            os.remove(history_file)
            st.success(f"{history_choice} history cleared successfully!")
            st.rerun()

    else:
        st.info("No history available yet.")
