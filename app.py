import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# Load scaler and selected features
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

with open('models/selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

st.title("💓 Heart Disease Prediction App (Local Model)")

# Numeric inputs
age = st.number_input("Age", min_value=1, max_value=120)
resting_bp = st.number_input("RestingBP")
chol = st.number_input("Cholesterol")
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
max_hr = st.number_input("MaxHR")
oldpeak = st.number_input("Oldpeak")

# Categorical inputs
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
chest_pain = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'LVH', 'ST'])
exercise_angina = st.selectbox("Exercise Angina", ['N', 'Y'])
st_slope = st.selectbox("ST Slope", ['Up', 'Flat'])

if st.button("Predict"):
    # Prepare numeric features
    numeric_features = [age, resting_bp, chol, fasting_bs, max_hr, oldpeak]
    input_df = pd.DataFrame([numeric_features], columns=numeric_cols)
    scaled_numeric = scaler.transform(input_df)

    # One-hot encode categorical features (simulate get_dummies with drop_first=True)
    chest_pain_encoded = [1 if chest_pain == 'ATA' else 0,
                          1 if chest_pain == 'NAP' else 0]

    resting_ecg_encoded = [1 if resting_ecg == 'LVH' else 0,
                           1 if resting_ecg == 'ST' else 0]

    sex_encoded = [sex]  # Already binary

    exercise_angina_encoded = [1 if exercise_angina == 'Y' else 0]

    st_slope_encoded = [1 if st_slope == 'Flat' else 0,
                        1 if st_slope == 'Up' else 0]

    # Combine all features
    input_vector = np.hstack([
        scaled_numeric.flatten(),
        sex_encoded,
        chest_pain_encoded,
        resting_ecg_encoded,
        exercise_angina_encoded,
        st_slope_encoded
    ])

    # Select features expected by model
    all_feature_names = numeric_cols + ['Sex', 'ChestPainType_ATA', 'ChestPainType_NAP',
                                        'RestingECG_LVH', 'RestingECG_ST',
                                        'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
    input_df_full = pd.DataFrame([input_vector], columns=all_feature_names)

    # Filter to selected features only
    input_selected = input_df_full[selected_features].values

    # Load model from local file
    model = joblib.load("models/best_model.pkl")

    # Predict
    prediction = model.predict(input_selected)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease!")
    else:
        st.success("✅ Low risk of Heart Disease!")
