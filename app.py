import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─── Load model & scaler ───────────────────────────────────────
model         = joblib.load('burnout_model.pkl')
scaler        = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Burnout Predictor",
    page_icon="🔥",
    layout="centered"
)

# ─── Header ────────────────────────────────────────────────────
st.title("🔥 Tech Burnout Predictor")
st.markdown("Answer a few questions to assess your burnout risk level.")
st.divider()

# ─── Input Form ────────────────────────────────────────────────
st.subheader("👤 Personal Info")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])

with col2:
    experience_years = st.slider("Experience (Years)", 0, 20, 3)
    job_role = st.selectbox("Job Role", [
        "Software Engineer", "Data Scientist", "ML Engineer",
        "Backend Developer", "Frontend Developer",
        "DevOps", "Product Manager", "QA Engineer"
    ])

st.divider()
st.subheader("🏢 Work Environment")
col3, col4 = st.columns(2)

with col3:
    company_size = st.selectbox("Company Size", ["Startup", "Mid-size", "Large", "MNC"])
    work_mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"])
    work_hours_per_week = st.slider("Work Hours/Week", 30, 84, 45)
    overtime_hours = st.slider("Overtime Hours/Week", 0, 24, 2)

with col4:
    meetings_per_day = st.slider("Meetings/Day", 0, 12, 3)
    deadlines_missed = st.slider("Deadlines Missed (monthly)", 0, 5, 0)
    job_satisfaction = st.slider("Job Satisfaction (1-10)", 1, 10, 5)
    manager_support = st.slider("Manager Support (1-10)", 1, 10, 5)

st.divider()
st.subheader("🧘 Health & Lifestyle")
col5, col6 = st.columns(2)

with col5:
    work_life_balance = st.slider("Work-Life Balance (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Sleep Hours/Night", 3, 10, 7)
    physical_activity_days = st.slider("Exercise Days/Week", 0, 7, 2)
    screen_time_hours = st.slider("Screen Time Hours/Day", 3, 16, 8)

with col6:
    caffeine_intake = st.slider("Caffeine Intake (cups/day)", 0, 5, 2)
    social_support_score = st.slider("Social Support (1-10)", 1, 10, 5)
    has_therapy = st.selectbox("Currently in Therapy?", ["No", "Yes"])
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

col7, col8 = st.columns(2)
with col7:
    anxiety_score = st.slider("Anxiety Score (1-10)", 1, 10, 5)
with col8:
    depression_score = st.slider("Depression Score (1-10)", 1, 10, 3)

st.divider()

# ─── Prediction ────────────────────────────────────────────────
if st.button("🔍 Predict My Burnout Risk", use_container_width=True):

    # Build input dict with all base features
    input_dict = {
        'age': age,
        'experience_years': experience_years,
        'work_hours_per_week': work_hours_per_week,
        'overtime_hours': overtime_hours,
        'meetings_per_day': meetings_per_day,
        'deadlines_missed': deadlines_missed,
        'job_satisfaction': job_satisfaction,
        'manager_support': manager_support,
        'work_life_balance': work_life_balance,
        'sleep_hours': sleep_hours,
        'physical_activity_days': physical_activity_days,
        'screen_time_hours': screen_time_hours,
        'caffeine_intake': caffeine_intake,
        'social_support_score': social_support_score,
        'has_therapy': 1 if has_therapy == "Yes" else 0,
        'stress_level': stress_level,
        'anxiety_score': anxiety_score,
        'depression_score': depression_score,
    }

    # One-hot encode categorical fields
    for col in feature_names:
        if col not in input_dict:
            input_dict[col] = 0

    # Set correct dummy columns
    if f'gender_{gender}' in feature_names:
        input_dict[f'gender_{gender}'] = 1
    if f'job_role_{job_role}' in feature_names:
        input_dict[f'job_role_{job_role}'] = 1
    if f'company_size_{company_size}' in feature_names:
        input_dict[f'company_size_{company_size}'] = 1
    if f'work_mode_{work_mode}' in feature_names:
        input_dict[f'work_mode_{work_mode}'] = 1

    # Create dataframe in correct column order
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # ─── Show Result ───────────────────────────────────────────
    st.subheader("📊 Your Burnout Risk Assessment")

    if prediction == 0:
        st.success("✅ Low Burnout Risk")
        st.markdown("You seem to be managing well! Keep maintaining healthy boundaries.")
    elif prediction == 1:
        st.warning("⚠️ Moderate Burnout Risk")
        st.markdown("Signs of burnout detected. Consider talking to your manager or a therapist.")
    else:
        st.error("🔴 High Burnout Risk")
        st.markdown("High burnout detected. Please seek professional support immediately.")

    # Show probabilities
    st.subheader("📈 Confidence Breakdown")
    prob_df = pd.DataFrame({
        'Risk Level': ['Low', 'Moderate', 'High'],
        'Probability': [f"{p*100:.1f}%" for p in probability]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Tips
    st.subheader("💡 Personalized Tips")
    if stress_level >= 7:
        st.markdown("- 🧘 **High stress detected** — try meditation or breathing exercises")
    if work_hours_per_week >= 55:
        st.markdown("- ⏰ **Overworking** — try to cap hours and take breaks")
    if sleep_hours <= 5:
        st.markdown("- 😴 **Poor sleep** — aim for 7-8 hours per night")
    if work_life_balance <= 3:
        st.markdown("- ⚖️ **Poor work-life balance** — set boundaries after work hours")
    if physical_activity_days == 0:
        st.markdown("- 🏃 **No exercise** — even a 20 min walk helps significantly")
    if has_therapy == "No" and prediction >= 1:
        st.markdown("- 🗣️ **Consider therapy** — talking to a professional can really help")