import streamlit as st
import pickle
import pandas as pd
import shap

# ================================
# 🔥 CONFIG
# ================================
st.set_page_config(page_title="Student AI", layout="wide")
if not os.path.exists("model.pkl"):
    file_id = "1U0KJbjJoG-kOJBwVnSMB_Jpj31Juiiwi"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, "model.pkl", quiet=False)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# ================================
# 🎯 LOAD MODEL
# ================================

# ================================
# 🎨 HEADER
# ================================
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
        🎓 Student Performance Assessment AI
    </h1>
    <p style='text-align: center;'>
        Predict student performance with Explainable AI insights
    </p>
""", unsafe_allow_html=True)

# ================================
# 🧩 LAYOUT (2 COLUMNS)
# ================================
col1, col2 = st.columns([1, 2])

# ================================
# 🎚️ INPUT PANEL
# ================================
with col1:
    st.subheader("📊 Student Profile")

    StudyHours = st.slider("Study Hours", 0, 12, 5)
    Attendance = st.slider("Attendance (%)", 0, 100, 75)
    Resources = st.slider("Resources Access", 0, 5, 2)
    Motivation = st.slider("Motivation Level", 0, 5, 3)
    StressLevel = st.slider("Stress Level", 0, 10, 5)

    Gender = st.selectbox("Gender", ["Female", "Male"])
    Internet = st.selectbox("Internet Access", ["No", "Yes"])
    Extracurricular = st.selectbox("Extracurricular", ["No", "Yes"])

    Age = st.slider("Age", 15, 40, 20)
    LearningStyle = st.selectbox("Learning Style", ["Visual", "Auditory", "Reading", "Kinesthetic"])
    OnlineCourses = st.slider("Online Courses", 0, 20, 5)
    Discussions = st.slider("Discussions Participation", 0, 10, 2)
    AssignmentCompletion = st.slider("Assignment Completion (%)", 0, 100, 60)
    EduTech = st.selectbox("EdTech Usage", ["No", "Yes"])

    predict_btn = st.button("🚀 Predict Performance")

# ================================
# 🔄 ENCODING
# ================================
Gender = 1 if Gender == "Male" else 0
Internet = 1 if Internet == "Yes" else 0
Extracurricular = 1 if Extracurricular == "Yes" else 0
EduTech = 1 if EduTech == "Yes" else 0

LearningStyle_map = {
    "Visual": 0,
    "Auditory": 1,
    "Reading": 2,
    "Kinesthetic": 3
}
LearningStyle = LearningStyle_map[LearningStyle]

# ================================
# 📊 DATAFRAME
# ================================
input_data = pd.DataFrame([{
    "StudyHours": StudyHours,
    "Attendance": Attendance,
    "Resources": Resources,
    "Extracurricular": Extracurricular,
    "Motivation": Motivation,
    "Internet": Internet,
    "Gender": Gender,
    "Age": Age,
    "LearningStyle": LearningStyle,
    "OnlineCourses": OnlineCourses,
    "Discussions": Discussions,
    "AssignmentCompletion": AssignmentCompletion,
    "EduTech": EduTech,
    "StressLevel": StressLevel
}])

# ================================
# 🎯 OUTPUT PANEL
# ================================
with col2:
    st.subheader("📈 Prediction Dashboard")

    if predict_btn:
        prediction = model.predict(input_data)[0]
        labels = ["Poor", "Average", "Good", "Excellent"]

        # 🎯 Result Card
        st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background:#e8f5e9;">
                <h2 style="color:#2e7d32;">Prediction: {labels[prediction]}</h2>
            </div>
        """, unsafe_allow_html=True)

        # 🔍 SHAP
        st.subheader("🔍 Explainable AI Insights")

        explainer = shap.Explainer(model.predict, input_data)
        shap_values = explainer(input_data)

        shap_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Impact": shap_values.values[0]
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.bar_chart(shap_df.set_index("Feature"))

        # 📋 Input Summary
        with st.expander("📋 View Input Data"):
            st.dataframe(input_data)

    else:
        st.info("👈 Enter student details and click Predict")