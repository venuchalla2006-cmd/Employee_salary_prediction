import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="💰 Income Prediction App",
    page_icon="💼",
    layout="wide"
)

# --- ANIMATED CSS STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: #fff;
            background: linear-gradient(-45deg, #1f1c2c, #2c5364, #203A43, #0F2027);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        h1 {
            text-align: center;
            color: #00BFFF;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
            font-size: 42px;
        }

        h2 {
            text-align: center;
            color: #87CEFA;
            font-weight: 600;
        }

        .stButton>button {
            background-color: #00BFFF;
            color: white;
            border-radius: 15px;
            border: none;
            font-size: 18px;
            padding: 10px 24px;
            font-weight: 600;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #1E90FF;
            transform: scale(1.05);
        }

        .main-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 25px;
            margin-top: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        }

        .prediction-box {
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 30px;
            animation: fadeIn 1s ease-in-out;
        }

        .success-box {
            background: linear-gradient(90deg, #00C9A7, #92FE9D);
            color: #003300;
        }

        .warning-box {
            background: linear-gradient(90deg, #FF512F, #DD2476);
            color: white;
        }

        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<h1>💼 Income Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#E0FFFF;'>Predict whether an individual earns more or less than 50K per year using Machine Learning.</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD MODEL ---
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "best_model.pkl")

try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found at {model_path}. Please ensure 'best_model.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- INPUTS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4205/4205973.png", width=130)
st.sidebar.markdown("<h2>🧍 Personal Details</h2>", unsafe_allow_html=True)

age = st.sidebar.slider("Age", 17, 75, 30)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'India', 'Canada', 'Mexico', 'Philippines', 'Germany', 'England', 'China', 'Japan', 'Others'])

st.sidebar.markdown("---")
st.sidebar.markdown("<h2>💼 Employment Info</h2>", unsafe_allow_html=True)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
fnlwgt = st.sidebar.number_input("Fnlwgt", value=100000)
educational_num = st.sidebar.slider("Education Level", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Widowed'])
occupation = st.sidebar.selectbox("Occupation", ['Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Craft-repair', 'Tech-support', 'Others'])
relationship = st.sidebar.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife'])
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# --- INPUT DATAFRAME ---
input_data = {
    'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt,
    'educational-num': educational_num, 'marital-status': marital_status,
    'occupation': occupation, 'relationship': relationship, 'race': race,
    'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
    'hours-per-week': hours_per_week, 'native-country': native_country
}
input_df = pd.DataFrame([input_data])
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

encoders = {col: LabelEncoder().fit(input_df[col]) for col in categorical_cols}
for col in categorical_cols:
    input_df[col] = encoders[col].transform(input_df[col])

# --- PREDICTION ---
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("### 🔍 Predict Income")
if st.button("🚀 Predict Now"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == '<=50K':
            st.markdown("<div class='prediction-box success-box'>💵 Predicted Income: ≤ 50K</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction-box warning-box'>🚀 Predicted Income: > 50K</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# --- FEATURE IMPORTANCE ---
if hasattr(model, 'feature_importances_'):
    st.markdown("<h2>📊 Feature Importance</h2>", unsafe_allow_html=True)
    feature_importances = pd.Series(model.feature_importances_, index=input_df.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='coolwarm', ax=ax)
    plt.title("Feature Importance for Income Prediction")
    st.pyplot(fig)
else:
    st.info("Feature importance is not available for this model.")
