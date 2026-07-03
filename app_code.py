import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv('cleaned_data.csv')
model = pickle.load(open("model.pkl", "rb"))
sc = pickle.load(open("sc.pkl", "rb"))

# ---------------- Background Image ---------------- #

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("diabetes.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255,255,255,0.7),
                                    rgba(255,255,255,0.7)),
                    url("data:image/jpeg;base64,{img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏥 Diabetic Patient Prediction App")

# ---------------- Sidebar ---------------- #

st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color: #FF8C00 !important;
    border-right: 2px solid #E67E22;
}

/* Sidebar Text */
section[data-testid="stSidebar"] *{
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.set_page_config(initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    [data-testid="stSidebarHeader"]{
        display:none;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("")

    st.markdown(
        "<h1 style='font-weight:900;font-size:40px;color:white;text-align:center;'>Factors</h1>",
        unsafe_allow_html=True
    )

    st.title("")

    Pregnancies = st.slider("🤰 Pregnancies", 0, 20, 3)
    Glucose = st.slider("🍬 Glucose", 44, 199, 117)
    BloodPressure = st.slider("🩺 Blood Pressure", 40, 122, 72)
    SkinThickness = st.slider("🧴 Skin Thickness", 7, 99, 23)
    Insulin = st.slider("💉 Insulin", 15, 184, 125)
    BMI = st.slider("⚖️ BMI", 10, 67, 32)
    DiabetesPedigreeFunction = st.slider(
        "🧬 Diabetes Pedigree Function", 0.078, 2.42, 0.37
    )
    Age = st.slider("🎂 Age", 21, 81, 33)

    # Slider Label Size

    st.markdown("""
    <style>
    [data-testid="stSlider"] p{
        font-size:20px !important;
        font-weight:normal !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Button Styling

    st.markdown("""
    <style>

    div.stButton > button{
        background: linear-gradient(135deg,#3498db,#2980b9);
        color:white;
        font-size:18px;
        font-weight:bold;
        border-radius:12px;
        padding:12px 30px;
        border:none;
        transition:all 0.3s ease;
        cursor:pointer;
        box-shadow:0px 4px 10px rgba(0,0,0,0.2);
    }

    div.stButton > button:hover{
        background: linear-gradient(135deg,#2980b9,#1f618d);
        transform:translateY(-3px);
        box-shadow:0px 8px 18px rgba(0,0,0,0.3);
    }

    </style>
    """, unsafe_allow_html=True)

    # Only ONE button
    diagnose = st.button("Diagnose Patient")

# ---------------- Prediction ---------------- #

if diagnose:

    myip = [[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]]

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    scaled = sc.transform(myip)
    frame = pd.DataFrame(scaled, columns=columns)

    
    data = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    table = pd.DataFrame({'data':[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age] , 'columns' : ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']})
    
   

    
    styled_table = table.style.format(precision=0).apply(
    lambda x: ['background-color: #FDEBD0' if x.name % 2 == 0 else 'background-color: #FAD7A0' for _ in x],
    axis=1
    )

    st.dataframe(
    styled_table,
    hide_index=True,
    width=300
    )
   
    st.title("")
    data1 = {'data':['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'] , 'columns' :  [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]}
    chart_df = pd.DataFrame(data1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(chart_df['data'], chart_df['columns'])
    plt.xticks(rotation = 90)
    st.pyplot(fig)

    result = model.predict(frame)
    st.title("")

    if result[0] == 1:
       st.markdown("""
       <div style="
       background: linear-gradient(135deg,#ff6b6b,#ff3b3b);
       padding:15px;
       border-radius:12px;
       color:white;
       font-size:18px;
       font-weight:bold;
       box-shadow:0px 4px 12px rgba(0,0,0,0.2);
       ">
⚠️     Patient is Diabetic
       </div>
       """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
        background: linear-gradient(135deg,#2ecc71,#27ae60);
        padding:15px;
        border-radius:12px;
        color:white;
        font-size:18px;
        font-weight:bold;
        box-shadow:0px 4px 12px rgba(0,0,0,0.2);
        ">
        ✅ Patient is Not Diabetic
        </div>
        """, unsafe_allow_html=True)

    st.title("")
    st.caption("Note : The final data is predicted based on other factors too. Accuracy may vary since it is assumed considering previous reords of the conditions.")