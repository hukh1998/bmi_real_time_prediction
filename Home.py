import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://bpb-us-w2.wpmucdn.com/voices.uchicago.edu/dist/e/2560/files/2019/04/UChicago_Phoenix-Wallpaper-Gray.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    


st.set_page_config(
    page_title = "BMI Prediction",
    layout = "wide",
    initial_sidebar_state="expanded")

add_bg_from_url() 

st.sidebar.success("Select a method above.")

st.write("# BMI Prediction! 👋")

st.markdown(
    """
    The goal of this application is to replicate the findings in this paper https://arxiv.org/abs/1703.03156
"""
)

col1, col2 = st.columns(2)



bmi = ['Below 18.5', '18.5 to 24.9', '25.0 to 29.9', '30 or higher', '40 or higher']
consider = ['Underweight', 'Healthy weight', 'Overweight', 'Obesity', 'Class 3 Obesity']

df = pd.DataFrame({"BMI": bmi, "Considered" : consider})
with col1:
    st.subheader("Adult Body Mass Index (BMI)")
    st.table(df)