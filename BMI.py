import streamlit as st

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