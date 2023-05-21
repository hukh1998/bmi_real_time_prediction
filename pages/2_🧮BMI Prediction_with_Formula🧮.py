import streamlit as st

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
    page_title = "BMI Prediction with Formula",
    layout = "wide",
    initial_sidebar_state="expanded")

add_bg_from_url() 

st.write("# Enter Information to Find Your BMI!🎚️")

height = st.text_input("Enter your height (cm): ")
weight = st.text_input("Enter your height (kg): ")
cal = st.button('Calculate')

if cal:
    try:
        bmi = (float(weight)/float(height)/float(height)) * 10000
        st.write("Your BMI is:", round(bmi, 4))
    except:
        st.write("The values you entered are not valid!")