import streamlit as st

st.title("Welcome to Health Risk Predictor")
st.write("Select the disease you want to check:")

st.sidebar.title("Choose Disease")
disease = st.sidebar.radio(
    "Go to:",
    ["Home", "Diabetes", "Asthma", "Hypertension","Sleep Dis-order"]
)

if disease == "Home":
    st.write("This is the home page. Click a disease on the sidebar to check your risk.")
elif disease == "Diabetes":
    import diabetes
    diabetes.run()
elif disease == "Asthma":
    import asthma
    asthma.run()
elif disease == "Hypertension":
    import hypertension
    hypertension.run()
elif disease == "Sleep Dis-order":
    import sleep
    sleep.run()
