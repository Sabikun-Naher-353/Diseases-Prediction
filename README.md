**Health Risk Predictor**
Machine Learning–Based Disease Risk Prediction System

**Overview**

Health Risk Predictor is a machine learning–driven web application developed using Python and Streamlit to estimate the potential risk of several common diseases based on user-provided health information.

The system does not perform medical diagnosis. Its purpose is to provide early risk indication and promote health awareness, prevention, and informed decision-making.

**Objectives**

The main objectives of this project are to promote early health awareness, demonstrate practical use of machine learning in healthcare, provide a clean and user-friendly interface, and showcase a modular and scalable prediction system.

**Disclaimer:**
This application is strictly for educational and awareness purposes and must not be considered a substitute for professional medical advice or clinical diagnosis.

**Diseases Supported**

The application currently supports risk prediction for the following diseases:

Diabetes
Asthma
Hypertension
Sleep Disorder

Each disease is handled by an independent machine learning model to ensure better specialization and prediction reliability.

**Application Concept**

The system follows a simple and logical workflow.

The user opens the application and selects a disease from the sidebar.
The system displays a disease-specific input interface.
The user provides health-related information.
A trained machine learning pipeline processes the data.
The system outputs a risk prediction based on learned patterns.

This modular design allows easy expansion with additional diseases in the future.

**Home Page Description**

The home page acts as the central navigation point of the application. It introduces the system and allows users to select a disease using the sidebar. Each selection dynamically loads the corresponding prediction module without restarting the application.

**Disease Prediction Modules**

**Diabetes Prediction**
Uses metabolic and lifestyle indicators to estimate diabetes risk.

**Asthma Prediction**
Analyzes respiratory, environmental, and personal factors to predict asthma risk.

**Hypertension Prediction**
Evaluates cardiovascular and lifestyle data to estimate high blood pressure risk.

**Sleep Disorder Prediction**
Assesses sleep patterns, stress levels, and daily habits to estimate sleep-related disorder risk.

Each module includes structured user input handling, data preprocessing, trained model loading, and clear prediction output.

**Technologies Used**

Python
Streamlit
Scikit-learn
Pandas
NumPy
Joblib and Pickle
Git Large File Storage for model management


**Video Link:** https://drive.google.com/file/d/1MtsVDsFyYko6E_6Tncqwmf3ucHRLWR_p/view?usp=sharing
