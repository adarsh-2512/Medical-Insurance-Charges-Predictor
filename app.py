import streamlit as st
import numpy as np
import joblib  # Import joblib to load the saved model
import warnings  # Import the warnings library
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns  # Import seaborn for more advanced plots

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')  # Load the model from the file
    return model

# Load dataset for data visualization
@st.cache_data
def load_data():
    return pd.read_csv('insurance.csv')

st.set_page_config(
    page_title="HealthWise: Medical Insurance Prediction App",
    page_icon="🏥",  # Use a hospital emoji or any other relevant emoji
)

# Function to make predictions
def predict_charges(input_data):
    # Changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore all warnings
        model = load_model()  # Ensure model is loaded before prediction
        prediction = model.predict(input_data_reshaped)
    
    return prediction[0]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Visualization", "Project Details"])

# Prediction Page
if page == "Prediction":
    
    # Set light theme using custom CSS
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        
        """,
        unsafe_allow_html=True
    )

    # Centered title
    st.markdown('<h1 class="title">HealthWise: Medical Insurance Prediction App</h1>', unsafe_allow_html=True)

    # Input features
    st.header('Submit Details')

    age = st.number_input('Age', min_value=0, max_value=120, value=31)
    sex = st.selectbox('Sex', options=['Female', 'Male'], index=1)
    bmi = st.slider('BMI', min_value=15.0, max_value=30.0, value=25.74, step=0.1)
    children = st.slider('Number of Children', min_value=0, max_value=5, value=0, step=1)
    smoker = st.selectbox('Smoker', options=['No', 'Yes'], index=1)
    region = st.selectbox('Region', options=['Southwest', 'Southeast', 'Northwest', 'Northeast'])

    sex_mapping = {'Female': 0, 'Male': 1}
    smoker_mapping = {'No': 1, 'Yes': 0}
    region_encoded = {'Southwest': 0, 'Southeast': 1, 'Northwest': 2, 'Northeast': 3}

    input_data = (age, sex_mapping[sex], bmi, children, smoker_mapping[smoker], region_encoded[region])

    if st.button('Predict'):
        prediction = predict_charges(input_data)
        st.success(f'The predicted insurance cost is USD {prediction:.2f}')

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization of Medical Insurance Dataset")
    df = load_data()

    st.write("Here is a glimpse of the dataset:")
    st.write(df.head())

    # Pie chart for smoker distribution
    st.subheader("Smoker Distribution (Pie Chart)")
    smoker_counts = df['smoker'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    st.pyplot(fig)

    # Count plot for sex
    st.subheader("Sex Distribution (Count Plot)")
    fig, ax = plt.subplots()
    sns.countplot(x='sex', data=df, ax=ax)
    st.pyplot(fig)

    # Count plot for number of children
    st.subheader("Children Distribution (Count Plot)")
    fig, ax = plt.subplots()
    sns.countplot(x='children', data=df, ax=ax)
    st.pyplot(fig)

    # Count plot for region
    st.subheader("Region Distribution (Count Plot)")
    fig, ax = plt.subplots()
    sns.countplot(x='region', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of BMI")
    fig, ax = plt.subplots()
    sns.histplot(df['bmi'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Medical Charges vs BMI")
    fig, ax = plt.subplots()
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, ax=ax)
    st.pyplot(fig)

# Project Details Page
elif page == "Project Details":
    st.title("Project Details")

    st.subheader("About the Project")
    st.write("""
    This project aims to predict medical insurance charges based on user inputs like age, BMI, smoking status, and region. 
    It uses a Random Forest Regression model trained on a dataset of medical charges.

    ### Project Goals:
    - Develop an accurate predictive model for estimating insurance costs.
    - Provide an easy-to-use web interface for users to get predictions.
    - Visualize the relationships between various factors like BMI, age, and insurance costs.

    ### Tools and Technologies:
    - **Python** for data processing and model building.
    - **Streamlit** for building the web app.
    - **Random Forest Regression** for prediction.
    - **Seaborn** and **Matplotlib** for data visualization.
    """)

    st.subheader("Dataset Used")
    st.write("""
    The dataset used in this project contains information about the following features:
    - `Age`: The age of the individual.
    - `Sex`: The gender of the individual.
    - `BMI`: Body Mass Index, a measure of body fat based on height and weight.
    - `Children`: Number of children/dependents covered by the insurance.
    - `Smoker`: Whether the individual is a smoker.
    - `Region`: The region in which the individual resides.
    - `Charges`: Medical insurance charges billed to the individual.
    """)

    st.subheader("Model Details")
    st.write("""
    A Random Forest Regressor was used as the predictive model. The Random Forest model was trained using the `insurance.csv` dataset, and the features were preprocessed accordingly before model training.
    """)
