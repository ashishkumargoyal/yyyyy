import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
import xgboost

# Function to connect to MySQL and load data
@st.cache_data
def load_data():
    try:
        conn = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='Dob@1980',  # Replace with your MySQL password
            port=3306,
            database='tumor'
        )
        query = "SELECT * FROM titanic"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except pymysql.Error as e:
        st.error(f"MySQL Error: {e}")
        return None

# Function to preprocess data
def preprocess_data(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Embarked'] = le.fit_transform(df['Embarked'])
    return df

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('ash_xg_boost.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Streamlit app
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability.")

# Load data and model
df = load_data()
if df is not None:
    df = preprocess_data(df)
    model = load_model()

    # Input fields for user
    st.subheader("Passenger Details")
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Preprocess user input
    sex_encoded = 1 if sex == "male" else 0
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    embarked_encoded = embarked_mapping[embarked]

    # Create input array for prediction
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    input_df = pd.DataFrame(input_data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Predict survival
    if st.button("Predict"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
        result = "Survived" if prediction[0] == 1 else "Did not survive"
        st.success(f"Prediction: {result}")
        st.write(f"Survival Probability: {probability:.2%}")
else:
    st.error("Failed to load data from the database.")

# Display sample data
if df is not None:
    st.subheader("Sample Data")
    st.dataframe(df.head())