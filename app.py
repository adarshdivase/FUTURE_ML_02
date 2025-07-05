# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# --- Configuration ---
CSV_FILE_PATH = "Customer Churn new.csv" # Make sure your CSV file is named this

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(path):
    """Loads and preprocesses the Bank Churn dataset."""
    df = pd.read_csv(path)
    
    # Drop unnecessary columns
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    
    # The target variable is 'Exited'
    return df

@st.cache_resource
def train_model(df):
    """Preprocesses data, trains a Random Forest model, and returns it."""
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # --- THIS IS THE FIX ---
    # Define features based on the columns ACTUALLY in your dataset
    categorical_features = ['Geography', 'Gender']
    # Removed 'NumOfProducts', 'HasCrCard', 'IsActiveMember' as they are not in the provided file
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Create a full pipeline for preprocessing and modeling
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train the model
    rf_pipeline.fit(X_train, y_train)
    
    return rf_pipeline, X_test, y_test

# --- Main App ---
st.title("🏦 Bank Customer Churn Prediction")
st.write("This app analyzes customer data to predict whether they will exit the bank (churn).")

# Load data and train model
try:
    df = load_data(CSV_FILE_PATH)
    if df is not None:
        pipeline, X_test, y_test = train_model(df)

        # --- Model Performance Section ---
        st.header("Model Performance Analysis")

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Display accuracy metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        with col2:
            st.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.3f}")

        st.markdown("---")

        # Display plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Confusion Matrix
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax1, cmap='Blues')
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_proba):.3f})')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        
        st.pyplot(fig)

        # --- Live Prediction Section ---
        st.sidebar.header("🔮 Predict Live Churn")
        st.sidebar.write("Enter a customer's details to predict their churn probability.")

        # --- THIS IS THE FIX ---
        # Create input fields based on the available columns
        credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
        geography = st.sidebar.selectbox("Geography", df['Geography'].unique())
        gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
        age = st.sidebar.slider("Age", 18, 100, 35)
        tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
        balance = st.sidebar.slider("Balance", 0.0, 250000.0, 0.0)
        estimated_salary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 50000.0)
        
        if st.sidebar.button("Predict Churn"):
            # Create a DataFrame from the available inputs
            input_data = pd.DataFrame({
                'CreditScore': [credit_score], 'Geography': [geography], 'Gender': [gender],
                'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
                'EstimatedSalary': [estimated_salary]
            })

            # Make prediction
            churn_proba = pipeline.predict_proba(input_data)[0, 1]

            st.sidebar.subheader("Prediction Result")
            
            if churn_proba > 0.5:
                st.sidebar.error(f"High Churn Risk ({churn_proba:.2%})")
            else:
                st.sidebar.success(f"Low Churn Risk ({churn_proba:.2%})")

except FileNotFoundError:
    st.error(f"Error: The data file was not found at '{CSV_FILE_PATH}'. Please make sure the file is in the same folder as the app and has the correct name.")
except Exception as e:
    st.error(f"An error occurred: {e}")
