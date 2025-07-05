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
    page_title="Customer Churn Prediction",
    page_icon="⚡",
    layout="wide"
)

# --- Configuration ---
CSV_FILE_PATH = "Customer Churn new.csv" # Make sure your CSV file is named this

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(path):
    """Loads and preprocesses the Telco Churn dataset."""
    df = pd.read_csv(path)
    
    # --- THIS IS THE FIX ---
    # Check for the column with a space and rename it to match the expected name.
    if 'Total Charges' in df.columns:
        df.rename(columns={'Total Charges': 'TotalCharges'}, inplace=True)

    # Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

@st.cache_resource
def train_model(df):
    """Preprocesses data, trains a Random Forest model, and returns it."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
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
st.title("⚡ Customer Churn Prediction Dashboard")
st.write("This app analyzes customer data to predict churn using a Random Forest model.")

# Load data and train model
try:
    df = load_data(CSV_FILE_PATH)
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

    # Create input fields in the sidebar
    gender = st.sidebar.selectbox("Gender", df['gender'].unique())
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", df['Partner'].unique())
    dependents = st.sidebar.selectbox("Dependents", df['Dependents'].unique())
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", df['PhoneService'].unique())
    multiple_lines = st.sidebar.selectbox("Multiple Lines", df['MultipleLines'].unique())
    internet_service = st.sidebar.selectbox("Internet Service", df['InternetService'].unique())
    online_security = st.sidebar.selectbox("Online Security", df['OnlineSecurity'].unique())
    online_backup = st.sidebar.selectbox("Online Backup", df['OnlineBackup'].unique())
    device_protection = st.sidebar.selectbox("Device Protection", df['DeviceProtection'].unique())
    tech_support = st.sidebar.selectbox("Tech Support", df['TechSupport'].unique())
    streaming_tv = st.sidebar.selectbox("Streaming TV", df['StreamingTV'].unique())
    streaming_movies = st.sidebar.selectbox("Streaming Movies", df['StreamingMovies'].unique())
    contract = st.sidebar.selectbox("Contract", df['Contract'].unique())
    paperless_billing = st.sidebar.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
    payment_method = st.sidebar.selectbox("Payment Method", df['PaymentMethod'].unique())
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.slider("Total Charges ($)", 18.0, 9000.0, 1400.0)

    if st.sidebar.button("Predict Churn"):
        # Create a DataFrame from the inputs
        input_data = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner], 'Dependents': [dependents],
            'tenure': [tenure], 'PhoneService': [phone_service], 'MultipleLines': [multiple_lines],
            'InternetService': [internet_service], 'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection], 'TechSupport': [tech_support], 'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies], 'Contract': [contract], 'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
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
