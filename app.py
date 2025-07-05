# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
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
    # Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

@st.cache_resource
def train_models(df):
    """Preprocesses data, trains models, and returns them along with test data."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create a full pipeline for preprocessing
    preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit and transform training data
    X_train_processed = preprocess_pipeline.fit_transform(X_train)
    X_test_processed = preprocess_pipeline.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    # 1. Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_resampled, y_train_resampled)

    # 2. Train Neural Network
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=32, verbose=0)
    
    return preprocess_pipeline, rf_model, nn_model, X_test, y_test

# --- Main App ---
st.title("⚡ Customer Churn Prediction Dashboard")
st.write("This app analyzes customer data to predict churn using two models: Random Forest and a Neural Network.")

# Load data and train models
try:
    df = load_data(CSV_FILE_PATH)
    pipeline, rf_model, nn_model, X_test, y_test = train_models(df)

    # --- Model Performance Section ---
    st.header("Model Performance Analysis")

    # Make predictions
    X_test_processed = pipeline.transform(X_test)
    rf_pred = rf_model.predict(X_test_processed)
    rf_proba = rf_model.predict_proba(X_test_processed)[:, 1]
    nn_pred_proba = nn_model.predict(X_test_processed).ravel()
    nn_pred = (nn_pred_proba > 0.5).astype(int)

    # Display accuracy metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest")
        st.metric("Accuracy", f"{accuracy_score(y_test, rf_pred):.2%}")
        st.metric("AUC Score", f"{roc_auc_score(y_test, rf_proba):.3f}")

    with col2:
        st.subheader("Neural Network")
        st.metric("Accuracy", f"{accuracy_score(y_test, nn_pred):.2%}")
        st.metric("AUC Score", f"{roc_auc_score(y_test, nn_pred_proba):.3f}")

    st.markdown("---")

    # Display plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Confusion Matrices
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Random Forest Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    sns.heatmap(confusion_matrix(y_test, nn_pred), annot=True, fmt='d', ax=ax2, cmap='Oranges')
    ax2.set_title('Neural Network Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    st.pyplot(fig)
    plt.clf() # Clear the figure for the next plot

    # ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_pred_proba)
    
    ax_roc.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_proba):.3f})')
    ax_roc.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {roc_auc_score(y_test, nn_pred_proba):.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve Comparison')
    ax_roc.legend()
    st.pyplot(fig_roc)

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

        # Preprocess the input data
        input_processed = pipeline.transform(input_data)

        # Make predictions
        rf_churn_proba = rf_model.predict_proba(input_processed)[0, 1]
        nn_churn_proba = nn_model.predict(input_processed)[0, 0]

        st.sidebar.subheader("Prediction Results")
        
        # Display Random Forest result
        st.sidebar.write("**Random Forest Prediction:**")
        if rf_churn_proba > 0.5:
            st.sidebar.error(f"High Churn Risk ({rf_churn_proba:.2%})")
        else:
            st.sidebar.success(f"Low Churn Risk ({rf_churn_proba:.2%})")
            
        # Display Neural Network result
        st.sidebar.write("**Neural Network Prediction:**")
        if nn_churn_proba > 0.5:
            st.sidebar.error(f"High Churn Risk ({nn_churn_proba:.2%})")
        else:
            st.sidebar.success(f"Low Churn Risk ({nn_churn_proba:.2%})")

except FileNotFoundError:
    st.error(f"Error: The data file was not found at '{CSV_FILE_PATH}'. Please make sure the file is in the same folder as the app and has the correct name.")
except Exception as e:
    st.error(f"An error occurred: {e}")

