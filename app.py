# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="⚡",
    layout="wide"
)

# --- Configuration ---
CSV_FILE_PATH = "Customer Churn new.csv"

# --- Data Loading and Caching ---
@st.cache_data
def load_data(path):
    """Loads and preprocesses the Bank Churn dataset."""
    df = pd.read_csv(path)
    # Drop columns that are identifiers and not useful for modeling
    if 'RowNumber' in df.columns:
        df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    return df

@st.cache_resource
def train_model(df):
    """Preprocesses data, trains a Random Forest model, and returns it."""
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Automatically detect categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Create a preprocessor
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
st.title("⚡ Bank Customer Churn Prediction")
st.write("This app analyzes customer data to predict whether they will exit the bank, using a Random Forest model.")

# --- Live Prediction Section (Sidebar) ---
st.sidebar.header("🔮 Predict Live Churn")
st.sidebar.write("Enter a customer's details to get a churn prediction.")

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
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax1, cmap='Blues')
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_proba):.3f})')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve'); ax2.legend()
        
        st.pyplot(fig)
        plt.close(fig)

        # --- Sidebar Prediction Form ---
        with st.sidebar.form("prediction_form"):
            input_data_dict = {}
            feature_cols = df.drop('Exited', axis=1).columns
            
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].nunique() > 2:
                        min_val, max_val, default_val = float(df[col].min()), float(df[col].max()), float(df[col].median())
                        input_data_dict[col] = st.slider(f"{col}", min_val, max_val, default_val)
                    else:
                        input_data_dict[col] = st.selectbox(f"{col}", df[col].unique(), format_func=lambda x: 'Yes' if x == 1 else 'No')
                else:
                    input_data_dict[col] = st.selectbox(f"{col}", df[col].unique())
            
            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            input_data = pd.DataFrame([input_data_dict])
            churn_proba = pipeline.predict_proba(input_data)[0, 1]
            
            st.sidebar.subheader("Prediction Result")
            if churn_proba > 0.5:
                st.sidebar.error(f"High Churn Risk ({churn_proba:.2%})")
            else:
                st.sidebar.success(f"Low Churn Risk ({churn_proba:.2%})")

except FileNotFoundError:
    st.error(f"Error: The data file was not found at '{CSV_FILE_PATH}'.")
except Exception as e:
    st.error(f"An error occurred: {e}")
