# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional Churn Prediction",
    page_icon="🏆",
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
    """Preprocesses data, trains a tuned XGBoost model, and returns it."""
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Define categorical and numerical features
    categorical_features = ['Geography', 'Gender']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Split data before training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Create an XGBoost Pipeline with SMOTE for handling imbalance ---
    xgb_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    # --- Hyperparameter Tuning with GridSearchCV ---
    # Define a focused parameter grid for efficient tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1]
    }

    # Grid search with cross-validation. n_jobs=1 is more stable for deployment.
    grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # The best pipeline found by the grid search
    best_pipeline = grid_search.best_estimator_
    
    return best_pipeline, X_test, y_test, X_train

# --- Plotting Functions ---
def plot_model_performance(y_test, y_pred, y_proba):
    """Plots confusion matrix and ROC curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_proba):.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve'); ax2.legend()
    
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

# --- Main App ---
st.title("🏆 Professional Bank Customer Churn Prediction")

# Load data and train models
try:
    df = load_data(CSV_FILE_PATH)
    if df is not None:
        with st.spinner("Training advanced models with hyperparameter tuning... This may take a few minutes."):
            pipeline, X_test, y_test, X_train = train_model(df)

        tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Live Prediction", "🧠 Model Explanation (SHAP)"])

        with tab1:
            st.header("Tuned XGBoost Model Performance")
            
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            col2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.3f}")
            col3.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            col4.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
            
            st.info(f"**Best Hyperparameters Found:** `{pipeline.named_steps['classifier'].get_params()}`")
            
            st.markdown("---")
            plot_model_performance(y_test, y_pred, y_proba)

        with tab2:
            st.header("🔮 Predict Churn for a New Customer")
            with st.form("prediction_form"):
                credit_score = st.slider("Credit Score", 300, 850, 650)
                geography = st.selectbox("Geography", df['Geography'].unique())
                gender = st.selectbox("Gender", df['Gender'].unique())
                age = st.slider("Age", 18, 100, 35)
                tenure = st.slider("Tenure (years)", 0, 10, 5)
                balance = st.slider("Balance", 0.0, 250000.0, 0.0)
                num_of_products = st.selectbox("Number of Products", df['NumOfProducts'].unique())
                has_cr_card = st.selectbox("Has Credit Card?", df['HasCrCard'].unique(), format_func=lambda x: 'Yes' if x == 1 else 'No')
                is_active_member = st.selectbox("Is Active Member?", df['IsActiveMember'].unique(), format_func=lambda x: 'Yes' if x == 1 else 'No')
                estimated_salary = st.slider("Estimated Salary", 0.0, 200000.0, 50000.0)
                
                submitted = st.form_submit_button("Predict Churn")

            if submitted:
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score], 'Geography': [geography], 'Gender': [gender],
                    'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
                    'NumOfProducts': [num_of_products], 'HasCrCard': [has_cr_card],
                    'IsActiveMember': [is_active_member], 'EstimatedSalary': [estimated_salary]
                })
                churn_proba = pipeline.predict_proba(input_data)[0, 1]
                
                st.subheader("Prediction Result")
                if churn_proba > 0.5:
                    st.error(f"High Churn Risk ({churn_proba:.2%})")
                else:
                    st.success(f"Low Churn Risk ({churn_proba:.2%})")

        with tab3:
            st.header("🧠 Explaining Predictions with SHAP")
            with st.spinner("Calculating SHAP values..."):
                preprocessor = pipeline.named_steps['preprocessor']
                model = pipeline.named_steps['classifier']
                
                X_test_transformed = preprocessor.transform(X_test)
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_transformed)

            st.subheader("Individual Prediction Explanation")
            st.write("Select a customer from the test set to see a detailed breakdown of their prediction.")
            selected_idx = st.selectbox("Select a customer index to explain:", X_test.index)
            
            if selected_idx is not None:
                idx_pos = X_test.index.get_loc(selected_idx)
                
                st.write("The plot below shows how each feature contributed to the final prediction. Red features increase churn risk, blue features decrease it.")
                
                # --- THIS IS THE FIX for the st.shap error ---
                # Generate the plot with Matplotlib and display it with st.pyplot
                fig_force, ax_force = plt.subplots()
                shap.force_plot(explainer.expected_value, shap_values[idx_pos,:], X_test.iloc[idx_pos,:], matplotlib=True, show=False)
                st.pyplot(fig_force, bbox_inches='tight')
                plt.close(fig_force) # Close the figure to free memory

except FileNotFoundError:
    st.error(f"Error: The data file was not found at '{CSV_FILE_PATH}'.")
except Exception as e:
    st.error(f"An error occurred: {e}")
