# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Churn Prediction",
    page_icon="🚀",
    layout="wide"
)

# --- Configuration ---
CSV_FILE_PATH = "Customer Churn new.csv"

# --- Data Loading and Caching ---
@st.cache_data
def load_data(path):
    """Loads and preprocesses the Bank Churn dataset."""
    df = pd.read_csv(path)
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    return df

@st.cache_resource
def train_models(df):
    """Preprocesses data, trains models, and returns them."""
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    categorical_features = ['Geography', 'Gender']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipelines for each model
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    
    return rf_pipeline, xgb_pipeline, X_test, y_test, X_train

# --- Plotting Functions ---
def plot_model_performance(y_test, rf_pred, xgb_pred, rf_proba, xgb_proba):
    """Plots confusion matrices and ROC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Random Forest Confusion Matrix')
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')

    sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', ax=ax2, cmap='Oranges')
    ax2.set_title('XGBoost Confusion Matrix')
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('Actual')
    st.pyplot(fig)
    plt.clf()

    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    
    ax_roc.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_proba):.3f})')
    ax_roc.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_proba):.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--'); ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate'); ax_roc.set_title('ROC Curve Comparison')
    ax_roc.legend()
    st.pyplot(fig_roc)
    plt.clf()

def plot_feature_importance(pipeline, X_train):
    """Plots feature importance from the Random Forest model."""
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline.named_steps['classifier'].feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Top 15 Feature Importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.pyplot(fig)
    plt.clf()

# --- Main App ---
st.title("🚀 Advanced Bank Customer Churn Prediction")

# Load data and train models
try:
    df = load_data(CSV_FILE_PATH)
    if df is not None:
        rf_pipeline, xgb_pipeline, X_test, y_test, X_train = train_models(df)

        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Live Prediction", "🧠 Model Explanation (SHAP)"])

        with tab1:
            st.header("Comparing Model Performance")
            st.write("Here we compare the performance of a Random Forest model against a more advanced XGBoost model.")
            
            # Make predictions
            rf_pred = rf_pipeline.predict(X_test)
            rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]
            xgb_pred = xgb_pipeline.predict(X_test)
            xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Random Forest")
                st.metric("Accuracy", f"{accuracy_score(y_test, rf_pred):.2%}")
                st.metric("AUC Score", f"{roc_auc_score(y_test, rf_proba):.3f}")
            with col2:
                st.subheader("XGBoost")
                st.metric("Accuracy", f"{accuracy_score(y_test, xgb_pred):.2%}")
                st.metric("AUC Score", f"{roc_auc_score(y_test, xgb_proba):.3f}")
            
            st.markdown("---")
            plot_model_performance(y_test, rf_pred, xgb_pred, rf_proba, xgb_proba)
            
            st.markdown("---")
            st.header("What Drives Churn? (Feature Importance)")
            plot_feature_importance(rf_pipeline, X_train)

        with tab2:
            st.header("🔮 Predict Churn for a New Customer")
            st.write("Enter a customer's details to get a live churn prediction from the best model (XGBoost).")

            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    credit_score = st.slider("Credit Score", 300, 850, 650)
                    geography = st.selectbox("Geography", df['Geography'].unique())
                    gender = st.selectbox("Gender", df['Gender'].unique())
                    age = st.slider("Age", 18, 100, 35)
                    tenure = st.slider("Tenure (years)", 0, 10, 5)
                with col2:
                    balance = st.slider("Balance", 0.0, 250000.0, 0.0)
                    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
                    has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
                    is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
                    estimated_salary = st.slider("Estimated Salary", 0.0, 200000.0, 50000.0)
                
                submitted = st.form_submit_button("Predict Churn")

            if submitted:
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score], 'Geography': [geography], 'Gender': [gender],
                    'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
                    'NumOfProducts': [num_of_products], 'HasCrCard': [has_cr_card],
                    'IsActiveMember': [is_active_member], 'EstimatedSalary': [estimated_salary]
                })

                churn_proba = xgb_pipeline.predict_proba(input_data)[0, 1]
                
                st.subheader("Prediction Result")
                if churn_proba > 0.5:
                    st.error(f"High Churn Risk ({churn_proba:.2%})")
                else:
                    st.success(f"Low Churn Risk ({churn_proba:.2%})")

        with tab3:
            st.header("🧠 Explaining the 'Why' Behind Predictions with SHAP")
            st.write("SHAP (SHapley Additive exPlanations) helps us understand why the model makes a certain decision. The plot below shows how much each feature contributed to pushing the prediction from the base value to the final output.")
            
            # Explain model predictions using SHAP
            preprocessor = xgb_pipeline.named_steps['preprocessor']
            model = xgb_pipeline.named_steps['classifier']
            
            X_test_transformed = preprocessor.transform(X_test)
            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=preprocessor.get_feature_names_out())
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_transformed)

            st.subheader("SHAP Summary Plot")
            st.write("This plot shows the most important features for churn prediction across all customers. Features at the top are most important.")
            fig_shap, ax_shap = plt.subplots()
            shap.summary_plot(shap_values, X_test_transformed_df, plot_type="bar", show=False)
            st.pyplot(fig_shap)
            plt.clf()

            st.subheader("Individual Prediction Explanation")
            st.write("Select a customer from the test set to see a detailed breakdown of their prediction.")
            selected_idx = st.selectbox("Select a customer index to explain:", X_test.index)
            
            if selected_idx is not None:
                shap_values_single = explainer.shap_values(preprocessor.transform(X_test.loc[[selected_idx]]))
                
                fig_force, ax_force = plt.subplots()
                shap.force_plot(explainer.expected_value, shap_values_single, X_test.loc[[selected_idx]], matplotlib=True, show=False)
                st.pyplot(fig_force)
                plt.clf()


except FileNotFoundError:
    st.error(f"Error: The data file was not found at '{CSV_FILE_PATH}'. Please make sure the file is in the same folder as the app and has the correct name.")
except Exception as e:
    st.error(f"An error occurred: {e}")

