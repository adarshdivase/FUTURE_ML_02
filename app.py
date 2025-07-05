
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

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Professional Churn Prediction",
    page_icon="🏆",
    layout="wide"
)

CSV_FILE_PATH = "customer_churn_cleaned.csv"

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model(df):
    if df is None or 'Exited' not in df.columns:
        st.error("Dataset is invalid or missing 'Exited' column.")
        return None, None, None, None

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    categorical_features = ['Geography', 'Gender']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
    try:
        grid.fit(X_train, y_train)
        return grid.best_estimator_, X_test, y_test, X_train
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None, None, None

def plot_model_performance(y_test, y_pred, y_proba):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curve'); ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.legend()

    st.pyplot(fig)
    plt.close(fig)

st.title("🏆 Professional Bank Customer Churn Prediction")

df = load_data(CSV_FILE_PATH)

if df is not None:
    with st.spinner("Training model..."):
        pipeline, X_test, y_test, X_train = train_model(df)

    if pipeline:
        tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Live Prediction", "🧠 SHAP Explanation"])

        with tab1:
            st.header("Model Metrics")
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            col2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
            col3.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            col4.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")

            st.markdown("---")
            plot_model_performance(y_test, y_pred, y_proba)

        with tab2:
            st.header("Live Prediction")
            with st.form("prediction_form"):
                credit_score = st.slider("Credit Score", 300, 850, 650)
                geography = st.selectbox("Geography", df['Geography'].unique())
                gender = st.selectbox("Gender", df['Gender'].unique())
                age = st.slider("Age", 18, 100, 35)
                tenure = st.slider("Tenure", 0, 10, 5)
                balance = st.slider("Balance", 0.0, 250000.0, 0.0)
                estimated_salary = st.slider("Estimated Salary", 0.0, 200000.0, 50000.0)
                submitted = st.form_submit_button("Predict")

            if submitted:
                input_df = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Geography': [geography],
                    'Gender': [gender],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'EstimatedSalary': [estimated_salary]
                })
                pred = pipeline.predict_proba(input_df)[0, 1]
                if pred > 0.5:
                    st.error(f"High Churn Risk: {pred:.2%}")
                else:
                    st.success(f"Low Churn Risk: {pred:.2%}")

        with tab3:
            st.header("SHAP Explanation")
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['classifier']
            transformed = preprocessor.transform(X_test)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
            feature_names = preprocessor.get_feature_names_out()

            st.subheader("Select a customer for explanation")
            display_map = {i: f"Customer {i}" for i in X_test.index}
            selected = st.selectbox("Select customer:", list(display_map.values()))
            index = list(display_map.values()).index(selected)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.force_plot(explainer.expected_value, shap_values[index], 
                            pd.DataFrame(transformed[index].reshape(1, -1), columns=feature_names),
                            matplotlib=True, show=False, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
