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

# Generate sample data if CSV doesn't exist
@st.cache_data
def create_sample_data():
    """Create sample churn data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'CreditScore': np.random.randint(300, 850, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'EstimatedSalary': np.random.uniform(0, 200000, n_samples),
    }
    
    # Create somewhat realistic churn patterns
    df = pd.DataFrame(data)
    
    # Higher churn probability for certain conditions
    churn_prob = 0.2  # Base probability
    
    # Increase churn probability based on features
    prob_adjustments = (
        (df['Age'] > 50) * 0.15 +  # Older customers more likely to churn
        (df['Balance'] == 0) * 0.2 +  # Zero balance customers
        (df['CreditScore'] < 600) * 0.15 +  # Poor credit score
        (df['Tenure'] <= 1) * 0.1  # New customers
    )
    
    final_prob = np.clip(churn_prob + prob_adjustments, 0, 0.8)
    df['Exited'] = np.random.binomial(1, final_prob)
    
    return df

@st.cache_data
def load_data(path=None):
    """Load data from CSV or create sample data"""
    if path:
        try:
            df = pd.read_csv(path)
            return df
        except FileNotFoundError:
            st.warning(f"CSV file '{path}' not found. Using sample data for demonstration.")
            return create_sample_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return create_sample_data()
    else:
        return create_sample_data()

@st.cache_resource
def train_model(df):
    """Train the churn prediction model"""
    if df is None or 'Exited' not in df.columns:
        st.error("Dataset is invalid or missing 'Exited' column.")
        return None, None, None, None, None

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
        return grid.best_estimator_, X_test, y_test, X_train, preprocessor
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None, None, None, None

def plot_model_performance(y_test, y_pred, y_proba):
    """Plot confusion matrix and ROC curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

def plot_shap_explanation(pipeline, X_test, customer_idx):
    """Create SHAP explanation plot for a specific customer"""
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']
        
        # Transform the data
        X_transformed = preprocessor.transform(X_test)
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        
        # Create waterfall plot for the selected customer
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use waterfall plot instead of force plot for better compatibility
        shap_exp = shap.Explanation(
            values=shap_values[customer_idx],
            base_values=explainer.expected_value,
            data=X_transformed[customer_idx],
            feature_names=feature_names
        )
        
        shap.waterfall_plot(shap_exp, show=False)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        st.error(f"Error creating SHAP explanation: {e}")
        return False

# Main app
st.title("🏆 Professional Bank Customer Churn Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (optional)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
else:
    st.info("No file uploaded. Using sample data for demonstration.")
    df = load_data()

if df is not None:
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Churned Customers", df['Exited'].sum())
    with col3:
        st.metric("Churn Rate", f"{df['Exited'].mean():.2%}")
    
    # Train model
    with st.spinner("Training model..."):
        pipeline, X_test, y_test, X_train, preprocessor = train_model(df)

    if pipeline:
        tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Live Prediction", "🧠 SHAP Explanation"])

        with tab1:
            st.header("Model Performance Metrics")
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            with col2:
                st.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.3f}")
            with col3:
                st.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            with col4:
                st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")

            st.markdown("---")
            plot_model_performance(y_test, y_pred, y_proba)

        with tab2:
            st.header("Live Churn Prediction")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    credit_score = st.slider("Credit Score", 300, 850, 650)
                    geography = st.selectbox("Geography", df['Geography'].unique())
                    gender = st.selectbox("Gender", df['Gender'].unique())
                    age = st.slider("Age", 18, 100, 35)
                
                with col2:
                    tenure = st.slider("Tenure (Years)", 0, 10, 5)
                    balance = st.slider("Balance ($)", 0.0, 250000.0, 0.0)
                    estimated_salary = st.slider("Estimated Salary ($)", 0.0, 200000.0, 50000.0)
                
                submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

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
                
                pred_proba = pipeline.predict_proba(input_df)[0, 1]
                
                # Display prediction with color coding
                if pred_proba > 0.5:
                    st.error(f"🚨 High Churn Risk: {pred_proba:.2%}")
                    st.write("**Recommendation:** Immediate intervention required!")
                elif pred_proba > 0.3:
                    st.warning(f"⚠️ Medium Churn Risk: {pred_proba:.2%}")
                    st.write("**Recommendation:** Monitor closely and consider retention strategies.")
                else:
                    st.success(f"✅ Low Churn Risk: {pred_proba:.2%}")
                    st.write("**Recommendation:** Customer is likely to stay.")

        with tab3:
            st.header("SHAP Model Explanation")
            
            if len(X_test) > 0:
                st.subheader("Feature Importance for Individual Predictions")
                
                # Customer selection
                n_customers = min(10, len(X_test))
                customer_options = [f"Customer {i+1}" for i in range(n_customers)]
                selected_customer = st.selectbox("Select a customer for detailed explanation:", customer_options)
                
                customer_idx = customer_options.index(selected_customer)
                
                # Show customer details
                customer_data = X_test.iloc[customer_idx]
                actual_churn = y_test.iloc[customer_idx]
                predicted_prob = pipeline.predict_proba(X_test.iloc[[customer_idx]])[0, 1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Customer Details:**")
                    for feature, value in customer_data.items():
                        st.write(f"- {feature}: {value}")
                
                with col2:
                    st.write("**Prediction Results:**")
                    st.write(f"- Predicted Churn Probability: {predicted_prob:.2%}")
                    st.write(f"- Actual Outcome: {'Churned' if actual_churn else 'Retained'}")
                    prediction_correct = (predicted_prob > 0.5) == actual_churn
                    st.write(f"- Prediction Correct: {'✅ Yes' if prediction_correct else '❌ No'}")
                
                st.markdown("---")
                
                # Generate SHAP explanation
                if st.button("Generate SHAP Explanation"):
                    with st.spinner("Generating explanation..."):
                        success = plot_shap_explanation(pipeline, X_test, customer_idx)
                        if success:
                            st.success("SHAP explanation generated successfully!")
            else:
                st.error("No test data available for SHAP explanation.")
    else:
        st.error("Failed to train the model. Please check your data and try again.")
else:
    st.error("No data available. Please upload a CSV file or check the sample data generation.")
