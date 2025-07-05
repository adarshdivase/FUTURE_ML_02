# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

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
    prob_adjustments = (
        (df['Age'] > 50) * 0.15 +   # Older customers more likely to churn
        (df['Balance'] == 0) * 0.2 +   # Zero balance customers
        (df['CreditScore'] < 600) * 0.15 +   # Poor credit score
        (df['Tenure'] <= 1) * 0.1   # New customers
    )
    
    final_prob = np.clip(0.2 + prob_adjustments, 0, 0.8) # Base probability 0.2
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

    # Calculate class weights to handle imbalanced data
    classes = np.unique(y_train)
    # Ensure both classes are present to compute weights
    if 0 in classes and 1 in classes:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        scale_pos_weight_val = class_weights[1] / class_weights[0]
    else:
        st.warning("Cannot compute class weights: one of the target classes (0 or 1) is missing in the training data. Setting scale_pos_weight to 1.")
        scale_pos_weight_val = 1
    
    # Create pipeline with class weight balancing
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            eval_metric='logloss',  
            random_state=42,
            scale_pos_weight=scale_pos_weight_val,
            use_label_encoder=False # Suppress future warning for XGBoost
        ))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.2]
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

def plot_feature_importance(pipeline, X_test, y_test):
    """Create feature importance plot using permutation importance - Fixed version"""
    try:
        # Get feature names from the preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Fit the preprocessor to get feature names
        preprocessor.fit(X_test)
        feature_names = preprocessor.get_feature_names_out()
        
        # Transform the test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # Check for zero variance features
        variances = np.var(X_test_transformed, axis=0)
        non_zero_variance_mask = variances > 1e-10
        
        if not np.all(non_zero_variance_mask):
            st.info(f"Found {np.sum(~non_zero_variance_mask)} features with zero or near-zero variance that will be excluded from importance analysis.")
        
        # Filter out zero variance features
        X_test_filtered = X_test_transformed[:, non_zero_variance_mask]
        feature_names_filtered = feature_names[non_zero_variance_mask]
        
        # Create a temporary pipeline with only the classifier for permutation importance
        classifier = pipeline.named_steps['classifier']
        
        # Calculate permutation importance on the filtered data
        perm_importance = permutation_importance(
            classifier, X_test_filtered, y_test, 
            n_repeats=5, random_state=42, n_jobs=1
        )
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names_filtered,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=True)
        
        # Plot with improved alignment
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(importance_df))
        
        # Create bars with better color scheme
        bars = ax.barh(y_pos, importance_df['importance'], 
                       xerr=importance_df['std'], capsize=3, 
                       color='lightblue', edgecolor='steelblue', linewidth=0.5)
        
        # Set y-axis properties
        ax.set_yticks(y_pos)
        
        # Clean up feature names for better readability
        cleaned_features = []
        for feature in importance_df['feature']:
            # Remove prefixes and make more readable
            if feature.startswith('num__'):
                cleaned_features.append(feature.replace('num__', ''))
            elif feature.startswith('cat__'):
                cleaned_features.append(feature.replace('cat__', '').replace('_', ' '))
            else:
                cleaned_features.append(feature.replace('_', ' '))
        
        ax.set_yticklabels(cleaned_features, fontsize=10)
        ax.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Analysis (Non-Zero Variance Features)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Improve grid appearance
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # Add value labels on bars with better positioning
        max_val = importance_df['importance'].max()
        for i, (bar, val, std) in enumerate(zip(bars, importance_df['importance'], importance_df['std'])):
            # Position text based on bar width
            text_x = bar.get_width() + max_val * 0.02
            ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Set x-axis limits to accommodate labels
        ax.set_xlim(0, max_val * 1.2)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Show additional information about excluded features
        if not np.all(non_zero_variance_mask):
            excluded_features = feature_names[~non_zero_variance_mask]
            st.info(f"**Excluded features (zero variance):** {', '.join(excluded_features)}")
        
        return importance_df
        
    except Exception as e:
        st.error(f"Error creating feature importance plot: {e}")
        return None

def analyze_customer_prediction(pipeline, customer_data, feature_names):
    """Analyze why a customer was predicted to churn or not - Fixed version"""
    try:
        # Get the prediction and probability
        pred_proba = pipeline.predict_proba(customer_data)[0, 1]
        pred_class = pipeline.predict(customer_data)[0]
        
        # Get feature importance from the model
        model = pipeline.named_steps['classifier']
        model_feature_importance = model.feature_importances_
        
        # Transform the customer data
        preprocessor = pipeline.named_steps['preprocessor']
        transformed_data = preprocessor.transform(customer_data)
        
        # Get feature names from preprocessor
        preprocessor_feature_names = preprocessor.get_feature_names_out()
        
        # Ensure we have the right number of features
        if len(preprocessor_feature_names) != len(model_feature_importance):
            st.warning(f"Feature count mismatch: preprocessor has {len(preprocessor_feature_names)} features, "
                      f"but model expects {len(model_feature_importance)} features. Using available features.")
            
            # Use the minimum length to avoid index errors
            min_len = min(len(preprocessor_feature_names), len(model_feature_importance))
            preprocessor_feature_names = preprocessor_feature_names[:min_len]
            model_feature_importance = model_feature_importance[:min_len]
            transformed_data = transformed_data[:, :min_len]
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'feature': preprocessor_feature_names,
            'value': transformed_data[0],
            'importance': model_feature_importance
        })
        
        # Sort by importance
        analysis_df = analysis_df.sort_values('importance', ascending=False)
        
        return pred_proba, pred_class, analysis_df
        
    except Exception as e:
        st.error(f"Error analyzing customer prediction: {e}")
        return None, None, None

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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Churned Customers", df['Exited'].sum())
    with col3:
        st.metric("Churn Rate", f"{df['Exited'].mean():.2%}")
    with col4:
        balance_ratio = df['Exited'].value_counts()
        st.metric("Class Balance", f"{balance_ratio[0]}:{balance_ratio[1]}")
    
    # Show class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    df['Exited'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
    ax.set_title('Customer Churn Distribution')
    ax.set_xlabel('Customer Status')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Retained (0)', 'Churned (1)'], rotation=0)
    for i, v in enumerate(df['Exited'].value_counts()):
        ax.text(i, v + 10, str(v), ha='center', va='bottom')
    st.pyplot(fig)
    plt.close(fig)
    
    # Train model
    st.subheader("Model Training")
    with st.spinner("Training XGBoost model with class balancing..."):
        pipeline, X_test, y_test, X_train, preprocessor = train_model(df)
    
    if pipeline:
        st.success("✅ Model trained successfully!")
        
        # Display model info
        model_info = pipeline.named_steps['classifier']
        st.info(f"**Model Details**: XGBoost Classifier with {model_info.n_estimators} estimators, "
                f"max depth: {model_info.max_depth}, learning rate: {model_info.learning_rate}")
        
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Live Prediction", "🧠 Feature Analysis"])

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
            st.header("Feature Importance Analysis")
            
            # Overall feature importance
            st.subheader("Overall Feature Importance")
            with st.spinner("Calculating feature importance..."):
                importance_df = plot_feature_importance(pipeline, X_test, y_test)
            
            if importance_df is not None:
                st.write("**Top 5 Most Important Features:**")
                top_features = importance_df.tail(5).sort_values('importance', ascending=False)
                for idx, row in top_features.iterrows():
                    st.write(f"• **{row['feature']}**: {row['importance']:.3f} ± {row['std']:.3f}")
            
            st.markdown("---")
            
            # Individual customer analysis
            st.subheader("Individual Customer Analysis")
            
            if len(X_test) > 0:
                # Customer selection
                n_customers = min(10, len(X_test))
                customer_options = [f"Customer {i+1}" for i in range(n_customers)]
                selected_customer = st.selectbox("Select a customer for detailed analysis:", customer_options)
                
                customer_idx = customer_options.index(selected_customer)
                
                # Show customer details
                customer_data = X_test.iloc[[customer_idx]]
                actual_churn = y_test.iloc[customer_idx]
                
                # Get feature names for analysis
                preprocessor = pipeline.named_steps['preprocessor']
                feature_names = preprocessor.get_feature_names_out()
                
                # Analyze prediction
                pred_proba, pred_class, analysis_df = analyze_customer_prediction(pipeline, customer_data, feature_names)
                
                if pred_proba is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Customer Details:**")
                        for feature, value in customer_data.iloc[0].items():
                            st.write(f"- {feature}: {value}")
                    
                    with col2:
                        st.write("**Prediction Results:**")
                        st.write(f"- Predicted Churn Probability: {pred_proba:.2%}")
                        st.write(f"- Predicted Class: {'Will Churn' if pred_class else 'Will Stay'}")
                        st.write(f"- Actual Outcome: {'Churned' if actual_churn else 'Retained'}")
                        prediction_correct = pred_class == actual_churn
                        st.write(f"- Prediction Correct: {'✅ Yes' if prediction_correct else '❌ No'}")
                    
                    # Show feature analysis
                    st.subheader("Feature Analysis for Selected Customer")
                    
                    # Plot top contributing features with improved styling
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    top_features = analysis_df.head(10)
                    
                    # Create better color scheme based on feature importance
                    colors = ['darkred' if x > top_features['importance'].mean() else 'steelblue' 
                             for x in top_features['importance']]
                    
                    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Clean up feature names for better readability
                    cleaned_feature_names = []
                    for feature in top_features['feature']:
                        if feature.startswith('num__'):
                            cleaned_feature_names.append(feature.replace('num__', ''))
                        elif feature.startswith('cat__'):
                            cleaned_feature_names.append(feature.replace('cat__', '').replace('_', ' '))
                        else:
                            cleaned_feature_names.append(feature.replace('_', ' '))
                    
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(cleaned_feature_names, fontsize=10)
                    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
                    ax.set_title('Top 10 Feature Contributions for Selected Customer', 
                                fontsize=14, fontweight='bold', pad=20)
                    
                    # Improve grid appearance
                    ax.grid(True, alpha=0.3, axis='x')
                    ax.set_axisbelow(True)
                    
                    # Add value labels with better positioning
                    max_val = top_features['importance'].max()
                    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
                        text_x = bar.get_width() + max_val * 0.02
                        ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                               f'{val:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
                    
                    # Set x-axis limits to accommodate labels
                    ax.set_xlim(0, max_val * 1.2)
                    
                    # Remove top and right spines for cleaner look
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    if pred_proba > 0.5:
                        st.error("🚨 **High Churn Risk Customer**")
                        st.write("**Key risk factors:**")
                        risk_factors = top_features.head(3)
                        for _, row in risk_factors.iterrows():
                            st.write(f"• {row['feature']}: Feature importance {row['importance']:.3f}")
                    else:
                        st.success("✅ **Low Churn Risk Customer**")
                        st.write("**Retention factors:**")
                        retention_factors = top_features.head(3)
                        for _, row in retention_factors.iterrows():
                            st.write(f"• {row['feature']}: Feature importance {row['importance']:.3f}")
            else:
                st.error("No test data available for analysis.")
    else:
        st.error("Failed to train the model. Please check your data and try again.")
else:
    st.error("No data available. Please upload a CSV file or check the sample data generation.")
