# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Professional Churn Prediction",
    page_icon="🏆",
    layout="wide"
)

# --- Helper Functions ---

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
            # Basic validation for expected columns
            required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Uploaded CSV must contain all required columns: {', '.join(required_cols)}. Using sample data.")
                return create_sample_data()
            return df
        except FileNotFoundError:
            st.warning(f"CSV file '{path}' not found. Using sample data for demonstration.")
            return create_sample_data()
        except Exception as e:
            st.error(f"Error loading data: {e}. Using sample data for demonstration.")
            return create_sample_data()
    else:
        return create_sample_data()

@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None}) # Invalidate cache if df changes
def train_model(df):
    """Train the churn prediction model"""
    if df is None or 'Exited' not in df.columns:
        st.error("Dataset is invalid or missing 'Exited' column.")
        return None, None, None, None, None

    # Check for sufficient data
    if len(df) < 50: # Arbitrary minimum for meaningful training
        st.error("Dataset has too few samples for meaningful training. Please provide at least 50 rows.")
        return None, None, None, None, None

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Ensure target variable has at least two classes
    if len(y.unique()) < 2:
        st.error("The 'Exited' column must contain at least two unique values (0 and 1) for classification.")
        return None, None, None, None, None

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough') # Keep other columns if any, though not expected here

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Calculate class weights to handle imbalanced data
    classes = np.unique(y_train)
    # Handle cases where a class might be missing in y_train (though stratify helps prevent this)
    if 0 in classes and 1 in classes:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        scale_pos_weight_val = class_weights[1] / class_weights[0]
    else:
        st.warning("Cannot compute class weights: one of the target classes (0 or 1) is missing in the training data. Setting scale_pos_weight to 1.")
        scale_pos_weight_val = 1 # No imbalance or only one class present

    # Create pipeline with class weight balancing
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            eval_metric='logloss',  # Use 'logloss' for binary classification
            random_state=42,
            use_label_encoder=False, # Suppress warning for older versions
            scale_pos_weight=scale_pos_weight_val # Handle imbalanced data
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
        st.error(f"Model training failed: {e}. Please check your data and parameters.")
        return None, None, None, None, None

def plot_model_performance(y_test, y_pred, y_proba):
    """Plot confusion matrix and ROC curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # Increased figure size
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues', cbar=False,
                xticklabels=['Predicted Not Churn', 'Predicted Churn'],
                yticklabels=['Actual Not Churn', 'Actual Churn'])
    ax1.set_title('Confusion Matrix', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', linewidth=2, color='darkorange')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box') # Make ROC curve square

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_feature_importance(pipeline, X_test, y_test):
    """Create feature importance plot using permutation importance - Fixed alignment version"""
    try:
        # Get feature names from the preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Fit the preprocessor to get feature names
        # Note: preprocessor might already be fitted within the pipeline, but explicit fit here ensures
        # get_feature_names_out() works correctly on the data used for importance.
        preprocessor.fit(X_test) 
        all_feature_names = preprocessor.get_feature_names_out()
        
        # Transform the test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # Check for zero variance features in the transformed data
        variances = np.var(X_test_transformed, axis=0)
        non_zero_variance_mask = variances > 1e-10 # Use a small epsilon for floating-point comparisons
        
        # Filter X_test_transformed and feature names to include only non-zero variance features
        X_test_filtered = X_test_transformed[:, non_zero_variance_mask]
        feature_names_filtered = all_feature_names[non_zero_variance_mask]
        
        # Create a temporary pipeline with only the classifier for permutation importance
        classifier = pipeline.named_steps['classifier']
        
        # Calculate permutation importance on the filtered data
        perm_importance = permutation_importance(
            classifier, X_test_filtered, y_test, 
            n_repeats=10, random_state=42, n_jobs=1 # Reduced n_repeats to 10 for faster execution in Streamlit
        )
        
        # Create a dictionary to store importance and std for all original preprocessor features,
        # initializing with 0.0 for those not covered by permutation_importance.
        feature_importance_data = {name: {'importance': 0.0, 'std': 0.0} for name in all_feature_names}

        # Populate with actual permutation importance results
        # Match filtered feature names to their importance values
        for i, feature_name_f in enumerate(feature_names_filtered):
            feature_importance_data[feature_name_f]['importance'] = perm_importance.importances_mean[i]
            feature_importance_data[feature_name_f]['std'] = perm_importance.importances_std[i]

        # Prepare lists for the DataFrame, ensuring all_feature_names are included
        features_for_df = []
        importances_for_df = []
        stds_for_df = []

        for feature_name in all_feature_names:
            features_for_df.append(feature_name)
            importances_for_df.append(feature_importance_data[feature_name]['importance'])
            stds_for_df.append(feature_importance_data[feature_name]['std'])
        
        # Warn if there was a discrepancy in counts (i.e., some features had 0 importance)
        if len(all_feature_names) != len(feature_names_filtered):
            st.warning(f"Warning: {len(all_feature_names) - len(feature_names_filtered)} features had zero or near-zero variance in the test set "
                       f"and were excluded from permutation importance calculation. Their importance is set to 0 in the plot. "
                       f"Excluded: {', '.join(all_feature_names[~non_zero_variance_mask])}")

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': features_for_df,
            'importance': importances_for_df,
            'std': stds_for_df
        }).sort_values('importance', ascending=True) # Sort ascending for horizontal bar plot
        
        # Clean up feature names for better readability in the plot
        cleaned_features = []
        for feature in importance_df['feature']:
            if feature.startswith('num__'):
                cleaned_features.append(feature.replace('num__', ''))
            elif feature.startswith('cat__'):
                cleaned_features.append(feature.replace('cat__', '').replace('_', ' ').title())
            else:
                cleaned_features.append(feature.replace('_', ' ').title())
        
        # Create the plot with improved dimensions
        fig, ax = plt.subplots(figsize=(14, max(8, len(importance_df) * 0.4))) # Dynamic height
        
        # Create color gradient based on importance values
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance_df))) # Red-Yellow-Blue reversed
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                       xerr=importance_df['std'], capsize=4, 
                       color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Set y-axis properties with proper alignment
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(cleaned_features, fontsize=11, ha='right') # Align labels right
        
        # Improve x-axis
        ax.set_xlabel('Permutation Importance', fontsize=14, fontweight='bold')
        ax.set_title('Overall Feature Importance Analysis\n(Higher values indicate more important features)', 
                     fontsize=16, fontweight='bold', pad=25)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_axisbelow(True) # Grid behind bars
        
        # Calculate proper x-axis limits
        max_val = importance_df['importance'].max()
        max_error = importance_df['std'].max()
        x_limit = max_val + max_error + (max_val * 0.25) # 25% padding for labels
        ax.set_xlim(0, x_limit if x_limit > 0 else 0.1) # Ensure positive limit
        
        # Add value labels with better positioning
        for i, (bar, val, std) in enumerate(zip(bars, importance_df['importance'], importance_df['std'])):
            # Position text at the end of the error bar
            text_x = val + std + (max_val * 0.02) if val > 0 else (max_val * 0.02) # Adjusted for zero importance
            ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', ha='left', 
                    fontsize=10, fontweight='bold', color='black')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Additional padding for y-axis labels
        plt.subplots_adjust(left=0.25)
        
        st.pyplot(fig)
        plt.close(fig)
        
        return importance_df
        
    except Exception as e:
        st.error(f"Error creating feature importance plot: {e}")
        return None

def analyze_customer_prediction(pipeline, customer_data, feature_names):
    """Analyze why a customer was predicted to churn or not - Fixed alignment version"""
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
            st.warning(f"Feature count mismatch in individual analysis: preprocessor has {len(preprocessor_feature_names)} features, "
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

def plot_individual_customer_analysis(analysis_df, customer_data):
    """Create improved individual customer analysis plot"""
    # Get top 10 features for individual analysis
    top_features = analysis_df.head(10).copy()
    
    # Clean up feature names for better readability
    cleaned_feature_names = []
    for feature in top_features['feature']:
        if feature.startswith('num__'):
            cleaned_feature_names.append(feature.replace('num__', ''))
        elif feature.startswith('cat__'):
            cleaned_feature_names.append(feature.replace('cat__', '').replace('_', ' ').title())
        else:
            cleaned_feature_names.append(feature.replace('_', ' ').title())
    
    # Create the plot with improved dimensions
    fig, ax = plt.subplots(figsize=(14, max(8, len(top_features) * 0.5))) # Dynamic height
    
    # Create color scheme based on feature importance
    importance_normalized = (top_features['importance'] - top_features['importance'].min()) / (top_features['importance'].max() - top_features['importance'].min())
    colors = plt.cm.RdYlGn_r(importance_normalized) # Red-Yellow-Green reversed
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                    color=colors, edgecolor='black', linewidth=0.7, alpha=0.8)
    
    # Set y-axis properties with proper alignment
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(cleaned_feature_names, fontsize=12, ha='right') # Align labels right
    
    # Improve x-axis
    ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
    ax.set_title('Top 10 Feature Contributions for Selected Customer\n(Higher values indicate stronger influence on prediction)', 
                 fontsize=16, fontweight='bold', pad=25)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_axisbelow(True) # Grid behind bars
    
    # Calculate proper x-axis limits
    max_val = top_features['importance'].max()
    x_limit = max_val + (max_val * 0.25) # 25% padding for labels
    ax.set_xlim(0, x_limit if x_limit > 0 else 0.1) # Ensure positive limit
    
    # Add value labels with better positioning
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        text_x = val + (max_val * 0.02) # Position text at the end of the bar
        ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', ha='left', 
                fontsize=11, fontweight='bold', color='black')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Additional padding for y-axis labels
    plt.subplots_adjust(left=0.3)
    
    return fig

# --- Main App ---

st.title("🏆 Professional Bank Customer Churn Prediction")
st.markdown("""
Welcome to the advanced customer churn prediction application.
Upload your customer data or use our sample dataset to train a powerful XGBoost model,
analyze its performance, predict churn risk for individual customers, and understand key factors driving churn.
""")

st.sidebar.header("⚙️ Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file (optional)", type=['csv'])

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("CSV file uploaded successfully!")
else:
    st.sidebar.info("No file uploaded. Using sample data for demonstration.")
    df = load_data()

# Initialize session state for model retraining
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

# Retrain button
if st.sidebar.button("🔄 Retrain Model", use_container_width=True):
    st.session_state.model_trained = False # Invalidate flag to force retraining
    st.cache_resource.clear() # Clear model cache
    st.cache_data.clear() # Clear data cache (if data was reloaded/changed)
    st.experimental_rerun() # Rerun the app to trigger retraining

if df is not None and not df.empty:
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
        if 0 in balance_ratio and 1 in balance_ratio:
            st.metric("Class Balance (0:1)", f"{balance_ratio[0]}:{balance_ratio[1]}")
        else:
            st.metric("Class Balance", "N/A (single class)")
    
    st.markdown("---")

    # Tabs for different sections
    tab0, tab1, tab2, tab3 = st.tabs(["🔍 Data Explorer", "📊 Model Performance", "🔮 Live Prediction", "🧠 Feature Analysis"])

    with tab0:
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown("Understand the characteristics and distributions of your customer data.")

        st.subheader("Dataset Sample")
        st.dataframe(df.head())

        st.subheader("Descriptive Statistics")
        st.write(df.describe().T)

        st.subheader("Numerical Feature Distributions")
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        if 'Exited' in numerical_features:
            numerical_features.remove('Exited') # Don't plot target as numerical distribution

        num_cols = st.columns(3)
        for i, col in enumerate(numerical_features):
            with num_cols[i % 3]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                st.pyplot(fig)
                plt.close(fig)
        
        st.subheader("Categorical Feature Distributions vs. Churn")
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = st.columns(2)
        for i, col in enumerate(categorical_features):
            with cat_cols[i % 2]:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.countplot(data=df, x=col, hue='Exited', palette='viridis', ax=ax)
                ax.set_title(f'{col} Distribution by Churn Status')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.legend(title='Exited', labels=['Retained', 'Churned'])
                st.pyplot(fig)
                plt.close(fig)

    # Train model (only if not already trained or if retraining requested)
    if not st.session_state.model_trained or st.session_state.pipeline is None:
        st.info("Training a new XGBoost model with optimized hyperparameters. This may take a moment...")
        with st.spinner("Training XGBoost model with class balancing and GridSearchCV..."):
            pipeline, X_test, y_test, X_train, preprocessor = train_model(df)
        if pipeline:
            st.success("✅ Model trained successfully!")
            st.session_state.pipeline = pipeline
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train
            st.session_state.preprocessor = preprocessor
            st.session_state.model_trained = True
        else:
            st.error("Model training failed. Please check the data and try retraining.")
            st.session_state.model_trained = False # Keep flag false if training failed

    # Access trained model from session state
    pipeline = st.session_state.get('pipeline')
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    X_train = st.session_state.get('X_train')
    preprocessor = st.session_state.get('preprocessor')

    if pipeline and X_test is not None and y_test is not None:
        # Display model info in sidebar
        model_info = pipeline.named_steps['classifier']
        st.sidebar.markdown("---")
        st.sidebar.subheader("✨ Model Summary")
        st.sidebar.info(f"**Best Estimators**: {model_info.n_estimators}\n"
                        f"**Max Depth**: {model_info.max_depth}\n"
                        f"**Learning Rate**: {model_info.learning_rate:.2f}")
        st.sidebar.success("Model ready for predictions!")
        
        with tab1:
            st.header("Model Performance Metrics")
            st.markdown("Evaluate the trained model's effectiveness in predicting customer churn.")
            
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            with col2:
                st.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.3f}")
            with col3:
                st.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            with col4:
                st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
            with col5:
                st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

            st.markdown("---")
            plot_model_performance(y_test, y_pred, y_proba)

        with tab2:
            st.header("Live Churn Prediction")
            st.markdown("Enter customer details below to get an instant churn risk prediction.")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    credit_score = st.slider("Credit Score", 300, 850, 650, help="Credit score of the customer.")
                    geography = st.selectbox("Geography", df['Geography'].unique(), help="Customer's country of residence.")
                    gender = st.selectbox("Gender", df['Gender'].unique(), help="Customer's gender.")
                    age = st.slider("Age", 18, 100, 35, help="Age of the customer.")
                
                with col2:
                    tenure = st.slider("Tenure (Years)", 0, 10, 5, help="Number of years the customer has been with the bank.")
                    # Dynamic max values for sliders based on loaded data
                    max_balance = float(df['Balance'].max()) if not df['Balance'].empty else 250000.0
                    balance = st.slider("Balance ($)", 0.0, max_balance, 0.0, help="Customer's account balance.")
                    max_salary = float(df['EstimatedSalary'].max()) if not df['EstimatedSalary'].empty else 200000.0
                    estimated_salary = st.slider("Estimated Salary ($)", 0.0, max_salary, 50000.0, help="Estimated salary of the customer.")
                
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
                
                st.subheader("Prediction Result:")
                # Display prediction with color coding and enhanced messages
                if pred_proba > 0.65: # Higher threshold for 'High Risk'
                    st.error(f"🚨 **High Churn Risk: {pred_proba:.2%}**")
                    st.write("This customer has a **significant probability of churning**. Immediate and targeted retention strategies are highly recommended, such as personalized offers, direct outreach, or addressing specific pain points (e.g., high fees, poor service).")
                elif pred_proba > 0.35: # Adjusted threshold for 'Medium Risk'
                    st.warning(f"⚠️ **Medium Churn Risk: {pred_proba:.2%}**")
                    st.write("This customer shows a **moderate risk of churning**. Proactive monitoring and general retention campaigns (e.g., loyalty programs, check-in calls) could be beneficial to prevent them from leaving.")
                else:
                    st.success(f"✅ **Low Churn Risk: {pred_proba:.2%}**")
                    st.write("This customer is **likely to remain with the bank**. Continue to provide excellent service and consider nurturing their relationship for long-term loyalty.")

        with tab3:
            st.header("Feature Importance Analysis")
            st.markdown("Understand which factors are most influential in predicting customer churn.")
            
            # Overall feature importance
            st.subheader("Overall Feature Importance")
            with st.spinner("Calculating overall feature importance (this may take a moment for larger datasets)..."):
                importance_df = plot_feature_importance(pipeline, X_test, y_test)
            
            if importance_df is not None:
                st.write("**Top 5 Most Important Features:**")
                top_features = importance_df.tail(5).sort_values('importance', ascending=False)
                for idx, row in top_features.iterrows():
                    # Clean up feature names for display
                    feature_name = row['feature']
                    if feature_name.startswith('num__'):
                        feature_name = feature_name.replace('num__', '')
                    elif feature_name.startswith('cat__'):
                        feature_name = feature_name.replace('cat__', '').replace('_', ' ').title()
                    else:
                        feature_name = feature_name.replace('_', ' ').title()
                    
                    st.write(f"• **{feature_name}**: {row['importance']:.4f} ± {row['std']:.4f}")
            
            st.markdown("---")
            
            # Individual customer analysis
            st.subheader("Individual Customer Prediction Insight")
            st.markdown("Select a customer from the test set to see their details and the top features influencing their prediction.")
            
            if len(X_test) > 0:
                # Customer selection with more descriptive options
                customer_options_df = X_test.copy()
                customer_options_df['Actual_Churn'] = y_test
                customer_options_df['Display_Name'] = [
                    f"Customer {i+1} (Actual: {'Churned' if actual else 'Retained'}, Age: {age}, Balance: ${balance:,.0f})"
                    for i, (actual, age, balance) in enumerate(zip(customer_options_df['Actual_Churn'], customer_options_df['Age'], customer_options_df['Balance']))
                ]
                
                selected_customer_display = st.selectbox(
                    "Select a customer for detailed analysis:", 
                    customer_options_df['Display_Name']
                )
                
                customer_idx = customer_options_df[customer_options_df['Display_Name'] == selected_customer_display].index[0]
                
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
                        st.write("**Selected Customer Details:**")
                        for feature, value in customer_data.iloc[0].items():
                            if isinstance(value, (int, float)):
                                st.write(f"- **{feature}**: {value:,.2f}" if feature in ['Balance', 'EstimatedSalary'] else f"- **{feature}**: {value}")
                            else:
                                st.write(f"- **{feature}**: {value}")
                    
                    with col2:
                        st.write("**Prediction Results:**")
                        st.write(f"- Predicted Churn Probability: **{pred_proba:.2%}**")
                        st.write(f"- Predicted Outcome: **{'Will Churn' if pred_class else 'Will Stay'}**")
                        st.write(f"- Actual Outcome: **{'Churned' if actual_churn else 'Retained'}**")
                        prediction_correct = pred_class == actual_churn
                        st.write(f"- Prediction Correct: {'✅ Yes' if prediction_correct else '❌ No'}")
                    
                    # Show feature analysis with improved plot
                    st.subheader("Feature Analysis for Selected Customer")
                    with st.spinner("Generating individual feature analysis plot..."):
                        fig = plot_individual_customer_analysis(analysis_df, customer_data)
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Interpretation with better formatting and actionable recommendations
                    st.subheader("Interpretation")
                    
                    # Get top 3 features for interpretation based on model's feature_importances_
                    top_3_features = analysis_df.head(3)
                    
                    if pred_proba > 0.5:
                        st.error("🚨 **High Churn Risk Customer**")
                        st.write("This customer is at **high risk of churning**. Key factors contributing to this prediction include:")
                        for idx, row in top_3_features.iterrows():
                            feature_name = row['feature']
                            if feature_name.startswith('num__'):
                                feature_name = feature_name.replace('num__', '')
                            elif feature_name.startswith('cat__'):
                                feature_name = feature_name.replace('cat__', '').replace('_', ' ').title()
                            else:
                                feature_name = feature_name.replace('_', ' ').title()
                            
                            st.write(f"• **{feature_name}**: (Importance: {row['importance']:.4f})")
                            
                        st.write("\n**Recommended Actions:**")
                        st.write("- **Immediate Contact:** Reach out with a personalized offer or to address recent issues.")
                        st.write("- **Deep Dive Analysis:** Investigate their recent activity and interactions for specific pain points.")
                        st.write("- **Senior Account Manager Review:** Assign a dedicated manager to understand and resolve their concerns.")
                        
                    else:
                        st.success("✅ **Low Churn Risk Customer**")
                        st.write("This customer is **likely to remain with the bank**. Factors contributing to this positive prediction include:")
                        for idx, row in top_3_features.iterrows():
                            feature_name = row['feature']
                            if feature_name.startswith('num__'):
                                feature_name = feature_name.replace('num__', '')
                            elif feature_name.startswith('cat__'):
                                feature_name = feature_name.replace('cat__', '').replace('_', ' ').title()
                            else:
                                feature_name = feature_name.replace('_', ' ').title()
                            
                            st.write(f"• **{feature_name}**: (Importance: {row['importance']:.4f})")
                            
                        st.write("\n**Recommended Actions:**")
                        st.write("- **Maintain Service Excellence:** Continue providing high-quality service to ensure satisfaction.")
                        st.write("- **Engagement Opportunities:** Offer relevant new products/services or loyalty program benefits.")
                        st.write("- **Feedback Collection:** Periodically solicit feedback to understand what keeps them satisfied.")
            else:
                st.error("No test data available for individual customer analysis. Please ensure data is loaded and model is trained.")
    else:
        st.error("Failed to train the model. Please check your data and try again.")
else:
    st.error("No data available. Please upload a CSV file or check the sample data generation.")

