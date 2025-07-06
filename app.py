# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os # Import os for path handling

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Enhanced page configuration
st.set_page_config(
    page_title="AI-Powered Customer Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        padding-top: 2rem;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        line-height: 1.6;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }

    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8f0fe 100%);
    }

    /* Success/Error/Warning boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .error-box {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .warning-box {
        background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        color: white;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.05);
    }

    /* Form styling */
    .prediction-form {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    /* Progress bar */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Feature importance styling */
    .feature-importance {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Individual prediction styling */
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }

    .prediction-high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }

    .prediction-medium-risk {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
    }

    .prediction-low-risk {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        color: white;
    }

    /* Animation for loading */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .loading-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

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
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(0, 200000, n_samples),
    }

    df = pd.DataFrame(data)

    # Make churn prediction more realistic
    prob_adjustments = (
        (df['Age'] > 45) * 0.15 +
        (df['Balance'] == 0) * 0.2 +
        (df['CreditScore'] < 600) * 0.15 +
        (df['Tenure'] <= 1) * 0.1 +
        (df['NumOfProducts'] > 1) * -0.05 +
        (df['IsActiveMember'] == 0) * 0.1
    )

    final_prob = np.clip(0.2 + prob_adjustments, 0.05, 0.8)
    df['Exited'] = np.random.binomial(1, final_prob)

    return df

@st.cache_data
def load_data(uploaded_file=None, default_file_path='customer_churn_with_added_features.csv'):
    """
    Load data from an uploaded CSV or a default file path.
    If both fail, create sample data.
    """
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {uploaded_file.name} successfully!")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}. Attempting to load default data.")

    if df is None: # If no file uploaded or uploaded file failed
        if os.path.exists(default_file_path):
            try:
                df = pd.read_csv(default_file_path)
                st.info(f"📊 Using default data from {default_file_path}")
            except Exception as e:
                st.error(f"Error loading default data from {default_file_path}: {e}. Creating sample data.")
        else:
            st.warning(f"Default data file not found at {default_file_path}. Creating sample data.")
            
    if df is None: # If all loading attempts failed
        df = create_sample_data()
        st.info("📊 Using automatically generated sample data for demonstration.")

    # Validate required columns in the loaded DataFrame
    required_cols_for_prediction = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    # Check if all prediction features are present and 'Exited' column for training is present
    if not all(col in df.columns for col in required_cols_for_prediction) or 'Exited' not in df.columns:
        missing_features = [col for col in required_cols_for_prediction if col not in df.columns]
        missing_target = "'Exited'" if 'Exited' not in df.columns else ""
        
        error_msg = f"Loaded data is missing required columns for training/prediction: "
        if missing_features:
            error_msg += f"Features: {', '.join(missing_features)}. "
        if missing_target:
            error_msg += f"Target: {missing_target}."
        
        st.warning(f"{error_msg} Falling back to sample data (if not already using it).")
        df = create_sample_data() # Force sample data if loaded data is incomplete
        st.info("📊 Using automatically generated sample data for demonstration due to incomplete data.")

    return df


@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def train_model(df):
    """Train the churn prediction model"""
    if df is None or 'Exited' not in df.columns:
        st.error("Dataset is invalid or missing 'Exited' column.")
        return None, None, None, None, None

    if len(df) < 50:
        st.error("Dataset has too few samples for meaningful training. Please provide at least 50 rows.")
        return None, None, None, None, None

    # Check if 'Exited' column has enough unique values
    if len(df['Exited'].unique()) < 2:
        st.error("The 'Exited' column must contain at least two unique values (0 and 1) for classification.")
        return None, None, None, None, None

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')

    # Ensure stratified split if 'Exited' is not balanced
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError as e:
        st.warning(f"Could not perform stratified split (e.g., only one class in 'Exited'). Falling back to non-stratified split. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    classes = np.unique(y_train)
    if 0 in classes and 1 in classes:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        scale_pos_weight_val = class_weights[1] / class_weights[0]
    else:
        st.warning("Cannot compute class weights: one of the target classes (0 or 1) is missing in the training data. Setting scale_pos_weight to 1.")
        scale_pos_weight_val = 1 # Default to 1 if not balanced or only one class

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            # Removed: use_label_encoder=False, # This parameter is deprecated in newer XGBoost versions
            scale_pos_weight=scale_pos_weight_val
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
        st.exception(e) # Show full traceback
        return None, None, None, None, None

def create_enhanced_metrics_display(y_test, y_pred, y_proba):
    """Create enhanced metrics display with plotly"""
    # Handle cases where y_pred might be all one class (e.g., if model is very biased)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC Score': auc_score
    }

    # Create radar chart for metrics
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name='Model Performance',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2
            )
        ),
        showlegend=True,
        title={
            'text': "Model Performance Metrics",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        height=400
    )

    return fig, metrics

def plot_enhanced_confusion_matrix(y_test, y_pred):
    """Create enhanced confusion matrix with plotly"""
    cm = confusion_matrix(y_test, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Stay', 'Predicted: Churn'],
        y=['Actual: Stay', 'Actual: Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
        hoverongaps=False
    ))

    fig.update_layout(
        title={
            'text': "Confusion Matrix",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )

    return fig

def plot_enhanced_roc_curve(y_test, y_proba):
    """Create enhanced ROC curve with plotly"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='rgb(102, 126, 234)', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title={
            'text': "ROC Curve Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        showlegend=True
    )

    return fig

# Helper function to clean feature names
def clean_feature_name(feature_name):
    if feature_name.startswith('num__'):
        return feature_name.replace('num__', '')
    elif feature_name.startswith('cat__'):
        parts = feature_name.replace('cat__', '').split('_')
        return f"{parts[0].title()}: {parts[1].title()}" if len(parts) > 1 else feature_name.replace('cat__', '').title()
    else:
        return feature_name.replace('_', ' ').title()

def create_feature_importance_chart(pipeline, X_test, y_test):
    """
    Create interactive feature importance charts (built-in and permutation).
    Returns built-in importance figure, combined importance DataFrame, and permutation importance figure.
    """
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']

        # Get feature names after one-hot encoding
        all_feature_names = preprocessor.get_feature_names_out()

        # Built-in feature importance (from XGBoost)
        built_in_importance = classifier.feature_importances_
        
        # Create DataFrame for built-in importance
        built_in_importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': built_in_importance
        })
        built_in_importance_df['cleaned_feature'] = built_in_importance_df['feature'].apply(clean_feature_name)
        built_in_importance_df = built_in_importance_df.sort_values('importance', ascending=True)

        # Create plotly bar chart for built-in importance
        fig_built_in = go.Figure()
        fig_built_in.add_trace(go.Bar(
            x=built_in_importance_df['importance'],
            y=built_in_importance_df['cleaned_feature'],
            orientation='h',
            marker=dict(
                color=built_in_importance_df['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{val:.4f}' for val in built_in_importance_df['importance']],
            textposition='outside'
        ))
        fig_built_in.update_layout(
            title={
                'text': "Built-in Feature Importance (XGBoost)",
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}
            },
            xaxis_title="Feature Importance Score",
            yaxis_title="Features",
            height=max(600, len(built_in_importance_df) * 30),
            showlegend=False
        )

        # Permutation Importance - FIXED VERSION
        # Use the FULL pipeline for permutation importance, not just the classifier
        # This ensures the preprocessing step is included in the permutation
        perm_importance = permutation_importance(
            pipeline, X_test, y_test,  # Use pipeline and original X_test
            n_repeats=10, random_state=42, n_jobs=1,
            scoring='roc_auc'
        )
        
        # Get the original feature names (before preprocessing)
        original_feature_names = X_test.columns
        
        perm_importance_df = pd.DataFrame({
            'feature': original_feature_names,
            'perm_importance': perm_importance.importances_mean,
            'perm_std': perm_importance.importances_std
        })
        perm_importance_df['cleaned_feature'] = perm_importance_df['feature'].apply(clean_feature_name)
        perm_importance_df = perm_importance_df.sort_values('perm_importance', ascending=True)

        # Create plotly bar chart for permutation importance
        fig_perm_importance = go.Figure()
        fig_perm_importance.add_trace(go.Bar(
            x=perm_importance_df['perm_importance'],
            y=perm_importance_df['cleaned_feature'],
            orientation='h',
            error_x=dict(type='data', array=perm_importance_df['perm_std']),
            marker=dict(
                color=perm_importance_df['perm_importance'],
                colorscale='Cividis', # Different color scale for distinction
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{val:.4f}' for val in perm_importance_df['perm_importance']],
            textposition='outside'
        ))
        fig_perm_importance.update_layout(
            title={
                'text': "Permutation Feature Importance (AUC Score Drop)",
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}
            },
            xaxis_title="Importance (Mean AUC Drop)",
            yaxis_title="Features",
            height=max(600, len(perm_importance_df) * 30),
            showlegend=False
        )
        
        # Create a combined importance DataFrame for summary
        # Map built-in importance to original features (approximate mapping)
        combined_importance_df = pd.DataFrame({
            'feature': original_feature_names,
            'cleaned_feature': [clean_feature_name(f) for f in original_feature_names],
            'perm_importance': perm_importance.importances_mean,
            'perm_std': perm_importance.importances_std
        })
        
        # Add built-in importance by summing importance of related transformed features
        built_in_by_original = []
        for orig_feat in original_feature_names:
            # Find all transformed features that correspond to this original feature
            # This handles both 'num__feature' and 'cat__Feature_Category'
            related_features_mask = np.array([orig_feat in f or f.endswith(orig_feat) for f in all_feature_names])
            
            if np.any(related_features_mask):
                # Sum the importance of related transformed features
                related_importance = built_in_importance[related_features_mask].sum()
                built_in_by_original.append(related_importance)
            else:
                built_in_by_original.append(0.0) # If no related transformed feature found
        
        combined_importance_df['importance'] = built_in_by_original
        combined_importance_df = combined_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        return fig_built_in, combined_importance_df, fig_perm_importance
            
    except Exception as e:
        st.error(f"Error creating feature importance chart: {e}")
        st.exception(e)
        return None, None, None # Return three Nones if an error occurs

def create_data_explorer_charts(df):
    """Create interactive charts for data exploration"""
    charts = {}

    # Age distribution
    fig_age = px.histogram(
        df, x='Age', color='Exited',
        title='Age Distribution by Churn Status',
        nbins=20,
        color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
        hover_data={'Age': ':.0f', 'Exited': True}
    )
    fig_age.update_layout(height=400, showlegend=True)
    charts['age'] = fig_age

    # Balance distribution
    fig_balance = px.box(
        df, x='Exited', y='Balance',
        title='Balance Distribution by Churn Status',
        color='Exited',
        color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
        hover_data={'Balance': ':.2f'}
    )
    fig_balance.update_layout(height=400, showlegend=True)
    charts['balance'] = fig_balance

    # Geography distribution
    geography_counts = df.groupby(['Geography', 'Exited']).size().reset_index(name='count')
    fig_geo = px.bar(
        geography_counts, x='Geography', y='count', color='Exited',
        title='Churn Distribution by Geography',
        color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
        barmode='group',
        text_auto=True
    )
    fig_geo.update_layout(height=400, showlegend=True)
    charts['geography'] = fig_geo

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if 'Exited' in numeric_df.columns:
        correlation_matrix = numeric_df.corr()
    else:
        correlation_matrix = numeric_df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig_corr.update_layout(
        title='Feature Correlation Matrix',
        height=500
    )
    charts['correlation'] = fig_corr

    return charts

def analyze_customer_prediction(pipeline, customer_data, X_train_original_columns):
    """Analyze why a customer was predicted to churn or not"""
    try:
        # Ensure customer_data has all columns present in X_train for the preprocessor
        # and in the correct order. Fill missing columns with default/0.
        customer_data_aligned = pd.DataFrame(columns=X_train_original_columns)
        for col in X_train_original_columns:
            if col in customer_data.columns:
                customer_data_aligned[col] = customer_data[col]
            else:
                # Fill missing columns: numerical to 0, categorical to 'Unknown'
                # This logic assumes the preprocessor for 'num' produces numerical columns
                # and for 'cat' produces one-hot encoded columns.
                # Accessing transformers by name and then their feature names is more robust
                num_features_in_preprocessor = [name for name, _, features in pipeline.named_steps['preprocessor'].transformers_ if name == 'num'][0][2] if 'num' in [t[0] for t in pipeline.named_steps['preprocessor'].transformers_] else []
                cat_features_in_preprocessor = [name for name, _, features in pipeline.named_steps['preprocessor'].transformers_ if name == 'cat'][0][2] if 'cat' in [t[0] for t in pipeline.named_steps['preprocessor'].transformers_] else []

                if col in num_features_in_preprocessor:
                    customer_data_aligned[col] = 0.0 # Default numerical to 0
                elif col in cat_features_in_preprocessor:
                    customer_data_aligned[col] = 'Unknown' # OneHotEncoder handles 'unknown' categories
                else:
                    customer_data_aligned[col] = np.nan # Fallback for other types or if not in preprocessor's known features

        customer_data_aligned = customer_data_aligned.fillna(0) # Fill any NaN introduced by alignment

        customer_data_for_prediction = customer_data_aligned.iloc[0:1] # Take only the first row
        
        pred_proba = pipeline.predict_proba(customer_data_for_prediction)[0, 1]
        pred_class = pipeline.predict(customer_data_for_prediction)[0]

        model = pipeline.named_steps['classifier']
        model_feature_importance = model.feature_importances_

        preprocessor = pipeline.named_steps['preprocessor']
        preprocessor_feature_names = preprocessor.get_feature_names_out()

        analysis_df = pd.DataFrame({
            'feature': preprocessor_feature_names,
            'importance': model_feature_importance
        })
        analysis_df['cleaned_feature'] = analysis_df['feature'].apply(clean_feature_name)
        analysis_df = analysis_df.sort_values('importance', ascending=False).reset_index(drop=True)

        return pred_proba, pred_class, analysis_df

    except Exception as e:
        st.error(f"Error analyzing customer prediction: {e}")
        st.exception(e)
        return None, None, None

def create_individual_analysis_chart(analysis_df):
    """Create individual customer analysis chart"""
    top_features = analysis_df.head(10).copy()
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['cleaned_feature'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{val:.4f}' for val in top_features['importance']],
        textposition='outside'
    ))

    fig.update_layout(
        title={
            'text': "Top 10 Feature Contributions for Selected Customer",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Feature Importance",
        yaxis_title="Features",
        height=500,
        showlegend=False
    )

    return fig

# --- Main App ---

# Enhanced header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🎯 AI-Powered Customer Churn Prediction</h1>
    <p class="header-subtitle">
        Harness the power of advanced machine learning to predict customer churn with precision.<br>
        Empowering businesses to retain customers through predictive analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'X_train_cols' not in st.session_state:
    st.session_state.X_train_cols = None

# Enhanced sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration Center")
    
    # Optional File uploader with enhanced styling
    uploaded_file = st.file_uploader(
        "📁 Upload Your Own Data (Optional)",
        type=['csv'],
        help="Upload a CSV file to override the default data. Must contain: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited"
    )
    
    # Load data using the updated function
    # The default_file_path is now set to 'customer_churn_with_added_features.csv'
    df = load_data(uploaded_file=uploaded_file, default_file_path='customer_churn_with_added_features.csv')
    
    # Model information section
    st.markdown("---")
    st.markdown("### 🤖 Model Information")
    
    # Retrain button with enhanced styling
    if st.button("🔄 Retrain Model", use_container_width=True):
        st.session_state.model_trained = False
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

if df is not None and not df.empty:
    # Enhanced dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    # Create enhanced metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churned = df['Exited'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churned:,}</div>
            <div class="metric-label">Churned Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        churn_rate = df['Exited'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1%}</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        balance_counts = df['Exited'].value_counts()
        if 0 in balance_counts and 1 in balance_counts:
            ratio_text = f"{balance_counts[0]}:{balance_counts[1]}"
        else:
            ratio_text = "N/A (single class)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{ratio_text}</div>
            <div class="metric-label">Class Balance (Stay:Churn)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Train model if not already trained
    if not st.session_state.model_trained or st.session_state.pipeline is None:
        with st.spinner("🚀 Training advanced XGBoost model with optimized hyperparameters..."):
            progress_bar = st.progress(0)
            import time
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            pipeline, X_test, y_test, X_train, preprocessor = train_model(df)
            
        if pipeline:
            st.success("✅ Model trained successfully!")
            st.session_state.pipeline = pipeline
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train_cols = X_train.columns.tolist()
            st.session_state.preprocessor = preprocessor
            st.session_state.model_trained = True
        else:
            st.error("❌ Model training failed. Please check your data and try again.")
    
    # Access trained model from session state
    pipeline = st.session_state.get('pipeline')
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    X_train_original_columns = st.session_state.get('X_train_cols')
    
    if pipeline and X_test is not None and y_test is not None and X_train_original_columns is not None:
        # Display model info in sidebar
        model_info = pipeline.named_steps['classifier']
        with st.sidebar:
            st.markdown("### ✨ Model Summary")
            st.markdown(f"""
            - **Estimators**: {model_info.n_estimators}
            - **Max Depth**: {model_info.max_depth}
            - **Learning Rate**: {model_info.learning_rate:.2f}
            """)
            st.success("🎯 Model ready for predictions!")
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Data Explorer",
            "📈 Model Performance",
            "🎯 Live Prediction",
            "📊 Batch Prediction"
        ])
        
        # Tab 1: Data Explorer
        with tab1:
            st.markdown("### 🔍 Data Explorer")
            
            # Create data exploration charts
            charts = create_data_explorer_charts(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(charts['age'], use_container_width=True)
                st.plotly_chart(charts['geography'], use_container_width=True)
            
            with col2:
                st.plotly_chart(charts['balance'], use_container_width=True)
                st.plotly_chart(charts['correlation'], use_container_width=True)
            
            # Enhanced data table
            st.markdown("### 📋 Raw Data Sample")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Data statistics
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(
                df.describe(),
                use_container_width=True
            )
        
        # Tab 2: Model Performance
        with tab2:
            st.markdown("### 📈 Model Performance Analytics")
            
            # Generate predictions for test set
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Create enhanced metrics display
            metrics_fig, metrics_dict = create_enhanced_metrics_display(y_test, y_pred, y_proba)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(metrics_fig, use_container_width=True)
                
                # Enhanced confusion matrix
                cm_fig = plot_enhanced_confusion_matrix(y_test, y_pred)
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                # Enhanced ROC curve
                roc_fig = plot_enhanced_roc_curve(y_test, y_proba)
                st.plotly_chart(roc_fig, use_container_width=True)
                
                # Metrics summary
                st.markdown("### 📊 Performance Metrics")
                for metric, value in metrics_dict.items():
                    st.metric(metric, f"{value:.3f}")
            
            # Feature importance analysis
            st.markdown("### 🎯 Feature Importance Analysis")

            result = create_feature_importance_chart(pipeline, X_test, y_test)

            if result[0] is not None:  # If we got at least the built-in feature importance figure
                feature_importance_fig, importance_df, perm_importance_fig = result
                
                # Display the built-in feature importance
                st.plotly_chart(feature_importance_fig, use_container_width=True)
                
                # If permutation importance was also calculated, display it
                if perm_importance_fig is not None:
                    st.plotly_chart(perm_importance_fig, use_container_width=True)
                
                # Top features summary
                st.markdown("### 🔝 Top 5 Most Important Features")
                if importance_df is not None:
                    # Sort by the built-in importance for consistent display of "Top 5"
                    # Or choose to sort by permutation importance if that's preferred for ranking
                    top_features = importance_df.sort_values('importance', ascending=False).head(5) 
                    for idx, row in top_features.iterrows():
                        if 'perm_importance' in row and not pd.isna(row['perm_importance']): # Check for NaN explicitly
                            st.write(f"**{row['cleaned_feature']}**: Built-in: {row['importance']:.4f}, Permutation: {row['perm_importance']:.4f} ± {row['perm_std']:.4f}")
                        else:
                            st.write(f"**{row['cleaned_feature']}**: {row['importance']:.4f}")
            else:
                st.error("Could not generate feature importance analysis. Please check the model and data.")
        
        # Tab 3: Live Prediction
        with tab3:
            st.markdown("### 🎯 Individual Customer Prediction")
            
            # Enhanced prediction form
            st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💳 Financial Information")
                credit_score = st.slider("Credit Score", 300, 850, 650)
                balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
                num_products = st.slider("Number of Products", 1, 4, 1)
                has_cc = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 75000.0)
            
            
            with col2:
                st.markdown("#### 👤 Personal Information")
                age = st.slider("Age", 18, 80, 40)
                tenure = st.slider("Tenure (years)", 0, 10, 3)
                geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
                gender = st.selectbox("Gender", ['Male', 'Female'])
            
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔮 Predict Churn Risk", use_container_width=True):
                # Create customer data
                customer_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Geography': [geography],
                    'Gender': [gender],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'NumOfProducts': [num_products],
                    'HasCrCard': [has_cc],
                    'IsActiveMember': [is_active],
                    'EstimatedSalary': [estimated_salary]
                })
                
                # Make prediction
                pred_proba, pred_class, analysis_df = analyze_customer_prediction(
                    pipeline, customer_data, X_train_original_columns
                )
                
                if pred_proba is not None:
                    # Display prediction result
                    risk_level = "High Risk" if pred_proba > 0.7 else "Medium Risk" if pred_proba > 0.3 else "Low Risk"
                    risk_color = "prediction-high-risk" if pred_proba > 0.7 else "prediction-medium-risk" if pred_proba > 0.3 else "prediction-low-risk"
                    
                    st.markdown(f"""
                    <div class="prediction-result {risk_color}">
                        <h2>🎯 Prediction Result</h2>
                        <h3>Churn Probability: {pred_proba:.1%}</h3>
                        <h3>Risk Level: {risk_level}</h3>
                        <p>{"⚠️ This customer is likely to churn. Consider retention strategies." if pred_class == 1 else "✅ This customer is likely to stay. Continue providing excellent service."}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Individual analysis chart
                    if analysis_df is not None:
                        st.markdown("### 🔍 Prediction Analysis")
                        individual_chart = create_individual_analysis_chart(analysis_df)
                        st.plotly_chart(individual_chart, use_container_width=True)
                        
                        # Detailed feature analysis
                        st.markdown("### 📊 Feature Contribution Details")
                        st.dataframe(
                            analysis_df.head(10)[['cleaned_feature', 'importance']].rename(columns={'cleaned_feature': 'Feature', 'importance': 'Importance'}),
                            use_container_width=True,
                            hide_index=True
                        )
        
        # Tab 4: Batch Prediction
        with tab4:
            st.markdown("### 📊 Batch Prediction")
            
            # Batch prediction file uploader
            batch_file = st.file_uploader(
                "📁 Upload CSV for Batch Prediction",
                type=['csv'],
                help="Upload a CSV file with customer data for batch prediction. It should contain the same columns as the training data, excluding the 'Exited' column."
            )
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.success(f"✅ Loaded {len(batch_df)} customers for prediction")
                    
                    # --- CRITICAL FIX for IndexError ---
                    # Ensure the batch_df only contains features expected by the pipeline
                    # Drop 'Exited' column if it exists in the uploaded batch file
                    if 'Exited' in batch_df.columns:
                        st.warning("⚠️ 'Exited' column found in uploaded batch file. It will be ignored for prediction.")
                        batch_df_for_prediction = batch_df.drop('Exited', axis=1)
                    else:
                        batch_df_for_prediction = batch_df.copy() # Use a copy to avoid modifying original df
                    
                    # Ensure columns align with training data before prediction
                    missing_cols = set(X_train_original_columns) - set(batch_df_for_prediction.columns)
                    extra_cols = set(batch_df_for_prediction.columns) - set(X_train_original_columns)
                    
                    if missing_cols:
                        st.warning(f"Batch file is missing columns expected by the model: {', '.join(missing_cols)}. These will be treated as absent (e.g., 0 for numerical, new category for categorical).")
                        for col in missing_cols:
                            # This needs to infer type from X_train_original_columns or preprocessor
                            # For simplicity, based on sample data, assume numeric gets 0, others 'Unknown'
                            if col in df.select_dtypes(include=np.number).columns: # Check original df's numeric columns
                                batch_df_for_prediction[col] = 0.0
                            else:
                                batch_df_for_prediction[col] = 'Unknown' # OneHotEncoder handles 'unknown' categories
                    
                    if extra_cols:
                        st.warning(f"Batch file contains extra columns not used by the model: {', '.join(extra_cols)}. These will be ignored.")
                        batch_df_for_prediction = batch_df_for_prediction[[col for col in X_train_original_columns if col in batch_df_for_prediction.columns]]
                    
                    batch_df_for_prediction = batch_df_for_prediction.reindex(columns=X_train_original_columns, fill_value=0)
                    
                    # Display sample data used for prediction
                    st.markdown("### 📋 Sample Data (for prediction)")
                    st.dataframe(batch_df_for_prediction.head(), use_container_width=True)
                    
                    if st.button("🚀 Generate Batch Predictions", use_container_width=True):
                        with st.spinner("🔄 Generating predictions..."):
                            proba_output = pipeline.predict_proba(batch_df_for_prediction)
                            
                            if proba_output.shape[1] > 1:
                                batch_predictions = proba_output[:, 1]
                            else:
                                st.error("Model's predict_proba returned only one column. This might indicate an issue with the model or input data.")
                                st.stop()
                            
                            batch_classes = pipeline.predict(batch_df_for_prediction)
                            
                            batch_df['Churn_Probability'] = batch_predictions
                            batch_df['Churn_Prediction'] = batch_classes
                            batch_df['Risk_Level'] = batch_df['Churn_Probability'].apply(
                                lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.3 else 'Low Risk'
                            )
                            
                            st.markdown("### 🎯 Batch Prediction Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                high_risk = (batch_df['Risk_Level'] == 'High Risk').sum()
                                st.metric("High Risk Customers", high_risk)
                            
                            with col2:
                                medium_risk = (batch_df['Risk_Level'] == 'Medium Risk').sum()
                                st.metric("Medium Risk Customers", medium_risk)
                            
                            with col3:
                                low_risk = (batch_df['Risk_Level'] == 'Low Risk').sum()
                                st.metric("Low Risk Customers", low_risk)
                            
                            csv = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name='batch_predictions.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                            risk_counts = batch_df['Risk_Level'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'])
                            risk_counts = risk_counts.fillna(0)
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title='Risk Level Distribution',
                                color_discrete_map={
                                    'High Risk': '#ff6b6b',
                                    'Medium Risk': '#feca57',
                                    'Low Risk': '#48dbfb'
                                }
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing batch file: {e}. Please ensure your CSV has the correct format and data types.")
                    st.exception(e)
            
            else:
                st.info("📁 Please upload a CSV file for batch prediction")
                
                st.markdown("### 📝 Required CSV Format")
                sample_format = pd.DataFrame({
                    'CreditScore': [650, 720, 580],
                    'Geography': ['France', 'Spain', 'Germany'],
                    'Gender': ['Male', 'Female', 'Male'],
                    'Age': [35, 45, 28],
                    'Tenure': [3, 7, 1],
                    'Balance': [50000, 75000, 0],
                    'NumOfProducts': [1, 2, 1],
                    'HasCrCard': [1, 0, 1],
                    'IsActiveMember': [1, 1, 0],
                    'EstimatedSalary': [75000, 85000, 60000]
                })
                
                st.dataframe(sample_format, use_container_width=True)
                
                csv_sample = sample_format.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sample Format",
                    data=csv_sample,
                    file_name='sample_format.csv',
                    mime='text/csv',
                    use_container_width=True
                )
    
    else:
        st.error("❌ Model training failed or no model available. Please check your data and try again.")

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🚀 AI-Powered Customer Churn Prediction</h3>
    <p>Built with Streamlit, XGBoost, and advanced machine learning techniques</p>
    <p>Empowering businesses to retain customers through predictive analytics</p>
</div>
""", unsafe_allow_html=True)
