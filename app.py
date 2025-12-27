import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.feature_engineering import FeatureEngineer
from src.explainability import Explainability

# Page Config
st.set_page_config(
    page_title=cfg.PAGE_TITLE,
    page_icon=cfg.PAGE_ICON,
    layout=cfg.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            padding: 1rem;
            font-weight: 700;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        .fraud-alert {
            background-color: #ff4b4b;
            padding: 1rem;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        .safe-badge {
            background-color: #00c851;
            padding: 1rem;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 1.2rem;
            padding: 0.5rem 2rem;
            border-radius: 5px;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Data and Models
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(cfg.DATA_PATH, 'insurance_claims_fraud.csv'))
    return df

@st.cache_resource
def load_models_and_artifacts():
    models = {}
    try:
        results = pd.read_csv(os.path.join(cfg.MODEL_PATH, 'model_comparison_results.csv'))
        model_names = results['model'].unique()
        for name in model_names:
            path = os.path.join(cfg.MODEL_PATH, f'{name}.pkl')
            if os.path.exists(path):
                models[name] = joblib.load(path)
                
        # Load scaler
        scaler = joblib.load(os.path.join(cfg.MODEL_PATH, 'scaler.pkl'))
        # Load feature names
        feature_names = joblib.load(os.path.join(cfg.MODEL_PATH, 'feature_names.pkl'))
        
        return models, scaler, feature_names, results
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None, [], pd.DataFrame()

df = load_data()
models, scaler, feature_names, model_results = load_models_and_artifacts()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Dashboard", "Fraud Prediction Tool", "Advanced Analytics", "Model Performance", "Data Insights", "Explainability", "Load Dataset"])

st.sidebar.markdown("---")
st.sidebar.header("Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()) if models else ['None'])
threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

# Helper for Metrics
def metric_card(label, value, prefix="", suffix=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{prefix}{value}{suffix}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "Executive Dashboard":
    st.markdown('<div class="main-header">🔍 Insurance Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: gray; margin-bottom: 2rem;">AI-Powered Claims Intelligence</div>', unsafe_allow_html=True)
    
    # KPIs
    total_claims = len(df)
    fraud_cases = df['is_fraud'].sum()
    fraud_pct = (fraud_cases / total_claims) * 100
    
    # Calculate Estimated Savings from Best Model (if available)
    try:
        biz_metrics = pd.read_csv(os.path.join(cfg.MODEL_PATH, 'business_metrics.csv'))
        savings = biz_metrics['Total_Savings'][0]
        accuracy = model_results[model_results['model'] == selected_model_name]['test_accuracy'].values[0] if selected_model_name in models else 0
    except:
        savings = 0
        accuracy = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Total Claims", f"{total_claims:,}")
    with col2: metric_card("Fraud Detected", f"{fraud_cases:,}")
    with col3: metric_card("Est. Savings", f"{savings/1e6:.1f}M", prefix="$")
    with col4: metric_card("Detection Accuracy", f"{accuracy*100:.1f}", suffix="%")
    
    st.markdown("---")
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fraud Trend Over Time")
        # Ensure incident_date is datetime
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        daily_fraud = df[df['is_fraud']==1].groupby('incident_date').size().reset_index(name='count')
        fig = px.line(daily_fraud, x='incident_date', y='count', title="Daily Fraud Cases")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Fraud by Claim Type")
        fraud_by_type = df[df['is_fraud']==1]['incident_type'].value_counts()
        fig = px.pie(values=fraud_by_type.values, names=fraud_by_type.index, title="Fraud Distribution by Type", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Model Performance Comparison")
    if not model_results.empty:
        fig = px.bar(model_results, x='model', y='test_f2', color='test_f2', title="Model F2-Score Comparison (Recall Focus)")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.title("Model Performance Evaluation")
    
    if models:
        tabs = st.tabs(list(models.keys()))
        for i, (name, model) in enumerate(models.items()):
            with tabs[i]:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Key Metrics")
                    res = model_results[model_results['model'] == name]
                    if not res.empty:
                        st.table(res[['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_f2', 'test_roc_auc']].T)
                with c2:
                    st.subheader("Confusion Matrix")
                    # Static image from evaluation step? or Plotly?
                    # Let's use Plotly for interactivity if possible, or static image
                    # For now, let's look for specific image if saved, or just placeholder text citing visualization folder
                    img_path = os.path.join(cfg.VIZ_PATH, 'confusion_matrices.png')
                    if os.path.exists(img_path):
                         st.image(img_path, caption="Confusion Matrices for All Models")
                    else:
                         st.info("Visualizations not generated yet.")

    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(os.path.join(cfg.VIZ_PATH, 'roc_curves.png')):
            st.image(os.path.join(cfg.VIZ_PATH, 'roc_curves.png'), caption="ROC Curves")
    with col2:
        if os.path.exists(os.path.join(cfg.VIZ_PATH, 'pr_curves.png')):
            st.image(os.path.join(cfg.VIZ_PATH, 'pr_curves.png'), caption="Precision-Recall Curves")

# --- PAGE 3: FRAUD PREDICTION TOOL ---
elif page == "Fraud Prediction Tool":
    st.title("🕵️ Fraud Prediction Tool")
    
    with st.form("claim_form"):
        c1, c2, c3, c4 = st.columns(4)
        
        with c1: # Demographic
            st.subheader("Demographic")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["M", "F"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            employ = st.selectbox("Employment", ["Employed", "Unemployed", "Retired", "Self-Employed"])
            edu = st.selectbox("Education", ["High School", "College", "Graduate"])
            
        with c2: # Policy
            st.subheader("Policy")
            p_type = st.selectbox("Policy Type", ["Auto", "Property", "Workers_Comp"])
            p_age = st.number_input("Policy Age (Days)", 0, 10000, 365)
            premium = st.number_input("Annual Premium ($)", 500, 20000, 1500)
            coverage = st.number_input("Coverage Amount ($)", 5000, 1000000, 50000)
            deductible = st.selectbox("Deductible ($)", [500, 1000, 2000, 5000])
            prior_claims = st.slider("Prior Claims", 0, 10, 0)
            
        with c3: # Claim
            st.subheader("Claim Details")
            claim_amt = st.number_input("Claim Amount ($)", 100, 1000000, 5000)
            days_report = st.number_input("Days to Report", 0, 365, 1)
            inc_type = st.selectbox("Incident Type", ["Collision", "Theft", "Fire", "Water_Damage", "Injury", "Vandalism"])
            severity = st.selectbox("Severity", ["Minor", "Moderate", "Severe", "Total_Loss"])
            police = st.checkbox("Police Report Filed", value=True)
            witnesses = st.slider("Witnesses", 0, 10, 1)
            attorney = st.checkbox("Attorney Involved", value=False)
            
        with c4: # Other
            st.subheader("Additional")
            hour = st.slider("Incident Hour", 0, 23, 14)
            repair = st.selectbox("Repair Shop", ["Authorized", "Independent", "Unknown"])
            addr_change = st.checkbox("Recent Address Change", value=False)
            phone_change = st.checkbox("Recent Phone Change", value=False)
            social = st.selectbox("Social Media Check", ["Clean", "Suspicious", "Not_Available"])
            
        submitted = st.form_submit_button("Analyze Claim")
        
    if submitted:
        # Prepare input data
        input_data = {
            'policy_holder_age': age,
            'gender': gender,
            'marital_status': marital,
            'employment_status': employ,
            'education_level': edu,
            'policy_type': p_type,
            'policy_age_days': p_age,
            'policy_premium_annual': premium,
            'coverage_amount': coverage,
            'deductible': deductible,
            'prior_claims_count': prior_claims,
            'claim_amount': claim_amt,
            'days_to_report': days_report,
            'incident_type': inc_type,
            'incident_severity': severity,
            'police_report_filed': 'Yes' if police else 'No',
            'witnesses_present': witnesses,
            'attorney_involved': 'Yes' if attorney else 'No',
            'incident_time': hour,
            'repair_shop_type': repair,
            'claimant_address_change_recent': 'Yes' if addr_change else 'No',
            'phone_change_recent': 'Yes' if phone_change else 'No',
            'social_media_check': social,
            # Defaults/Pass-throughs for unlisted
            'claim_description_length': 100, 
            'medical_provider_count': 1,
            'incident_date': '2023-01-01', # Dummy
            'report_date': '2023-01-02', # Dummy
            'multiple_claims_same_incident': 'No' 
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        fe = FeatureEngineer()
        # Initialize scaler from file if possible, or we need to ensure FE uses the loaded scaler
        fe.scaler = scaler
        fe.feature_names = feature_names
        
        # We need to process this single row
        # Warning: Scaling requires fitting on training data. We should use loaded scaler.
        # FE pipeline usually fits scaler. Modified FE needed to transform only.
        # Check `full_pipeline` in `feature_engineering.py`. It has `is_training` flag.
        
        X_processed, _ = fe.full_pipeline(input_df, is_training=False)
        
        # Predict
        model = models[selected_model_name]
        prob = model.predict_proba(X_processed)[0][1]
        
        # Display Results
        st.markdown("---")
        st.subheader("Analysis Results")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            if prob > threshold:
                st.markdown('<div class="fraud-alert">⛔ HIGH RISK DETECTED</div>', unsafe_allow_html=True)
                st.write(f"This claim has a **{prob:.1%}** probability of being fraudulent.")
                st.warning("Recommendation: Flag for immediate Special Investigation Unit (SIU) review.")
            else:
                st.markdown('<div class="safe-badge">✅ LOW RISK</div>', unsafe_allow_html=True)
                st.write(f"This claim has a **{prob:.1%}** probability of being fraudulent.")
                st.success("Recommendation: Auto-process or standard review.")
                
            # Explainability Text
            try:
                exp = Explainability(model_name=selected_model_name)
                exp.model = model
                exp.explainer = None  # Re-init might be complex here without recreating logic
                # For demo, just text based on heuristic or if we can quickly init explainer
                # Loading explainer object if saved would be best.
                # Just show top contributing features logic from `feature_engineering` raw values?
                # Or create a SHAP explainer on the fly (expensive).
                st.info("Top Risk Factors: (SHAP calculation would appear here)")
            except:
                pass

# --- PAGE 4: DATA INSIGHTS ---
elif page == "Data Insights":
    st.title("Data Insights & Exploratory Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Correlations")
        # Compute correlation on numeric cols
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Claim Amount Distribution")
        fig = px.histogram(df, x="claim_amount", color="is_fraud", nbins=50, title="Claim Amount by Fraud Status")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 5: ADVANCED ANALYTICS ---
elif page == "Advanced Analytics":
    st.title("Advanced Analytics & Forecasting")
    
    st.subheader("Multi-Dimensional Fraud Analysis")
    # Sunburst: Policy Type -> Incident Type -> Fraud Status
    # Re-map fraud to string for clearer chart
    df_chart = df.copy()
    df_chart['status'] = df['is_fraud'].map({0: 'Legit', 1: 'Fraud'})
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.sunburst(df_chart, path=['policy_type', 'incident_type', 'status'], 
                          title="Policy -> Incident -> Fraud Hierarchy")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Parallel Categories
        # Binning claim amount for categorical view
        df_chart['claim_size'] = pd.cut(df_chart['claim_amount'], bins=[0, 5000, 20000, 100000, 1000000], labels=['Micro', 'Small', 'Medium', 'Large'])
        fig = px.parallel_categories(df_chart, dimensions=['policy_type', 'incident_severity', 'claim_size', 'status'],
                                     color="is_fraud", color_continuous_scale=px.colors.sequential.Inferno,
                                     title="Claim Flow: Policy to Status")
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("3D Cluster Analysis")
    # 3D Scatter
    fig = px.scatter_3d(df_chart, x='claim_amount', y='policy_age_days', z='policy_premium_annual',
                        color='status', opacity=0.7, title="3D Risk Clusters: Claim vs Policy Age vs Premium",
                        height=900)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔮 Fraud Volume Forecasting")
    
    # Forecasting
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    # Weekly fraud count
    daily_fraud = df[df['is_fraud']==1].set_index('incident_date').resample('W').size().reset_index(name='count')
    
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        # Simple rolling average forecast
        daily_fraud['MA_4_Week'] = daily_fraud['count'].rolling(window=4).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_fraud['incident_date'], y=daily_fraud['count'], mode='lines+markers', name='Actual Fraud'))
        fig.add_trace(go.Scatter(x=daily_fraud['incident_date'], y=daily_fraud['MA_4_Week'], mode='lines', name='4-Week Trend', line=dict(dash='dash', color='orange')))
        
        fig.update_layout(title="Weekly Fraud Volume with Trend Forecasting")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_f2:
        st.info("Forecasting Insight")
        avg_wk = daily_fraud['count'].mean()
        last_wk = daily_fraud['count'].iloc[-1]
        trend = "Increasing" if last_wk > avg_wk else "Decreasing"
        st.metric("Avg Weekly Fraud Cases", f"{avg_wk:.1f}")
        st.metric("Last Week", f"{last_wk}")
        st.write(f"Trend: **{trend}**")

# --- PAGE 6: EXPLAINABILITY ---
elif page == "Explainability":
    st.title("Model Explainability (SHAP)")
    
    st.write("Understanding why the model makes predictions is crucial for trust.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance")
        img_path = os.path.join(cfg.SHAP_PATH, f'shap_summary_{selected_model_name}.png')
        # Fallback to general summary if specific model doesn't exist
        if not os.path.exists(img_path):
             # Try finding any summary
             files = [f for f in os.listdir(cfg.SHAP_PATH) if 'shap_summary' in f]
             if files: img_path = os.path.join(cfg.SHAP_PATH, files[0])
            
        if os.path.exists(img_path):
            st.image(img_path, caption="SHAP Summary Plot")
        else:
            st.info("SHAP summary not found.")
            
    with col2:
        st.subheader("Example Prediction Path")
        # Waterfall
        img_path = os.path.join(cfg.SHAP_PATH, f'shap_waterfall_0_{selected_model_name}.png')
        if not os.path.exists(img_path):
             files = [f for f in os.listdir(cfg.SHAP_PATH) if 'shap_waterfall' in f]
             if files: img_path = os.path.join(cfg.SHAP_PATH, files[0])

        if os.path.exists(img_path):
            st.image(img_path, caption="Waterfall Plot (Example)")
        else:
            st.info("Waterfall plot not found.")

# --- PAGE 7: LOAD DATASET ---
elif page == "Load Dataset":
    st.title("📂 Data Management")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current Dataset Info")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        st.dataframe(df.head(), use_container_width=True)
    
    with c2:
        st.subheader("Upload New Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success(f"File loaded successfully! Shape: {new_df.shape}")
                
                if st.button("Use This Dataset"):
                     # In a real app, we'd update session_state or overwrite file. 
                     # For this demo, we can just show it works, as overwriting might break the model prediction if schema differs.
                     st.warning("Note: Updating the main dataset requires the specific schema expected by the model. For this demo, we are just inspecting the new file.")
                     st.dataframe(new_df.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
        st.markdown("---")
        st.subheader("Export Current Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "insurance_claims.csv",
            "text/csv",
            key='download-csv'
        )
            
