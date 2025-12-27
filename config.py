import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data/raw/')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data/processed/')
MODEL_PATH = os.path.join(BASE_DIR, 'models/saved_models/')
VIZ_PATH = os.path.join(BASE_DIR, 'visualizations/')
SHAP_PATH = os.path.join(BASE_DIR, 'shap_explanations/')

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(VIZ_PATH, exist_ok=True)
os.makedirs(SHAP_PATH, exist_ok=True)

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 20,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': 10,
        'random_state': 42
    },
    'lightgbm': {
        'is_unbalance': True,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    },
    'catboost': {
        'auto_class_weights': 'Balanced',
        'iterations': 500,
        'random_state': 42,
        'verbose': 0
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'svm': {
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    },
    'logistic_regression': {
        'class_weight': 'balanced',
        'penalty': 'l2',
        'solver': 'liblinear', 
        'random_state': 42
    },
    'mlp': {
        'hidden_layer_sizes': (100, 50),
        'early_stopping': True,
        'random_state': 42
    }
}

# Business costs
INVESTIGATION_COST = 500  # Cost to investigate each flagged claim
AVERAGE_FRAUD_AMOUNT = 25000  # Average fraudulent claim payout

# Fraud thresholds
FRAUD_THRESHOLD_LOW = 0.3
FRAUD_THRESHOLD_MEDIUM = 0.5
FRAUD_THRESHOLD_HIGH = 0.7
FRAUD_THRESHOLD_CRITICAL = 0.9

# Feature lists
NUMERICAL_FEATURES = [
    'policy_holder_age', 
    'policy_age_days', 
    'policy_premium_annual', 
    'coverage_amount', 
    'deductible', 
    'prior_claims_count', 
    'claim_amount', 
    'days_to_report', 
    'incident_time',
    'witnesses_present', 
    'claim_description_length', 
    'medical_provider_count'
]

CATEGORICAL_FEATURES = [
    'gender', 
    'marital_status', 
    'employment_status', 
    'education_level', 
    'policy_type', 
    'incident_type', 
    'incident_severity', 
    'police_report_filed', 
    'attorney_involved', 
    'multiple_claims_same_incident',
    'repair_shop_type', 
    'claimant_address_change_recent', 
    'phone_change_recent', 
    'social_media_check'
]

DERIVED_FEATURES = [
    'claim_to_premium_ratio', 
    'claim_to_coverage_ratio', 
    'days_policy_active', 
    'claim_velocity', 
    'reporting_delay_category', 
    'age_risk_group', 
    'high_risk_time', 
    'suspicious_behavior_score',
    'risk_flag_count', 
    'credibility_score'
]

# All features after encoding and engineering (approximate list for now)
ALL_FEATURES = [] # To be populated dynamically or used for reference

# Streamlit settings
PAGE_TITLE = "Insurance Fraud Detection"
PAGE_ICON = "🔍"
LAYOUT = "wide"
