import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def preprocess_data(self, df, is_training=True):
        """
        Main function to orchestrate feature engineering
        """
        df = df.copy()
        
        # 1. Date Handling
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        df['report_date'] = pd.to_datetime(df['report_date'])
        df['policy_start_date'] = df['incident_date'] - pd.to_timedelta(df['policy_age_days'], unit='D')
        
        # 2. Derived Features
        # claim_to_premium_ratio
        df['claim_to_premium_ratio'] = df['claim_amount'] / df['policy_premium_annual']
        
        # claim_to_coverage_ratio (if not already present or needs recalculation)
        df['claim_to_coverage_ratio'] = df['claim_amount'] / df['coverage_amount']
        
        # days_policy_active
        df['days_policy_active'] = df['policy_age_days']
        
        # claim_velocity
        # Avoid division by zero: if policy_age_days is 0, treat as very high velocity or 0? 
        # Logic: prior_claims / years_active. If active < 1 year, normalize.
        years_active = df['policy_age_days'] / 365.0
        df['claim_velocity'] = df.apply(lambda x: x['prior_claims_count'] / (x['policy_age_days']/365.0) if x['policy_age_days'] > 30 else 0, axis=1)

        # reporting_delay_category
        def get_delay_category(days):
            if days == 0: return 'Immediate'
            elif days == 1: return 'Same_Day'
            elif days <= 7: return 'Week'
            else: return 'Late'
        df['reporting_delay_category'] = df['days_to_report'].apply(get_delay_category)
        
        # age_risk_group
        def get_age_group(age):
            if age < 30: return 'Young'
            elif age < 60: return 'Middle'
            else: return 'Senior'
        df['age_risk_group'] = df['policy_holder_age'].apply(get_age_group)
        
        # high_risk_time
        df['high_risk_time'] = df['incident_time'].apply(lambda x: 1 if x in [0,1,2,3,4,23] else 0)
        
        # suspicious_behavior_score (sum of boolean flags that we need to ensure are present/convert to int)
        # Assuming flags like 'police_report_filed'='No' are bad? 
        # The prompt says: "sum of red flags". Let's define red flags available in raw data.
        # We need to be careful not to leak the target.
        # Flags: no police report for theft, late report, etc.
        # But for simpler implementation, let's use the ones explicitly asked or obvious binary columns.
        
        # 3. Categorical Encoding
        # Feature: education_level (Ordinal)
        education_map = {'High School': 0, 'College': 1, 'Graduate': 2}
        df['education_level_encoded'] = df['education_level'].map(education_map)
        
        # Binary Encoding (Yes/No)
        binary_cols = ['police_report_filed', 'attorney_involved', 'multiple_claims_same_incident', 
                       'claimant_address_change_recent', 'phone_change_recent']
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
        
        # Social media check (Ordinal-ish or OHE? Prompt says "Binary encode: all Yes/No variables". 
        # Social media has 3 values. Let's map Clean=0, Not_Available=0.5, Suspicious=1 ?)
        # Or just OHE. Let's OHE it as part of generic categorical.
        
        # One-Hot Encoding
        ohe_cols = ['policy_type', 'incident_type', 'incident_severity', 'marital_status', 
                    'employment_status', 'reporting_delay_category', 'age_risk_group',
                    'repair_shop_type', 'social_media_check']
        
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
        
        # 4. Interaction Features
        # attorney_early_involvement = attorney_involved & days_to_report < 3
        df['attorney_early_involvement'] = ((df['attorney_involved'] == 1) & (df['days_to_report'] < 3)).astype(int)
        
        # new_policy_large_claim = (policy_age_days < 90) & (claim_amount > 20000)
        df['new_policy_large_claim'] = ((df['policy_age_days'] < 90) & (df['claim_amount'] > 20000)).astype(int)
        
        # no_evidence_major_claim = (witnesses_present == 0) & (police_report_filed == 0) & (incident_severity == Severe)
        # Note: incident_severity is OHE now. We need access to original or use OHE columns.
        # Easier to define before dropping/OHE, but since I OHE'd, I check OHE columns.
        # Incident Severity: Severe, Total_Loss are major.
        # Let's reconstruct or assume logic.
        # Re-implementing logic pre-OHE would be cleaner. 
        pass # Moving Interaction Features block up before OHE in actual execution flow.

        # 5. Aggregation Features
        # risk_flag_count = count of suspicious indicators
        # Indicators: attorney_involved, multiple_claims, address_change, phone_change, social_media_suspicious, no_police_report
        # We need to standardized "suspicious" direction.
        # attorney_involved=1 (suspicious)
        # multiple_claims=1 (suspicious)
        # address_change=1 (suspicious)
        # phone_change=1 (suspicious)
        # police_report_filed=0 (suspicious?) -> depends on incident. Generally yes.
        # Let's create a temporary "police_report_missing" flag.
        
        # ... Reworking process flow for correctness below ...

        return df

    def full_pipeline(self, df, is_training=True):
        # 1. Derived Features & Date handling
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        
        # Ratios
        df['claim_to_premium_ratio'] = df['claim_amount'] / df['policy_premium_annual']
        df['claim_to_coverage_ratio'] = df['claim_amount'] / df['coverage_amount']
        df['days_policy_active'] = df['policy_age_days']
        
        # Velocity
        df['claim_velocity'] = df.apply(lambda x: x['prior_claims_count'] / (x['policy_age_days']/365.0) if x['policy_age_days'] > 30 else 0, axis=1)
        
        # Categories
        df['reporting_delay_category'] = df['days_to_report'].apply(
            lambda x: 'Immediate' if x==0 else ('Same_Day' if x==1 else ('Week' if x<=7 else 'Late'))
        )
        df['age_risk_group'] = df['policy_holder_age'].apply(
            lambda x: 'Young' if x<30 else ('Middle' if x<60 else 'Senior')
        )
        df['high_risk_time'] = df['incident_time'].apply(lambda x: 1 if x in [0,1,2,3,4,23] else 0)

        # 2. Binary Encoding & Mapping
        binary_map = {'Yes': 1, 'No': 0}
        binary_cols = ['police_report_filed', 'attorney_involved', 'multiple_claims_same_incident', 
                       'claimant_address_change_recent', 'phone_change_recent']
        for col in binary_cols:
             if col in df.columns:
                df[col] = df[col].map(binary_map)

        education_map = {'High School': 0, 'College': 1, 'Graduate': 2}
        df['education_level_encoded'] = df['education_level'].map(education_map)

        # 3. Interaction Features
        df['attorney_early_involvement'] = ((df['attorney_involved'] == 1) & (df['days_to_report'] < 3)).astype(int)
        df['new_policy_large_claim'] = ((df['policy_age_days'] < 90) & (df['claim_amount'] > 20000)).astype(int)
        
        # no_evidence_major_claim
        is_major = df['incident_severity'].isin(['Severe', 'Total_Loss'])
        no_witness = df['witnesses_present'] == 0
        no_police = df['police_report_filed'] == 0
        df['no_evidence_major_claim'] = (is_major & no_witness & no_police).astype(int)

        # 4. Aggregation Features
        # Risk flags
        df['police_report_missing'] = (df['police_report_filed'] == 0).astype(int)
        df['social_media_suspicious'] = (df['social_media_check'] == 'Suspicious').astype(int)
        
        risk_cols = ['attorney_involved', 'multiple_claims_same_incident', 'claimant_address_change_recent', 
                     'phone_change_recent', 'police_report_missing', 'social_media_suspicious', 'high_risk_time',
                     'new_policy_large_claim']
        
        df['risk_flag_count'] = df[risk_cols].sum(axis=1)
        df['suspicious_behavior_score'] = df['risk_flag_count'] # Synonym in prompt
        
        # Credibility score (inverse of risk, simple heuristic)
        # Weighted sum: witness(+1), police(+1), older_policy(+1), employed(+1)
        df['credibility_score'] = (
            df['witnesses_present'].clip(upper=1) * 1 + 
            df['police_report_filed'] * 1.5 + 
            (df['policy_age_days'] > 365).astype(int) * 1 +
            (df['employment_status'] != 'Unemployed').astype(int) * 1 -
            df['risk_flag_count'] * 0.5
        )

        # 5. One-Hot Encoding
        ohe_cols = ['policy_type', 'incident_type', 'incident_severity', 'marital_status', 
                    'employment_status', 'reporting_delay_category', 'age_risk_group',
                    'repair_shop_type', 'social_media_check', 'gender']
        
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
        
        # Drop unused columns
        drop_cols = ['claim_id', 'incident_date', 'report_date', 'policy_start_date', 'education_level', 
                     'police_report_missing', 'social_media_suspicious'] # Clean up temp cols
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        # 6. Scaling
        # Separate target if exists
        target = None
        if 'is_fraud' in df.columns:
            target = df['is_fraud']
            df = df.drop(columns=['is_fraud'])
            
        feature_cols = df.columns.tolist()
        
        if is_training:
            self.feature_names = feature_cols
            df_scaled = self.scaler.fit_transform(df)
        else:
            # Align columns (missing columns in test set filled with 0)
            missing_cols = set(self.feature_names) - set(df.columns)
            for c in missing_cols:
                df[c] = 0
            df = df[self.feature_names] # Ensure order
            df_scaled = self.scaler.transform(df)
            
        return df_scaled, target

    def handle_imbalance(self, X, y):
        # Implementation of SMOTE + RandomUnderSampler
        print(f"Original class distribution: {pd.Series(y).value_counts(normalize=True).to_dict()}")
        
        over = SMOTE(sampling_strategy=0.3, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        
        X_res, y_res = pipeline.fit_resample(X, y)
        print(f"Resampled class distribution: {pd.Series(y_res).value_counts(normalize=True).to_dict()}")
        
        return X_res, y_res

    def save_artifacts(self):
        with open(os.path.join(cfg.MODEL_PATH, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(cfg.MODEL_PATH, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join(cfg.DATA_PATH, 'insurance_claims_fraud.csv'))
    
    # Process
    fe = FeatureEngineer()
    X, y = fe.full_pipeline(df, is_training=True)
    
    # Handle Imbalance
    X_res, y_res = fe.handle_imbalance(X, y)
    
    # Save artifacts
    fe.save_artifacts()
    
    # Save processed data for debugging/next steps
    X_df = pd.DataFrame(X_res, columns=fe.feature_names)
    X_df['is_fraud'] = y_res
    X_df.to_csv(os.path.join(cfg.PROCESSED_DATA_PATH, 'train_processed.csv'), index=False)
    
    print("Feature engineering completed and data saved.")
