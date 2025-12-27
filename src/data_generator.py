import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

def generate_data(num_samples=1000):
    print(f"Generating {num_samples} insurance claims...")
    np.random.seed(42)
    random.seed(42)

    # 1. Demographic Info
    data = {
        'claim_id': [f'CLM{i:05d}' for i in range(1, num_samples + 1)],
        'policy_holder_age': np.random.randint(18, 81, num_samples),
        'gender': np.random.choice(['M', 'F'], num_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], num_samples),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Retired', 'Self-Employed'], num_samples, p=[0.6, 0.1, 0.2, 0.1]),
        'education_level': np.random.choice(['High School', 'College', 'Graduate'], num_samples, p=[0.4, 0.4, 0.2]),
    }

    # 2. Policy Info
    data['policy_type'] = np.random.choice(['Auto', 'Property', 'Workers_Comp'], num_samples, p=[0.6, 0.3, 0.1])
    data['policy_age_days'] = np.random.randint(0, 3651, num_samples)
    data['policy_premium_annual'] = np.random.randint(500, 10001, num_samples)
    data['coverage_amount'] = np.random.randint(10000, 500001, num_samples)
    data['deductible'] = np.random.choice([500, 1000, 2000, 5000], num_samples)
    data['prior_claims_count'] = np.random.choice([0, 1, 2, 3, 4, 5], num_samples, p=[0.5, 0.25, 0.15, 0.05, 0.03, 0.02])

    df = pd.DataFrame(data)

    # 3. Claim Details
    # Report date and incident date logic
    incident_dates = []
    report_dates = []
    days_to_report_list = []

    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_samples):
        incident_date = start_date + timedelta(days=np.random.randint(0, 365))
        days_to_report = int(np.random.exponential(scale=5)) # Most reports are quick
        if days_to_report > 90: days_to_report = 90
        report_date = incident_date + timedelta(days=days_to_report)
        
        incident_dates.append(incident_date.date())
        report_dates.append(report_date.date())
        days_to_report_list.append(days_to_report)

    df['incident_date'] = incident_dates
    df['report_date'] = report_dates
    df['days_to_report'] = days_to_report_list

    df['claim_amount'] = np.random.randint(100, 200001, num_samples)
    df['incident_type'] = np.random.choice(['Collision', 'Theft', 'Fire', 'Water_Damage', 'Injury', 'Vandalism'], num_samples)
    df['incident_severity'] = np.random.choice(['Minor', 'Moderate', 'Severe', 'Total_Loss'], num_samples)
    df['police_report_filed'] = np.random.choice(['Yes', 'No'], num_samples)
    df['witnesses_present'] = np.random.choice(range(6), num_samples, p=[0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

    # 4. Suspicious Indicators
    df['attorney_involved'] = np.random.choice(['Yes', 'No'], num_samples)
    df['incident_time'] = np.random.randint(0, 24, num_samples)
    df['claim_description_length'] = np.random.randint(20, 500, num_samples)
    df['multiple_claims_same_incident'] = np.random.choice(['Yes', 'No'], num_samples, p=[0.05, 0.95])
    df['repair_shop_type'] = np.random.choice(['Authorized', 'Independent', 'Unknown'], num_samples, p=[0.6, 0.3, 0.1])
    df['medical_provider_count'] = np.random.choice(range(11), num_samples) # Mostly 0-2 but up to 10
    
    # 5. Network/Behavioral
    df['claimant_address_change_recent'] = np.random.choice(['Yes', 'No'], num_samples, p=[0.1, 0.9])
    df['phone_change_recent'] = np.random.choice(['Yes', 'No'], num_samples, p=[0.05, 0.95])
    df['social_media_check'] = np.random.choice(['Clean', 'Suspicious', 'Not_Available'], num_samples, p=[0.7, 0.1, 0.2])

    # 6. Fraud Logic Assignment
    df['is_fraud'] = 0
    fraud_indices = []

    for idx, row in df.iterrows():
        fraud_prob = 0.0
        
        # --- High Fraud Probability Rules ---
        # Claim amount very close to coverage limit (>90%)
        if row['claim_amount'] > 0.9 * row['coverage_amount']:
            fraud_prob += 0.3
        
        # Reported >14 days after incident
        if row['days_to_report'] > 14:
            fraud_prob += 0.2
        
        # No police report for theft/vandalism
        if row['incident_type'] in ['Theft', 'Vandalism'] and row['police_report_filed'] == 'No':
            fraud_prob += 0.25
        
        # Attorney involved within 2 days (proxy: if attorney involved and short report time)
        if row['attorney_involved'] == 'Yes' and row['days_to_report'] <= 2:
            fraud_prob += 0.2
            
        # Multiple claims from same incident
        if row['multiple_claims_same_incident'] == 'Yes':
            fraud_prob += 0.25
            
        # New policy (<30 days) with large claim (>20k)
        if row['policy_age_days'] < 30 and row['claim_amount'] > 20000:
            fraud_prob += 0.4
            
        # Incident at suspicious times (2am-4am)
        if 2 <= row['incident_time'] <= 4:
            fraud_prob += 0.15
            
        # Unemployed with high-value claim (>50k)
        if row['employment_status'] == 'Unemployed' and row['claim_amount'] > 50000:
            fraud_prob += 0.2

        # --- Medium Fraud Probability Rules ---
        # Claim amount >50% of coverage
        if row['claim_amount'] > 0.5 * row['coverage_amount']:
            fraud_prob += 0.1
        
        # Prior claims > 3
        if row['prior_claims_count'] > 3:
            fraud_prob += 0.1
            
        # Unknown repair shop
        if row['repair_shop_type'] == 'Unknown':
            fraud_prob += 0.15
            
        # No witnesses for major incident (Severe/Total Loss)
        if row['incident_severity'] in ['Severe', 'Total_Loss'] and row['witnesses_present'] == 0:
            fraud_prob += 0.15

        # Recent address/phone changes
        if row['claimant_address_change_recent'] == 'Yes' or row['phone_change_recent'] == 'Yes':
            fraud_prob += 0.1
            
        # Social media suspicious
        if row['social_media_check'] == 'Suspicious':
            fraud_prob += 0.2

        # --- Low Fraud Probability / Mitigating Factors ---
        # Older policies -> more legitimate (reduce fraud prob)
        if row['policy_age_days'] > 2000:
            fraud_prob -= 0.1
        
        # Police reports -> less fraud
        if row['police_report_filed'] == 'Yes':
            fraud_prob -= 0.1
            
        # Witnesses -> less fraud
        if row['witnesses_present'] > 0:
            fraud_prob -= 0.05
            
        # Higher education -> less fraud (correlation assumption)
        if row['education_level'] == 'Graduate':
            fraud_prob -= 0.05

        # Cap probability
        fraud_prob = max(0, min(1, fraud_prob))
        
        # Scaling down to achieve target 8-12% rate (previous run was ~22%)
        fraud_prob = fraud_prob * 0.45

        # Stochastic assignment based on calculated probability
        if np.random.random() < fraud_prob:
             df.at[idx, 'is_fraud'] = 1

    # Adjust fraud rate to be within 8-12% if needed
    current_fraud_rate = df['is_fraud'].mean()
    print(f"Initial generated fraud rate: {current_fraud_rate:.2%}")
    
    # If too low, flip some high-risk borders. If too high, flip some low-risk borders.
    # For now, let's rely on the rules. If it's wildly off, we can force it.
    
    # Ensure raw data matches config expectations (add any missing cols if needed)
    # claim_amount_to_coverage_ratio is derived, so not in raw? User said "claim_amount_to_coverage_ratio (0-1)" under Suspicious Indicators in prompt 44.
    # But in feature engineering prompt 53, it says "DERIVED FEATURES... claim_to_coverage_ratio".
    # I will stick to Raw columns being pure data, derived columns generated later.
    # However, user prompt 44 listed it under "Suspicious Indicators" to include in generation.
    # I'll enable it as a "raw" column if the user explicitly asked for it in dataset generation, 
    # but typically it's derived. I'll add it as a column if requested. 
    # user prompt 44: "SUSPICIOUS INDICATORS: ... claim_amount_to_coverage_ratio ... TARGET: is_fraud"
    # I will add it.
    df['claim_amount_to_coverage_ratio'] = df['claim_amount'] / df['coverage_amount']

    # Dates to string for CSV
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['report_date'] = pd.to_datetime(df['report_date'])
    
    # Save
    save_path = os.path.join(cfg.DATA_PATH, 'insurance_claims_fraud.csv')
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

    # Data Quality Report
    print("\nData Quality Report:")
    print("-" * 30)
    print(f"Total Samples: {len(df)}")
    print(f"Fraud Probability: {df['is_fraud'].mean():.2%}")
    print(f"Class Balance:\n{df['is_fraud'].value_counts(normalize=True)}")
    print(f"\nMissing Values:\n{df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    generate_data()
