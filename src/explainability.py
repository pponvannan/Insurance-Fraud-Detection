import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
import shap.plots

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# SHAP JS visualization in matplotlib figures - sometimes tricky in scripts.
# We will focus on saving static plots (summary, waterfall) and minimal HTML if needed.

class Explainability:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.original_feature_names = None
        
        # Load feature names
        try:
             with open(os.path.join(cfg.MODEL_PATH, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = joblib.load(f)
        except:
            print("Warning: feature_names.pkl not found.")
            
    def load_model(self, model_name=None):
        if model_name:
            self.model_name = model_name
        
        # Load best model if not specified, or logic to find best
        if not self.model_name:
            # Try to find best model from results
            try:
                res = pd.read_csv(os.path.join(cfg.MODEL_PATH, 'model_comparison_results.csv'))
                # Assuming sorted by f2 in training script or we sort here
                self.model_name = res.sort_values(by='test_f2', ascending=False).iloc[0]['model']
                print(f"Loaded best model for explanation: {self.model_name}")
            except:
                self.model_name = 'xgboost' # Default
        
        model_path = os.path.join(cfg.MODEL_PATH, f'{self.model_name}.pkl')
        self.model = joblib.load(model_path)
        return self.model

    def generate_shap_values(self, X_data, sample_size=100):
        """Generate SHAP values for a sample of data"""
        print(f"Generating SHAP values using {self.model_name}...")
        
        # Determine explainer type
        # TreeExplainer for tree models, Linear for Logistic, Kernel for others
        
        # Subsample for speed if dataset is large, but 1000 is small.
        X_sample = X_data
        if len(X_data) > sample_size:
            X_sample = X_data.sample(sample_size, random_state=42)
            
        if self.model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'gradient_boosting']:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer(X_sample)
        else:
            # Generic - likely slower
            # Using partial dependence or kmeans as background
            masker = shap.maskers.Independent(data=X_sample)
            self.explainer = shap.Explainer(self.model.predict, masker)
            self.shap_values = self.explainer(X_sample)
            
        return self.explainer, self.shap_values, X_sample

    def plot_shap_summary(self, plot_type='dot'):
        """Plot summary (beeswarm)"""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.shap_values.data, feature_names=self.feature_names, show=False, plot_type=plot_type)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.SHAP_PATH, f'shap_summary_{self.model_name}.png'))
        plt.close()
        
    def plot_shap_waterfall(self, index=0):
        """Plot waterfall for a specific instance"""
        plt.figure()
        # shap.plots.waterfall is for a single explanation
        # The new API: shap_values[index] is an Explanation object
        shap.plots.waterfall(self.shap_values[index], show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.SHAP_PATH, f'shap_waterfall_{index}_{self.model_name}.png'))
        plt.close()
        
    def plot_feature_dependence(self, feature_name):
        """Plot dependence plot for a feature"""
        try:
            plt.figure(figsize=(8, 6))
            shap.plots.scatter(self.shap_values[:, feature_name], show=False, color=self.shap_values)
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.SHAP_PATH, f'shap_dependence_{feature_name}_{self.model_name}.png'))
            plt.close()
        except Exception as e:
            print(f"Could not plot dependence for {feature_name}: {e}")

    def explain_prediction(self, row_data, feature_names=None):
        """
        Generate human-readable text explanation for a single prediction using SHAP values.
        row_data: numpy array or pandas series (processed)
        """
        # Calculate local shap values
        # We need the explainer already fitted
        if not self.explainer:
             raise ValueError("Explainer not initialized. Run generate_shap_values first.")
             
        # Shape check
        if len(row_data.shape) == 1:
            row_data = row_data.reshape(1, -1)
            
        # Get Shap values for this single instance
        if self.model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'gradient_boosting']:
             shap_vals = self.explainer.shap_values(row_data)
             # TreeExplainer.shap_values returns matrix for binary classification (sometimes list of matrices)
             if isinstance(shap_vals, list): 
                  # For binary, it might return [neg_class_vals, pos_class_vals]
                  shap_vals = shap_vals[1]
             base_value = self.explainer.expected_value
             if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
                 base_value = base_value[1] # Positive class
        else:
             # Generic wrapper
             explanation = self.explainer(row_data)
             shap_vals = explanation.values[0]
             base_value = explanation.base_values[0]

        # Handle different output shapes from TreeExplainer (sometimes (1, features))
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0]
            
        # Create dict of feature -> contribution
        contributions = dict(zip(self.feature_names, shap_vals))
        
        # Sort by absolute value desc
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Probability
        prob = self.model.predict_proba(row_data)[0][1]
        risk_level = "CRITICAL" if prob > 0.8 else ("HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.3 else "LOW"))
        
        text = f"Claim Risk Analysis: **{risk_level}** (Fraud Probability: {prob:.1%})\n\n"
        text += "Top Factors Driving This Score:\n"
        
        for i, (feat, val) in enumerate(sorted_contribs[:5]):
            direction = "Increased risk" if val > 0 else "Decreased risk"
            text += f"{i+1}. **{feat}**: {direction} (+{val:.2f} impact)\n"
            
        return text

if __name__ == "__main__":
    # Load processed data
    # We need a sample to initialize explainer
    train_df = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_PATH, 'train_processed.csv'))
    X = train_df.drop(columns=['is_fraud'])
    
    # Initialize
    exp = Explainability()
    exp.load_model() # Will load best model
    
    # Generate SHAP
    exp.generate_shap_values(X, sample_size=200)
    
    # Plotting
    print("Generating SHAP plots...")
    exp.plot_shap_summary()
    exp.plot_shap_waterfall(index=0) # Example waterfall for first sample
    
    # Plot dependence for top features (finding top feature first)
    # Simple heuristic: take first from summary 
    vals = np.abs(exp.shap_values.values).mean(0)
    top_feat_idx = np.argsort(vals)[-1]
    top_feat_name = exp.feature_names[top_feat_idx]
    exp.plot_feature_dependence(top_feat_name)
    
    print("Explainability analysis completed.")
