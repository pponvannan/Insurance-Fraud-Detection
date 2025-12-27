import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, cohen_kappa_score, 
                             matthews_corrcoef, make_scorer, fbeta_score)
import joblib

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.feature_importance = pd.DataFrame()
        
    def initialize_models(self):
        """Initialize all 8 models with config parameters"""
        p = cfg.MODEL_PARAMS
        
        self.models['logistic_regression'] = LogisticRegression(**p['logistic_regression'])
        self.models['random_forest'] = RandomForestClassifier(**p['random_forest'])
        self.models['xgboost'] = xgb.XGBClassifier(**p['xgboost'])
        self.models['lightgbm'] = lgb.LGBMClassifier(**p['lightgbm'])
        self.models['catboost'] = cb.CatBoostClassifier(**p['catboost'])
        self.models['gradient_boosting'] = GradientBoostingClassifier(**p['gradient_boosting'])
        self.models['svm'] = SVC(**p['svm'])
        self.models['mlp'] = MLPClassifier(**p['mlp'])
        
    def train_and_evaluate(self, X, y):
        """Train models and evaluate using 5-fold CV"""
        print("Starting model training and evaluation...")
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'f2': make_scorer(fbeta_score, beta=2),
            'roc_auc': 'roc_auc',
        }
        
        cv_results_list = []
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            # Cross-validation
            try:
                # Note: For some models like CatBoost/LGBM inside CV, we accept default verbosity or handle it in init
                scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=1) # n_jobs=1 to avoid issues with some boosters in this environment
                
                # Train final model on full dataset
                model.fit(X, y)
                
                # Calculate additional metrics on full set (training score - beware overfitting, but useful for stats)
                # Actually, better to rely on CV scores for comparison.
                # But we need predictions for business metrics logic if we want to run "Cost Benefit" on a holdout.
                # Since we used all data for CV, we can use the mean CV scores.
                
                avg_scores = {k: v.mean() for k, v in scores.items()}
                avg_scores['model'] = name
                avg_scores['training_time'] = time.time() - start_time
                
                # Calculate PR-AUC, Kappa, MCC manually if needed, but standard CV is good for selection.
                # Let's trust f2 for selection as requested.
                
                cv_results_list.append(avg_scores)
                
                # Save model
                joblib.dump(model, os.path.join(cfg.MODEL_PATH, f'{name}.pkl'))
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                
        self.results_df = pd.DataFrame(cv_results_list)
        return self.results_df
        
    def select_best_model(self):
        """Select best model based on weighted F2 Score (Recall prioritized)"""
        # Sort by test_f2 (cross_validate returns 'test_metric')
        if 'test_f2' in self.results_df.columns:
            best_row = self.results_df.loc[self.results_df['test_f2'].idxmax()]
            self.best_model_name = best_row['model']
            print(f"\nBest Model selected: {self.best_model_name} (F2 Score: {best_row['test_f2']:.4f})")
        else:
            # Fallback
            self.best_model_name = 'xgboost' 
            print("Could not find F2 score, defaulting to xgboost")
            
        return self.best_model_name

    def extract_feature_importance(self, X_columns):
        """Extract feature importance for tree-based models"""
        importance_data = []
        
        for name in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']:
            if name in self.models:
                model = self.models[name]
                if hasattr(model, 'feature_importances_'):
                    imps = model.feature_importances_
                    importance_data.append(pd.DataFrame({
                        'feature': X_columns,
                        'importance': imps,
                        'model': name
                    }))
                    
        if importance_data:
            self.feature_importance = pd.concat(importance_data, ignore_index=True)
            # Save feature importance
            self.feature_importance.to_csv(os.path.join(cfg.MODEL_PATH, 'feature_importance.csv'), index=False)
            
    def calculate_business_metrics(self, X, y):
        """
        Calculate business impact:
        - Cost of False Positives (Investigation Cost)
        - Cost of False Negatives (Fraud Amount)
        
        We'll use the BEST model to calculate this on the full provided dataset (X, y).
        Ideally this should be on a test set.
        """
        best_model = self.models[self.best_model_name]
        y_pred = best_model.predict(X)
        
        # Confusion elements
        # TP: Fraud detected (Good)
        # FP: Legit flagged as fraud (Bad - Investigation Cost)
        # FN: Fraud missed (Bad - Payout Cost)
        # TN: Legit ignored (Good)
        
        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))
        TN = np.sum((y_pred == 0) & (y == 0))
        
        cost_fp = FP * cfg.INVESTIGATION_COST
        cost_fn = FN * cfg.AVERAGE_FRAUD_AMOUNT
        possible_fraud_loss = (TP + FN) * cfg.AVERAGE_FRAUD_AMOUNT
        actual_fraud_loss = FN * cfg.AVERAGE_FRAUD_AMOUNT
        saved_fraud_loss = TP * cfg.AVERAGE_FRAUD_AMOUNT
        
        total_cost_model = cost_fp + cost_fn
        # Baseline cost (no model) = all fraud missed
        baseline_cost = possible_fraud_loss
        
        savings = baseline_cost - total_cost_model
        roi = (savings / (FP * cfg.INVESTIGATION_COST + 1e-6)) * 100 # ROI on investigation spend
        
        metrics = {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Cost_False_Positives': cost_fp,
            'Cost_Missed_Fraud': cost_fn,
            'Total_Model_Cost': total_cost_model,
            'Total_Savings': savings,
            'ROI_Percent': roi
        }
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(os.path.join(cfg.MODEL_PATH, 'business_metrics.csv'), index=False)
        print("\nBusiness Metrics for Best Model:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_PATH, 'train_processed.csv'))
    X = train_df.drop(columns=['is_fraud'])
    y = train_df['is_fraud']
    
    # Initialize and Train
    trainer = ModelTrainer()
    trainer.initialize_models()
    results = trainer.train_and_evaluate(X, y)
    
    print("\nModel Performance (CV Mean):")
    print(results[['model', 'test_accuracy', 'test_f2', 'test_roc_auc']].sort_values(by='test_f2', ascending=False))
    
    # Select Best
    best_model = trainer.select_best_model()
    
    # Feature Importance
    trainer.extract_feature_importance(X.columns)
    
    # Business Metrics (on full training set for demonstration - usually holdout)
    trainer.calculate_business_metrics(X, y)
    
    # Save results summary
    results.to_csv(os.path.join(cfg.MODEL_PATH, 'model_comparison_results.csv'), index=False)
    print("\nTraining Pipeline Completed.")
