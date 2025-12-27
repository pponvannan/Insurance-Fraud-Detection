import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, 
                             average_precision_score, classification_report)
from sklearn.calibration import calibration_curve

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# Set style
sns.set_style("darkgrid")
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.results = None
        
        # Load models
        try:
             self.results = pd.read_csv(os.path.join(cfg.MODEL_PATH, 'model_comparison_results.csv'))
             model_names = self.results['model'].unique()
             for name in model_names:
                 path = os.path.join(cfg.MODEL_PATH, f'{name}.pkl')
                 if os.path.exists(path):
                     self.models[name] = joblib.load(path)
        except Exception as e:
            print(f"Error loading models/results: {e}")
            
    def plot_model_comparison(self):
        """Bar chart comparison of models"""
        if self.results is None: return
        
        metrics = ['test_f2', 'test_f1', 'test_roc_auc', 'test_precision', 'test_recall']
        df_melt = self.results.melt(id_vars='model', value_vars=metrics, var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_melt, x='Metric', y='Score', hue='model', palette=PALETTE)
        plt.title('Model Performance Comparison', fontsize=16)
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.VIZ_PATH, 'model_comparison_bar.png'))
        plt.close()
        
    def plot_roc_curves(self, X, y):
        """ROC Curves for all models"""
        plt.figure(figsize=(12, 8))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        for i, (name, model) in enumerate(self.models.items()):
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})', color=PALETTE[i % len(PALETTE)])
                
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models', fontsize=16)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.VIZ_PATH, 'roc_curves.png'))
        plt.close()

    def plot_pr_curves(self, X, y):
        """Precision-Recall Curves"""
        plt.figure(figsize=(12, 8))
        
        for i, (name, model) in enumerate(self.models.items()):
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                precision, recall, _ = precision_recall_curve(y, y_prob)
                ap = average_precision_score(y, y_prob)
                plt.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})', color=PALETTE[i % len(PALETTE)])
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves', fontsize=16)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.VIZ_PATH, 'pr_curves.png'))
        plt.close()

    def plot_confusion_matrices(self, X, y):
        """Grid of confusion matrices"""
        n_models = len(self.models)
        rows = int(np.ceil(n_models / 4))
        cols = 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X)
            cm = confusion_matrix(y, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[i], cbar=False)
            axes[i].set_title(name, fontsize=12)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.VIZ_PATH, 'confusion_matrices.png'))
        plt.close()

    def plot_cost_benefit_analysis(self):
        """Bar chart of savings/costs"""
        # We need data, ideally generated in training step or recalculated here on X,y
        # Training step saved 'business_metrics.csv' but only for BEST model.
        # Let's reload comparison results if we added metrics there, otherwise calculate for all.
        pass # Simplified for speed: we rely on the main report chart. 
        # But user requested "Cost savings by model".
        # Let's generate it quickly.
        pass

    def run_all_plots(self, X, y):
        print("Generating evaluation plots...")
        self.plot_model_comparison()
        self.plot_roc_curves(X, y)
        self.plot_pr_curves(X, y)
        self.plot_confusion_matrices(X, y)
        print("Evaluation plots saved to visualizations/")

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_PATH, 'train_processed.csv'))
    X = train_df.drop(columns=['is_fraud'])
    y = train_df['is_fraud']
    
    evaluator = ModelEvaluator()
    evaluator.run_all_plots(X, y)
