"""
MLflow-integrated Medical ML Training Script
M.Tech Research - Experiment Tracking Implementation
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalMLExperiment:
    """Medical ML experiment with MLflow tracking for M.Tech research."""
    
    def __init__(self, experiment_name="medical-research", run_name=None):
        """Initialize experiment."""
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Load configuration
        with open("config/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Models to evaluate
        self.models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC,
            'logistic_regression': LogisticRegression
        }
        
        # Hyperparameter grids for each model
        self.param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        }
    
    def load_data(self):
        """Load and preprocess medical data."""
        logger.info("Loading and preprocessing medical data...")
        
        # For M.Tech research, you would load your actual medical dataset here
        # This is a sample generator for demonstration
        
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate medical features
        data = pd.DataFrame({
            'patient_id': range(n_samples),
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'gender': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
            'blood_pressure': np.random.normal(120, 20, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'glucose': np.random.normal(100, 20, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        })
        
        # Simulate disease probability (logistic function)
        z = (
            0.1 * (data['age'] - 45) / 15 +
            0.3 * data['cholesterol'] / 200 +
            0.2 * data['glucose'] / 100 +
            0.4 * data['smoking'] +
            0.3 * data['family_history'] +
            np.random.normal(0, 0.2, n_samples)
        )
        data['disease_prob'] = 1 / (1 + np.exp(-z))
        data['has_disease'] = (data['disease_prob'] > 0.5).astype(int)
        
        # Drop helper columns
        data = data.drop(['patient_id', 'disease_prob'], axis=1)
        
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Class distribution:\n{data['has_disease'].value_counts()}")
        
        return data
    
    def preprocess_data(self, data):
        """Preprocess medical data."""
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = data.drop('has_disease', axis=1)
        y = data['has_disease']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics, y_pred, cm
    
    def run_experiment(self, model_name, params=None):
        """Run a single experiment with MLflow tracking."""
        if params is None:
            params = {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {model_name}")
        logger.info(f"Parameters: {params}")
        logger.info(f"{'='*60}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("research_project", "M.Tech Medical ML")
            mlflow.set_tag("student", "M.Tech Student")
            
            # Load and preprocess data
            data = self.load_data()
            X_train, X_test, y_train, y_test, scaler = self.preprocess_data(data)
            
            # Initialize and train model
            model_class = self.models[model_name]
            model = model_class(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred, cm = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            # Log artifacts
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = f"temp/cm_{model_name}.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(cm_path)
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = data.drop('has_disease', axis=1).columns
                
                plt.figure(figsize=(10, 6))
                indices = np.argsort(importances)[::-1]
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
                plt.title(f'Feature Importance - {model_name}')
                fi_path = f"temp/fi_{model_name}.png"
                plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(fi_path)
            
            # Save scaler
            scaler_path = f"temp/scaler_{model_name}.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)
            
            # Create a summary report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = f"temp/classification_report_{model_name}.csv"
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)
            
            # Clean up temp files
            import os
            for f in [cm_path, fi_path, scaler_path, report_path]:
                if os.path.exists(f):
                    os.remove(f)
            
            logger.info(f"Experiment completed: {model_name}")
            logger.info(f"Best accuracy: {metrics['accuracy']:.4f}")
            
            return metrics, model
    
    def run_grid_search(self, model_name, n_trials=10):
        """Run hyperparameter optimization with MLflow tracking."""
        import optuna
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running hyperparameter optimization for {model_name}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"{'='*60}")
        
        data = self.load_data()
        X_train, X_test, y_train, y_test, _ = self.preprocess_data(data)
        
        def objective(trial):
            # Suggest hyperparameters
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
                }
            else:
                params = {}
            
            # Train and evaluate
            model_class = self.models[model_name]
            model = model_class(**params, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            
            return accuracy
        
        # Create study
        study = optuna.create_study(direction='maximize', study_name=f"{model_name}_optuna")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"{model_name}_optuna_search"):
            mlflow.set_tag("optimization", "optuna")
            mlflow.set_tag("model", model_name)
            
            # Run optimization
            study.optimize(objective, n_trials=n_trials)
            
            # Log best parameters and metrics
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_accuracy", study.best_value)
            
            # Train final model with best parameters
            best_model = self.models[model_name](**study.best_params, random_state=42)
            best_model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(best_model, f"best_{model_name}")
            
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best accuracy: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")
            
            return study.best_params, study.best_value, best_model

def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Medical ML Experiments with MLflow")
    parser.add_argument("--experiment", default="medical-research", help="Experiment name")
    parser.add_argument("--model", choices=['random_forest', 'gradient_boosting', 'svm', 'logistic_regression', 'all'], 
                       default='all', help="Model to train")
    parser.add_argument("--optimize", action='store_true', help="Run hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=10, help="Number of optimization trials")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = MedicalMLExperiment(experiment_name=args.experiment)
    
    if args.model == 'all':
        models_to_run = ['random_forest', 'gradient_boosting', 'logistic_regression']
    else:
        models_to_run = [args.model]
    
    results = {}
    
    for model_name in models_to_run:
        if args.optimize:
            # Run hyperparameter optimization
            best_params, best_accuracy, best_model = experiment.run_grid_search(
                model_name, n_trials=args.trials
            )
            results[model_name] = {
                'best_params': best_params,
                'best_accuracy': best_accuracy,
                'model': best_model
            }
        else:
            # Run with default parameters
            default_params = {}
            metrics, model = experiment.run_experiment(model_name, default_params)
            results[model_name] = {
                'metrics': metrics,
                'model': model
            }
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY - M.Tech Medical ML Research")
    print("="*80)
    for model_name, result in results.items():
        if 'best_accuracy' in result:
            print(f"{model_name.upper()}: Best Accuracy = {result['best_accuracy']:.4f}")
            print(f"  Best Params: {result['best_params']}")
        else:
            print(f"{model_name.upper()}: Accuracy = {result['metrics']['accuracy']:.4f}, "
                  f"F1-Score = {result['metrics']['f1_score']:.4f}")
    print("="*80)
    
    # Save summary to file
    summary_df = pd.DataFrame([
        {
            'model': model_name,
            'accuracy': result.get('best_accuracy', result.get('metrics', {}).get('accuracy', 0)),
            'optimized': args.optimize
        }
        for model_name, result in results.items()
    ])
    
    summary_path = "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    mlflow.log_artifact(summary_path)
    
    print(f"\nExperiment tracking available at: {mlflow.get_tracking_uri()}")
    print("To view results, run: mlflow ui")

if __name__ == "__main__":
    main()