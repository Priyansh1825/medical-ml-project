#!/usr/bin/env python
"""
MLflow Experiment Manager for Medical ML Research
M.Tech Project - Automated Experiment Tracking
"""

import mlflow
import yaml
import argparse
import sys
from pathlib import Path
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow experiments for medical research."""
    
    def __init__(self, config_path="config/mlflow_config.yaml"):
        """Initialize MLflow manager with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set tracking URI
        if 'tracking_uri' in self.config['tracking']:
            mlflow.set_tracking_uri(self.config['tracking']['tracking_uri'])
        else:
            # Local file-based tracking
            tracking_dir = Path(self.config['tracking']['store'].split(':')[-1])
            tracking_dir.parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(self.config['tracking']['store'])
        
        # Create default experiment
        self.default_experiment = self.config['experiments']['default_experiment']
        mlflow.set_experiment(self.default_experiment)
        
        # Set artifact location
        self.artifact_location = Path(self.config['artifacts']['artifact_location'])
        self.artifact_location.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MLflow initialized with experiment: {self.default_experiment}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Artifact location: {self.artifact_location}")
    
    def create_experiment(self, name, tags=None):
        """Create a new experiment for M.Tech research."""
        if tags is None:
            tags = {
                "project": "mtech-research",
                "student_id": "MTECH_2024",
                "university": "Your University",
                "department": "Computer Science/Medical AI"
            }
        
        try:
            experiment_id = mlflow.create_experiment(
                name=name,
                artifact_location=str(self.artifact_location / name),
                tags=tags
            )
            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id
        except Exception as e:
            logger.warning(f"Experiment {name} may already exist: {e}")
            return mlflow.get_experiment_by_name(name).experiment_id
    
    def list_experiments(self):
        """List all experiments."""
        experiments = mlflow.search_experiments()
        print("\n" + "="*80)
        print("MLflow Experiments (M.Tech Research)")
        print("="*80)
        for exp in experiments:
            print(f"\nExperiment: {exp.name}")
            print(f"  ID: {exp.experiment_id}")
            print(f"  Artifact Location: {exp.artifact_location}")
            print(f"  Tags: {exp.tags}")
            print(f"  Created: {exp.creation_time}")
        print("="*80)
        return experiments
    
    def search_runs(self, experiment_name=None, filter_string=""):
        """Search runs in experiment."""
        if experiment_name:
            experiment_id = self.create_experiment(experiment_name)
        else:
            experiment_id = None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id] if experiment_id else None,
            filter_string=filter_string,
            order_by=["metrics.accuracy DESC"]
        )
        
        if not runs.empty:
            print(f"\nFound {len(runs)} runs:")
            print(runs[['run_id', 'metrics.accuracy', 'metrics.precision', 
                       'metrics.recall', 'metrics.f1_score', 'status']])
        else:
            print("No runs found.")
        
        return runs
    
    def start_mlflow_server(self):
        """Start MLflow UI server."""
        import subprocess
        import threading
        import time
        
        def run_server():
            cmd = [
                "mlflow", "ui",
                "--host", self.config['ui']['host'],
                "--port", str(self.config['ui']['port']),
                "--backend-store-uri", self.config['tracking']['store'],
                "--default-artifact-root", self.config['artifacts']['artifact_location']
            ]
            subprocess.run(cmd)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give server time to start
        time.sleep(3)
        print(f"\nMLflow UI started at: http://{self.config['ui']['host']}:{self.config['ui']['port']}")
        print("Press Ctrl+C to stop the server")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping MLflow server...")
    
    def export_experiments_to_csv(self, output_path="experiments_summary.csv"):
        """Export all experiments to CSV for thesis reporting."""
        experiments = mlflow.search_experiments()
        
        summary_data = []
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            
            if not runs.empty:
                for _, run in runs.iterrows():
                    summary_data.append({
                        'experiment_name': exp.name,
                        'run_id': run['run_id'],
                        'start_time': run['start_time'],
                        'end_time': run['end_time'],
                        'status': run['status'],
                        'accuracy': run.get('metrics.accuracy', None),
                        'precision': run.get('metrics.precision', None),
                        'recall': run.get('metrics.recall', None),
                        'f1_score': run.get('metrics.f1_score', None),
                        'roc_auc': run.get('metrics.roc_auc', None),
                        'model_type': run.get('tags.model_type', 'Unknown')
                    })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} runs to {output_path}")
        return df

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="MLflow Manager for M.Tech Medical Research")
    parser.add_argument("--action", choices=['init', 'list', 'create', 'search', 'server', 'export'], 
                       default='init', help="Action to perform")
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--filter", default="", help="Filter string for search")
    parser.add_argument("--output", default="experiments_summary.csv", help="Output file for export")
    
    args = parser.parse_args()
    
    manager = MLflowManager()
    
    if args.action == 'init':
        print("MLflow initialized successfully.")
        print(f"Default experiment: {manager.default_experiment}")
    
    elif args.action == 'list':
        manager.list_experiments()
    
    elif args.action == 'create':
        if not args.name:
            print("Error: Experiment name required with --name")
            sys.exit(1)
        manager.create_experiment(args.name)
    
    elif args.action == 'search':
        manager.search_runs(args.name, args.filter)
    
    elif args.action == 'server':
        manager.start_mlflow_server()
    
    elif args.action == 'export':
        manager.export_experiments_to_csv(args.output)

if __name__ == "__main__":
    main()