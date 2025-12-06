"""
MLflow Dashboard for M.Tech Research Progress Tracking
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MtechResearchDashboard:
    """Dashboard for M.Tech research progress tracking with MLflow."""
    
    def __init__(self, tracking_uri=None):
        """Initialize dashboard."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiments = mlflow.search_experiments()
        
    def get_research_summary(self):
        """Get summary of all research experiments."""
        summary = []
        
        for exp in self.experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            
            if not runs.empty:
                latest_run = runs.iloc[0]
                best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
                
                summary.append({
                    'experiment': exp.name,
                    'total_runs': len(runs),
                    'latest_date': latest_run['start_time'].date(),
                    'best_accuracy': best_run['metrics.accuracy'] if 'metrics.accuracy' in best_run else None,
                    'best_model': best_run['tags.mlflow.runName'] if 'tags.mlflow.runName' in best_run else None,
                    'tags': exp.tags
                })
        
        return pd.DataFrame(summary)
    
    def plot_experiment_progress(self, experiment_name):
        """Plot progress of a specific experiment."""
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return None
        
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        if runs.empty:
            return None
        
        # Convert time
        runs['start_time'] = pd.to_datetime(runs['start_time'])
        runs = runs.sort_values('start_time')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Progress', 'Loss Progress', 
                          'Hyperparameter Evolution', 'Model Comparison'),
            vertical_spacing=0.15
        )
        
        # Accuracy progress
        fig.add_trace(
            go.Scatter(x=runs['start_time'], y=runs['metrics.accuracy'],
                      mode='lines+markers', name='Accuracy'),
            row=1, col=1
        )
        
        # Parameter trends (if available)
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        if param_cols:
            param_data = runs[param_cols[:3]]  # First 3 parameters
            for i, col in enumerate(param_data.columns):
                fig.add_trace(
                    go.Scatter(x=runs['start_time'], y=param_data[col],
                              mode='markers', name=col.replace('params.', '')),
                    row=1, col=2
                )
        
        # Model comparison
        if 'tags.mlflow.runName' in runs.columns:
            model_groups = runs.groupby('tags.mlflow.runName')['metrics.accuracy'].mean()
            fig.add_trace(
                go.Bar(x=model_groups.index, y=model_groups.values,
                      name='Model Accuracy'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, showlegend=True, title_text=f"Research Progress: {experiment_name}")
        return fig
    
    def generate_research_report(self, output_path="research_report.md"):
        """Generate a comprehensive research report in Markdown format."""
        report = []
        report.append("# M.Tech Research Progress Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Executive Summary")
        
        summary = self.get_research_summary()
        if not summary.empty:
            total_experiments = len(summary)
            total_runs = summary['total_runs'].sum()
            best_accuracy = summary['best_accuracy'].max()
            
            report.append(f"- **Total Experiments:** {total_experiments}")
            report.append(f"- **Total Runs:** {total_runs}")
            report.append(f"- **Best Accuracy Achieved:** {best_accuracy:.4f}")
            
            report.append("\n## Detailed Experiment Analysis")
            
            for _, row in summary.iterrows():
                report.append(f"\n### {row['experiment']}")
                report.append(f"- **Runs:** {row['total_runs']}")
                report.append(f"- **Latest Run:** {row['latest_date']}")
                report.append(f"- **Best Accuracy:** {row['best_accuracy']:.4f}")
                report.append(f"- **Best Model:** {row['best_model']}")
                
                if row['tags']:
                    report.append("- **Tags:**")
                    for key, value in row['tags'].items():
                        report.append(f"  - {key}: {value}")
        
        report.append("\n## Key Findings")
        report.append("1. **Model Performance:** [Add key findings about model performance]")
        report.append("2. **Hyperparameter Insights:** [Add insights about hyperparameters]")
        report.append("3. **Data Observations:** [Add observations about the data]")
        
        report.append("\n## Next Steps")
        report.append("1. [Add next research steps]")
        report.append("2. [Add planned experiments]")
        report.append("3. [Add improvements to consider]")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Research report saved to: {output_path}")
        return output_path
    
    def create_thesis_figures(self, output_dir="thesis_figures"):
        """Create publication-ready figures for thesis."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        summary = self.get_research_summary()
        
        # Figure 1: Experiment timeline
        plt.figure(figsize=(12, 6))
        for _, exp in summary.iterrows():
            exp_runs = mlflow.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(exp['experiment']).experiment_id]
            )
            if not exp_runs.empty:
                exp_runs['start_time'] = pd.to_datetime(exp_runs['start_time'])
                plt.scatter(exp_runs['start_time'], exp_runs.get('metrics.accuracy', 0), 
                           label=exp['experiment'], s=50)
        
        plt.title('Experiment Timeline - Accuracy Progression', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/experiment_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Model comparison
        plt.figure(figsize=(10, 6))
        model_accuracies = []
        
        for exp in self.experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty and 'tags.model_type' in runs.columns:
                for model_type in runs['tags.model_type'].unique():
                    model_runs = runs[runs['tags.model_type'] == model_type]
                    if 'metrics.accuracy' in model_runs.columns:
                        model_accuracies.append({
                            'model': model_type,
                            'accuracy': model_runs['metrics.accuracy'].max()
                        })
        
        if model_accuracies:
            model_df = pd.DataFrame(model_accuracies)
            model_df = model_df.groupby('model')['accuracy'].max().sort_values()
            
            plt.barh(model_df.index, model_df.values)
            plt.title('Best Accuracy by Model Type', fontsize=16)
            plt.xlabel('Accuracy', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Thesis figures saved to: {output_dir}/")
        return output_dir

def main():
    """Command-line interface for the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="M.Tech Research Dashboard")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--report", action='store_true', help="Generate research report")
    parser.add_argument("--figures", action='store_true', help="Create thesis figures")
    parser.add_argument("--summary", action='store_true', help="Show research summary")
    
    args = parser.parse_args()
    
    dashboard = MtechResearchDashboard(tracking_uri=args.tracking_uri)
    
    if args.report:
        dashboard.generate_research_report()
    
    if args.figures:
        dashboard.create_thesis_figures()
    
    if args.summary or not (args.report or args.figures):
        summary = dashboard.get_research_summary()
        print("\n" + "="*80)
        print("M.Tech Research Progress Summary")
        print("="*80)
        print(summary.to_string())
        
        print("\nTo start interactive dashboard:")
        print("1. Install streamlit: pip install streamlit")
        print("2. Run: streamlit run src/experiments/mlflow_dashboard.py -- --tracking-uri YOUR_URI")

if __name__ == "__main__":
    main()