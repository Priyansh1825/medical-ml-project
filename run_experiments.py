#!/usr/bin/env python
"""
M.Tech Experiment Runner - Automates ML experiments
"""

import subprocess
import sys
import time
from datetime import datetime

def run_experiments():
    """Run a series of experiments for M.Tech research."""
    
    experiments = [
        {
            "name": "medical-research-baseline",
            "command": ["python", "src/experiments/medical_ml_experiment.py", 
                       "--experiment", "medical-research-baseline",
                       "--model", "all"]
        },
        {
            "name": "medical-research-optimized",
            "command": ["python", "src/experiments/medical_ml_experiment.py",
                       "--experiment", "medical-research-optimized",
                       "--model", "all", "--optimize", "--trials", "20"]
        },
        {
            "name": "medical-research-deep-learning",
            "command": ["python", "src/experiments/medical_ml_experiment.py",
                       "--experiment", "medical-research-deep-learning",
                       "--model", "random_forest", "--optimize", "--trials", "30"]
        }
    ]
    
    print("="*80)
    print("M.Tech Medical ML Experiment Runner")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Experiments: {len(experiments)}")
    print("="*80)
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running experiment: {exp['name']}")
        print(f"Command: {' '.join(exp['command'])}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run experiment
            result = subprocess.run(exp['command'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            
            # Log output
            with open(f"experiment_log_{exp['name']}.txt", "w") as f:
                f.write(f"Experiment: {exp['name']}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n--- STDOUT ---\n")
                f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
            
            elapsed_time = time.time() - start_time
            
            results.append({
                "experiment": exp['name'],
                "status": "SUCCESS",
                "time": elapsed_time
            })
            
            print(f"✓ Experiment completed successfully in {elapsed_time:.2f} seconds")
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            
            results.append({
                "experiment": exp['name'],
                "status": "FAILED",
                "time": elapsed_time,
                "error": str(e)
            })
            
            print(f"✗ Experiment failed after {elapsed_time:.2f} seconds")
            print(f"Error: {e}")
        
        print("-" * 60)
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT RUNNER SUMMARY")
    print("="*80)
    
    for result in results:
        status_icon = "✓" if result["status"] == "SUCCESS" else "✗"
        print(f"{status_icon} {result['experiment']}: {result['status']} ({result['time']:.2f}s)")
    
    print("\nNext steps:")
    print("1. View results: mlflow ui")
    print("2. Generate report: python src/experiments/mlflow_dashboard.py --report")
    print("3. Create thesis figures: python src/experiments/mlflow_dashboard.py --figures")
    print("="*80)
    
    return results

if __name__ == "__main__":
    run_experiments()
    