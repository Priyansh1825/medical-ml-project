#!/bin/bash
# Jupyter startup script for medical ML project

echo "Starting Jupyter Notebook for Medical ML Project..."
echo "Virtual Environment: $(which python)"
echo "Kernel: medical-ml"

# Start Jupyter notebook
jupyter notebook --notebook-dir=./notebooks --config=.jupyter/jupyter_notebook_config.py
