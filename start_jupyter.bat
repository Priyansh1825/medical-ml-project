@echo off
echo Starting Jupyter Notebook for Medical ML Project...
echo Virtual Environment: medical-ml
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start Jupyter notebook
jupyter notebook --notebook-dir=./notebooks --config=.jupyter/jupyter_notebook_config.py

pause
