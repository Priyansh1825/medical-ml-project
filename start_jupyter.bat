@echo off
echo Starting Jupyter Notebook for M.Tech Medical Research...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Creating necessary directories...
if not exist notebooks mkdir notebooks
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed

echo.
echo Starting Jupyter Notebook...
echo Notebook will be available at: http://localhost:8888
echo.

REM Start Jupyter notebook
jupyter notebook --notebook-dir=./notebooks --port=8888 --no-browser

pause