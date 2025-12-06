@echo off
echo Starting MLflow for M.Tech Medical Research...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Creating necessary directories...
if not exist mlflow\mlruns mkdir mlflow\mlruns
if not exist mlflow\artifacts mkdir mlflow\artifacts
if not exist temp mkdir temp

echo.
echo Starting MLflow Tracking Server...
echo MLflow UI will be available at: http://localhost:5000
echo.

REM Start MLflow server
mlflow server ^
    --host 127.0.0.1 ^
    --port 5000 ^
    --backend-store-uri file:./mlflow/mlruns ^
    --default-artifact-root ./mlflow/artifacts

pause