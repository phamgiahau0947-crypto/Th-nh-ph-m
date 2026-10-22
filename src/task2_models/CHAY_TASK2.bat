@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0..\.."
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
echo ============================================================
echo   TASK 2 - TOAN HOC, ML, LIQUIDITY STRESS ^& LOI SUAT
echo ============================================================
if not exist "%VPY%" (
  echo [!] Chua co .venv - hay chay src\task1_data\CHAY_TASK1.bat truoc.
  goto :fail
)
"%VPY%" -c "import sklearn, xgboost, statsmodels, matplotlib, scipy" >nul 2>nul
if errorlevel 1 (
  echo [1/2] Cai thu vien Task 2...
  "%VPY%" -m pip install -r src\task2_models\requirements.txt || goto :fail
)
echo [2/2] Chay toan bo pipeline (mat khoang 1-2 gio, DUNG TAT CUA SO)...
if not exist outputs\logs mkdir outputs\logs
"%VPY%" -m src.task2_models.run_all %* || goto :fail
echo.
echo HOAN TAT. Mo hinh: models\   Bang: outputs\tables\   Hinh: outputs\figures\
pause
exit /b 0
:fail
echo.
echo [!] Co loi o tren. Chup man hinh gui lai de duoc ho tro.
pause
exit /b 1
