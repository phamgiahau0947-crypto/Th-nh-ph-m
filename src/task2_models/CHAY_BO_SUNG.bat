@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0..\.."
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
echo ============================================================
echo   TASK 2 - BO SUNG: DU BAO SPREAD + BO MO HINH CHO BANG XEP HANG
echo   (khong doi mo hinh / so lieu da bao cao)
echo ============================================================
if not exist "%VPY%" (
  echo [!] Chua co .venv - hay chay src\task1_data\CHAY_TASK1.bat truoc.
  goto :fail
)
if not exist models\model_direction.pkl (
  echo [!] Chua co mo hinh chinh - hay chay src\task2_models\CHAY_TASK2.bat truoc.
  goto :fail
)
echo [1/2] Du bao spread 10 giay toi (walk-forward, ~5-10 phut)...
"%VPY%" -m src.task2_models.spread_forecast || goto :fail
echo [2/2] Bo mo hinh cho bang xep hang (~3-5 phut)...
"%VPY%" -m src.task2_models.model_zoo || goto :fail
echo.
echo HOAN TAT. models\model_spread.pkl, models\model_zoo.pkl
echo Bang: outputs\tables\spread_forecast_*.csv, model_zoo_check.csv
pause
exit /b 0
:fail
echo.
echo [!] Co loi o tren. Chup man hinh gui lai de duoc ho tro.
pause
exit /b 1
