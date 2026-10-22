@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0..\.."
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
echo ============================================================
echo   TASK 3 - DASHBOARD LIQUIDITY STRESS METER
echo ============================================================
if not exist "%VPY%" (
  echo [!] Chua co .venv - hay chay src\task1_data\CHAY_TASK1.bat truoc.
  goto :fail
)
"%VPY%" -c "import streamlit, plotly" >nul 2>nul
if errorlevel 1 (
  echo [1/2] Cai thu vien Task 3 (lan dau, ~1 phut^)...
  "%VPY%" -m pip install -r src\task3_dashboard\requirements.txt || goto :fail
)
echo [2/2] Mo dashboard tren trinh duyet: http://localhost:8501
echo       Chon nguon du lieu o thanh ben trai: Gia lap / Live Binance / Replay.
echo       DE CUA SO NAY MO trong luc xem. Tat dashboard: dong cua so nay (hoac Ctrl+C).
echo.
rem headless = khong hoi email lan dau; tu mo trinh duyet sau 4 giay
start "" /min cmd /c "timeout /t 4 >nul & start "" http://localhost:8501"
"%VPY%" -m streamlit run src\task3_dashboard\app.py --server.port 8501 --server.headless true || goto :fail
exit /b 0
:fail
echo.
echo [!] Co loi o tren. Chup man hinh gui lai de duoc ho tro.
pause
exit /b 1
