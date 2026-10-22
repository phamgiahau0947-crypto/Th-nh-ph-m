@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "VPY=%CD%\.venv\Scripts\python.exe"
echo ============================================================
echo   BUOC 2 - MO DASHBOARD "LIQUIDITY STRESS METER"
echo ============================================================
if not exist "%VPY%" goto :noinstall
"%VPY%" -c "import streamlit, plotly, xgboost, sklearn" >nul 2>nul
if errorlevel 1 goto :noinstall
echo Trinh duyet se tu mo http://localhost:8501 sau khoang 5 giay.
echo   - Chon nguon du lieu o thanh ben trai: Gia lap / Live Binance / Replay lich su.
echo   - Nut "Huong dan va cach doc ket qua" o dau trang giai thich tung con so.
echo   - DE CUA SO DEN NAY MO trong luc xem. Tat dashboard: dong cua so nay (hoac Ctrl+C).
echo.
rem headless = khong hoi email lan dau; tu mo trinh duyet sau 5 giay
start "" /min cmd /c "timeout /t 5 >nul & start "" http://localhost:8501"
"%VPY%" -m streamlit run src\task3_dashboard\app.py --server.port 8501 --server.headless true || goto :fail
exit /b 0
:noinstall
echo [!] Chua cai dat (hoac cai chua xong). Hay bam dup file 1_CAI_DAT.bat truoc.
pause
exit /b 1
:fail
echo.
echo [!] Co loi o tren. Xem muc "Loi thuong gap" trong README.md, hoac chup man hinh cua so nay gui lai.
pause
exit /b 1
