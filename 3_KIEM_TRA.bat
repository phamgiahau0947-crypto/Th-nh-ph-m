@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
echo ============================================================
echo   (TUY CHON) TU KIEM TRA TOAN BO CODE  (~2-5 phut)
echo   PASS / DAT = code chay dung, KHONG co nghia mo hinh du bao gioi
echo ============================================================
if not exist "%VPY%" (
  echo [!] Chua cai dat. Hay bam dup file 1_CAI_DAT.bat truoc.
  pause
  exit /b 1
)
set "LOI=0"
echo.
echo [1/3] Phan du lieu: 34 kiem tra tren du lieu gia, khong can Internet...
"%VPY%" -m src.task1_data.tools.selftest || set "LOI=1"
echo.
echo [2/3] Phan toan hoc va mo hinh: cong thuc OFI, khong nhin truoc tuong lai, live = offline...
"%VPY%" -m src.task2_models.selftest || set "LOI=1"
echo.
echo [3/3] Phan dashboard: giao dien, Replay, bang xep hang, ket noi Binance that (can Internet)...
"%VPY%" -m src.task3_dashboard.selftest --live || set "LOI=1"
echo.
if "%LOI%"=="0" (echo KET QUA: TAT CA DEU DAT.) else (echo KET QUA: CO MUC CHUA DAT - xem dong [FAIL] / [LOI] o tren. Muc "Live" can Internet va khong bi chan Binance.)
pause
exit /b %LOI%
