@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0..\.."
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
echo ============================================================
echo   TASK 1 - DU LIEU SO LENH BTCUSDT  (chi can bam file nay)
echo ============================================================
if exist "%VPY%" goto :check_libs
echo [1/3] Tao moi truong Python rieng cho du an (.venv)...
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -m venv .venv || goto :fail
:check_libs
"%VPY%" -c "import pandas, pyarrow, requests, websocket, yaml" >nul 2>nul
if not errorlevel 1 goto :installed
echo [1/3] Cai thu vien (can Internet, mat vai phut - DUNG TAT CUA SO)...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r src\task1_data\requirements.txt || goto :fail
"%VPY%" -c "import pandas, pyarrow, requests, websocket, yaml" || goto :fail
"%VPY%" -m src.task1_data.tools.selftest || goto :fail
:installed
dir /b "data\raw\raw_lob_trades.parquet\*_seed.parquet" >nul 2>nul
if errorlevel 1 (
  echo [2/3] Dung kho nen lan dau - co the mat 30-90 phut, cu de may chay...
  "%VPY%" -m src.task1_data.update_store --init || goto :fail
) else (
  echo [2/3] Cap nhat kho nen...
  "%VPY%" -m src.task1_data.update_store || goto :fail
)
powershell -NoProfile -Command "if (Get-ScheduledTask -TaskName Task1_CapNhatKhoLOB -ErrorAction SilentlyContinue) {exit 0} else {exit 1}" >nul 2>nul
if not errorlevel 1 goto :done
echo [3/3] Dang ky tu dong cap nhat moi ngay 07:30 (may tat luc do thi lan bat sau se tu chay bu)...
powershell -NoProfile -Command "$a=New-ScheduledTaskAction -Execute '%CD%\.venv\Scripts\pythonw.exe' -Argument '\"%CD%\src\task1_data\tools\run_update.py\"' -WorkingDirectory '%CD%'; $t=New-ScheduledTaskTrigger -Daily -At 07:30; $s=New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries; Register-ScheduledTask -TaskName Task1_CapNhatKhoLOB -Action $a -Trigger $t -Settings $s -Force | Out-Null"
if errorlevel 1 (echo [!] Khong dang ky duoc lich tu dong. Khong sao: thinh thoang bam lai file nay la duoc.) else (echo     Da dang ky. Nhat ky chay ngam: data\processed\update_log.txt)
:done
echo.
echo HOAN TAT. Kho nen: data\raw\raw_lob_trades.parquet   ^|   Xem bang Excel: data\processed\BTCUSDT_view.csv
pause
exit /b 0
:fail
echo.
echo [!] Co loi o tren. Chup man hinh cua so nay gui lai de duoc ho tro.
pause
exit /b 1
