@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
echo ============================================================
echo   BUOC 1 - CAI DAT  (chi can lam 1 lan, can Internet, ~5-15 phut)
echo ============================================================
if exist "%VPY%" goto :libs
echo [1/4] Tim Python tren may...
set "PY="
py -3.13 -c "import sys" >nul 2>nul && set "PY=py -3.13"
if not defined PY (py -3 -c "import sys" >nul 2>nul && set "PY=py -3")
if not defined PY (python -c "import sys" >nul 2>nul && set "PY=python")
if not defined PY goto :nopython
%PY% -c "import sys; print('    Dung Python', sys.version.split()[0])"
%PY% -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)"
if errorlevel 1 echo [!] Khuyen dung Python 3.13 (da thu). Phien ban khac co the cai loi - van thu tiep...
echo       Tao moi truong Python rieng trong thu muc nay (.venv)...
%PY% -m venv .venv || goto :fail
:libs
echo [2/4] Cai thu vien (DUNG TAT CUA SO)...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt || goto :fail
echo [3/4] Cai PyTorch ban CPU cho mo hinh DeepLOB (khong bat buoc, tai ~130 MB)...
"%VPY%" -c "import torch" >nul 2>nul
if not errorlevel 1 goto :check
"%VPY%" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 echo [!] Khong cai duoc PyTorch. Khong sao: dashboard van chay, chi thieu DeepLOB o trang So sanh mo hinh.
:check
echo [4/4] Kiem tra nhanh...
"%VPY%" -c "import pandas, pyarrow, sklearn, xgboost, statsmodels, streamlit, plotly, websocket, yaml" || goto :fail
"%VPY%" -c "import pickle, pathlib; [pickle.loads(pathlib.Path('models', f).read_bytes()) for f in ('model_direction.pkl', 'model_stress.pkl', 'model_spread.pkl', 'model_zoo.pkl')]; print('    Doc duoc 4 file mo hinh: OK')" || goto :fail
echo.
echo HOAN TAT CAI DAT. Buoc tiep theo: bam dup file 2_MO_DASHBOARD.bat
pause
exit /b 0
:nopython
echo.
echo [!] Khong tim thay Python tren may.
echo     1. Tai Python 3.13 tai https://www.python.org/downloads/
echo     2. Khi cai, NHO TICK o "Add python.exe to PATH" o man hinh dau tien.
echo     3. Cai xong, bam dup lai file nay.
pause
exit /b 1
:fail
echo.
echo [!] Co loi o tren. Xem muc "Loi thuong gap" trong README.md, hoac chup man hinh cua so nay gui lai.
pause
exit /b 1
