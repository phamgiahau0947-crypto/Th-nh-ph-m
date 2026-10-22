@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\.."
echo Xoa bo nho dem (cache) cua pip o o C de giai phong dung luong...
set "VPY=%CD%\.venv\Scripts\python.exe"
if exist "%VPY%" ("%VPY%" -m pip cache purge) else (python -m pip cache purge)
if exist "%LOCALAPPDATA%\pip\Cache" rmdir /s /q "%LOCALAPPDATA%\pip\Cache"
echo Xong. Tu nay cache (neu co) se nam trong thu muc du an: .cache\pip
pause
