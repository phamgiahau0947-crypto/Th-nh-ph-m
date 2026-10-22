@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\.."
if not exist ".venv\Scripts\python.exe" ( echo Hay bam src\task1_data\CHAY_TASK1.bat truoc. & pause & exit /b 1 )
echo Test ket noi live 60 giay va do do tre...
".venv\Scripts\python.exe" -m src.task1_data.websocket_streamer --seconds 60
pause
