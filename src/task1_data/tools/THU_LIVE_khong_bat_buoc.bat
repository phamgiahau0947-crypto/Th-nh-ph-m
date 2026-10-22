@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\.."
if not exist ".venv\Scripts\python.exe" ( echo Hay bam src\task1_data\CHAY_TASK1.bat truoc. & pause & exit /b 1 )
echo Thu so lenh live BTCUSDT (cua so nay) va ETHUSDT (cua so thu 2) vao kho khi may dang mo.
echo Cu de 2 cua so chay; dong cua so hoac Ctrl+C de dung.
start "ETHUSDT live" ".venv\Scripts\python.exe" -m src.task1_data.websocket_streamer --symbol ETHUSDT --record --seconds 0
".venv\Scripts\python.exe" -m src.task1_data.websocket_streamer --symbol BTCUSDT --record --seconds 0
pause
