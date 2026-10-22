@echo off
chcp 65001 >nul
echo Tat tu dong thu live...
powershell -NoProfile -Command "Stop-ScheduledTask -TaskName Task1_ThuLiveKhiMoMay -ErrorAction SilentlyContinue; Unregister-ScheduledTask -TaskName Task1_ThuLiveKhiMoMay -Confirm:$false -ErrorAction SilentlyContinue; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_live_capture.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo DA TAT. Du lieu da thu van giu nguyen trong data\raw\*\live_capture va se duoc nhap kho o lan cap nhat sau.
pause
