@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\.."
if not exist ".venv\Scripts\pythonw.exe" ( echo Hay bam src\task1_data\CHAY_TASK1.bat truoc. & pause & exit /b 1 )
echo ============================================================
echo   BAT TU DONG THU SO LENH LIVE MOI KHI MO MAY (BTC + ETH)
echo ============================================================
echo  - Moi lan dang nhap Windows, chuong trinh tu chay ngam (khong hien cua so).
echo  - May tat / ngu thi tam ngung, mo lai thi tu thu tiep. Khong can bat 24/24.
echo  - Dung luong: ~0.6 MB / gio / cap. Nhat ky: data\processed\live_capture_log.txt
echo  - Muon tat: bam TAT_TU_THU_LIVE.bat
echo.
powershell -NoProfile -Command "$a=New-ScheduledTaskAction -Execute '%CD%\.venv\Scripts\pythonw.exe' -Argument '\"%CD%\src\task1_data\tools\run_live_capture.py\"' -WorkingDirectory '%CD%'; $t=New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME; $s=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew; Register-ScheduledTask -TaskName Task1_ThuLiveKhiMoMay -Action $a -Trigger $t -Settings $s -Force | Out-Null; Start-ScheduledTask -TaskName Task1_ThuLiveKhiMoMay"
if errorlevel 1 ( echo [!] Khong dang ky duoc. Chup man hinh gui lai. & pause & exit /b 1 )
echo DA BAT. Dang thu ngay bay gio va se tu chay moi lan mo may.
pause
