@echo off
setlocal
title Aura Studio

echo [Aura Studio] Starting backend engine...
start "Aura Engine" /B cmd /c "cd /d "%~dp0backend" && python main.py"

echo Waiting for backend to start...
set /a tries=0
:wait
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8766/health >nul 2>&1
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% lss 30 goto wait
echo [Aura Studio] WARNING: backend did not become healthy after 60s.
echo (First run may still be installing dependencies - check the engine output.)
:ready

echo [Aura Studio] Backend ready! Starting UI...
start "Aura UI" /B cmd /c "cd /d "%~dp0" && npm run dev"

echo.
echo ════════════════════════════════════════
echo  Aura Studio is running!
echo  Backend:  http://127.0.0.1:8766
echo  Frontend: http://localhost:1420
echo ════════════════════════════════════════
echo.
echo Press any key to stop all services...
pause >nul

echo Shutting down...
rem Window-title filtering doesn't work with "start /B", so find the
rem listeners by port instead and kill their whole process tree.
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8766" ^| findstr "LISTENING"') do taskkill /f /t /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":1420" ^| findstr "LISTENING"') do taskkill /f /t /pid %%a >nul 2>&1
echo Done.
endlocal
