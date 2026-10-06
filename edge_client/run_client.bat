@echo off
set SERVER_IP=%1
if "%SERVER_IP%"=="" set SERVER_IP=127.0.0.1
set SOURCE=%2
if "%SOURCE%"=="" set SOURCE=0

echo ==================================================
echo   EdgePPE-Vision: Client Launcher (Windows Test)
echo   Connecting to http://%SERVER_IP%:8000
echo   Video Source: %SOURCE%
echo ==================================================

python pi_client.py --server http://%SERVER_IP%:8000 --source %SOURCE% --fps 6.0
pause
