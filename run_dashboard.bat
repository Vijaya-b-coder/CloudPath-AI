@echo off
title CloudPath AI - Web Dashboard Server
echo =================================================================
echo   CloudPath AI: Cloud Movement Tracking and Rainfall Prediction
echo   Starting Web Application on http://localhost:8050/
echo =================================================================
set PYTHONUTF8=1
set PYTHONUSERBASE=%APPDATA%\Python
cd /d "d:\bengaluru.csv"
python -u dashboard_app.py --port 8050
pause

