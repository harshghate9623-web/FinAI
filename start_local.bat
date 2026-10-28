@echo off
cd /d "%~dp0"
echo Starting FinDoc AI locally...
call .\venv\Scripts\python.exe run_app.py
pause

