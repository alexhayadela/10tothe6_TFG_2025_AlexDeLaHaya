@echo off
REM Move to current script directory
cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\Activate

REM Launch VS Code in current folder
start "" code .