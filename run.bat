@echo off
REM PotSpot one-click start for Windows. Needs Python 3.10-3.13 on PATH.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv || goto :nopython
)
echo Installing requirements (first run downloads PyTorch, about 250 MB; it takes a few minutes)...
.venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt || goto :error
if not exist .env copy .env.example .env >nul
echo.
echo PotSpot is starting at http://127.0.0.1:8002  (press Ctrl+C to stop)
start "" http://127.0.0.1:8002
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8002
goto :eof
:nopython
echo Python was not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
pause
goto :eof
:error
echo Installing requirements failed. Check your internet connection and try again.
pause
